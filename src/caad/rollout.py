"""
caad/rollout.py
===============
Rollout engine: samples the student's completion(s) y ~ pi_student(. | q, V_corrupted)
on the CORRUPTED image. Three backends (rollout.backend):

  mock   canned text, no model — pipeline smoke tests / CI.
  hf     generate with the LIVE training model (.generate()). Single-/multi-rank,
         always-fresh weights (no sync needed). Slower; good to start.
  vllm   DEDICATED vLLM server on a reserved GPU (your production topology:
         GPU 0 = server, the rest = training). The server holds a separate copy
         of the weights, so it must be re-synced as the student trains:
           sync_mode=lora  save the student adapter -> vLLM hot-loads it
                           (vLLM /v1/load_lora_adapter). Cheap, robust.
           sync_mode=full  save the student's full weights -> vLLM reloads them
                           in place (dev-mode /collective_rpc reload_weights).
         Each training rank is an independent HTTP client to the one server, so
         there is no cross-rank broadcast — the server batches concurrent requests.

VALIDATE ON HARDWARE: the multimodal image field in chat() and the LoRA-admin
endpoint names depend on your installed vLLM version — both are flagged below.
"""

from __future__ import annotations

import logging
from pathlib import Path

from .data.vision import build_messages, process_image
from .eval.tasks.vqa_c import PROMPT_FORMATS

_LOG = logging.getLogger("caad")


class Completion(str):
    """Rollout text that also carries the token ids the policy actually sampled
    (``None`` when unknown). Re-tokenizing the text can merge tokens across the
    sampled boundaries (InternVL: sampled ">", "c" -> re-tokenized ">c"), so the
    trainer would score a sequence the student never generated."""
    token_ids = None

    def __new__(cls, text, token_ids=None):
        obj = super().__new__(cls, text)
        obj.token_ids = token_ids
        return obj


class RolloutEngine:
    def __init__(self, cfg, processor, model=None):
        self.cfg = cfg
        self.processor = processor
        self.sampling = cfg["rollout"]
        self.backend = self.sampling.get("backend", "vllm")
        self.n = self.sampling.get("num_generations", 1)
        self.model = model                 # hf backend uses the live model
        self._client = None                # VLLMServerClient (vllm backend)
        self._sync_count = 0
        self._active_adapter = None        # current LoRA name served by vLLM
        self._warned = False
        d = cfg["data"]
        fmt_system, self.user_tpl = PROMPT_FORMATS[d.get("prompt_format", "inline")]
        self.system = d.get("system_prompt") or fmt_system
        # sync_mode: auto -> lora if LoRA training else full
        sm = self.sampling.get("sync_mode", "auto")
        self.sync_mode = ("lora" if cfg["lora"]["enabled"] else "full") if sm == "auto" else sm

    # ------------------------------------------------------------------ #
    def generate(self, sample):
        if self.backend == "mock":
            txt = self.sampling.get("mock_text", "Let me reason. The answer is A.")
            return [txt] * self.n
        if self.backend == "hf":
            return self._generate_hf(sample)
        return self._generate_vllm(sample)

    def sync_weights(self, model, accelerator=None):
        """Make the rollout policy reflect the latest student weights.
        mock: no-op. hf: rebind the live model. vllm: push to the server."""
        if self.backend == "mock":
            return
        if self.backend == "hf":
            self.model = model
            return
        self._last_sync = (model, accelerator)
        self._sync_vllm(model, accelerator)

    # ------------------------------------------------------------------ #
    # shared prompt construction
    # ------------------------------------------------------------------ #
    def _messages(self, sample):
        return build_messages(sample["corrupted_image"],
                              self.user_tpl.format(question=sample["question"]),
                              self.system, self.cfg["model"].get("max_pixels"))

    # ------------------------------------------------------------------ #
    # hf backend (live model)
    # ------------------------------------------------------------------ #
    def _generate_hf(self, sample):
        import torch
        model = self.model
        if model is None:
            raise RuntimeError("hf backend needs RolloutEngine(cfg, processor, model=...)")
        messages = self._messages(sample)
        text = self.processor.apply_chat_template(messages, tokenize=False,
                                                  add_generation_prompt=True)
        inputs = process_image(self.processor, text, messages).to(model.device)
        temp = self.sampling["temperature"]
        with torch.no_grad():
            out = model.generate(
                **inputs, do_sample=temp > 0,
                temperature=temp if temp > 0 else None,
                top_p=self.sampling.get("top_p", 1.0),
                max_new_tokens=self.sampling["max_completion_length"],
                num_return_sequences=self.n)
        gen = out[:, inputs["input_ids"].shape[1]:]
        texts = self.processor.tokenizer.batch_decode(gen, skip_special_tokens=True)
        if not self.sampling.get("token_ids", False):
            return texts
        pad = self.processor.tokenizer.pad_token_id
        return [Completion(t, [i for i in g.tolist() if i != pad]) for t, g in zip(texts, gen)]

    # ------------------------------------------------------------------ #
    # vllm backend (dedicated server)
    # ------------------------------------------------------------------ #
    def _ensure_client(self):
        if self._client is None:
            srv = self.sampling["server"]
            self._client = VLLMServerClient(
                base_url=f"http://{srv.get('host', '127.0.0.1')}:{srv.get('port', 8000)}",
                base_model=self.cfg["model"]["name"])
            self._client.wait_ready(timeout=srv.get("startup_timeout", 600))
        return self._client

    def _generate_vllm(self, sample):
        import requests
        client = self._ensure_client()
        served = self._active_adapter or self.cfg["model"]["name"]
        for attempt in range(3):
            try:
                return client.chat(self._messages(sample), model=served, n=self.n,
                                   temperature=self.sampling["temperature"],
                                   top_p=self.sampling.get("top_p", 1.0),
                                   max_tokens=self.sampling["max_completion_length"],
                                   stop=self.sampling.get("stop"),
                                   token_ids=self.sampling.get("token_ids", False))
            except (requests.HTTPError, requests.ConnectionError) as err:
                # full-FT only: a dead vLLM engine (e.g. a rare "Encoder cache miss"
                # after weight reloads) is replaced instead of ending the run
                if self.sync_mode != "full" or attempt == 2 or not hasattr(self, "_last_sync"):
                    raise
                _LOG.warning("vLLM request failed (%s); restarting the server (attempt %d)",
                             str(err)[:200], attempt + 1)
                self._restart_server()

    def _restart_server(self):
        """Kill the (dead) vLLM server on our port, start a fresh one on the
        server GPU with the job script's flags, wait until healthy, and push the
        CURRENT student weights (a fresh server holds the base model)."""
        import atexit
        import os
        import subprocess
        import time
        srv, model = self.sampling["server"], self.cfg["model"]["name"]
        port = int(srv.get("port", 8000))
        vis = os.environ.get("CUDA_VISIBLE_DEVICES", "1,0").split(",")
        gpu = vis[1] if len(vis) > 1 else "0"            # the server GPU (trainer runs with 1,0)
        mem = float(os.environ.get("VLLM_MEM", "0.55"))
        _free_server_gpu(model, port, gpu, need_frac=mem)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu,
                   VLLM_SERVER_DEV_MODE="1", VLLM_USE_FLASHINFER_SAMPLER="0")
        n = getattr(self, "_restarts", 0) + 1
        self._restarts = n
        log = open(f"/gscratch/dsikdar/logs/vllm-server-{os.environ.get('SLURM_JOB_ID', 'local')}"
                   f"-restart{n}.log", "w")
        cmd = [os.environ.get("VLLM_BIN", "/gscratch/dsikdar/vllm_env/bin/vllm"), "serve", model,
               "--port", str(port), "--dtype", "bfloat16", "--seed", "0",
               "--gpu-memory-utilization", os.environ.get("VLLM_MEM", "0.55"),
               "--max-model-len", "18432", "--limit-mm-per-prompt", '{"image": 1}',
               "--allowed-local-media-path", "/gscratch/dsikdar/caad_data"]
        proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)

        def _stop(pgid=proc.pid):                        # never outlive the trainer (whole process group)
            import signal
            try:
                os.killpg(pgid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        atexit.register(_stop)
        t0 = time.time()
        self._ensure_client().wait_ready(timeout=srv.get("startup_timeout", 900))
        model_obj, acc = self._last_sync
        self._sync_full(model_obj, acc)
        _LOG.warning("vLLM server restarted (#%d, pid %d) and re-synced in %.0fs", n, proc.pid, time.time() - t0)

    def _sync_vllm(self, model, accelerator):
        if self.sync_mode == "lora":
            self._sync_lora(model, accelerator)
        else:
            self._sync_full(model, accelerator)
        self._sync_count += 1

    def _sync_lora(self, model, accelerator):
        """Save the student adapter and hot-load it into the vLLM server. Only the
        main process writes + registers; a barrier ensures every rank sees the new
        adapter before its next generate(). The adapter name is derived from the
        sync counter, so all ranks agree without a broadcast."""
        name = f"student-{self._sync_count}"
        out_root = Path(self.sampling["server"].get("adapter_dir", "outputs/_vllm_adapters"))
        adir = out_root / name
        is_main = accelerator is None or accelerator.is_main_process
        if is_main:
            core = accelerator.unwrap_model(model) if accelerator else model
            adir.mkdir(parents=True, exist_ok=True)
            core.save_pretrained(str(adir), selected_adapters=["student"])  # student only
            # PEFT writes a non-"default" adapter into a subfolder named after it
            path = adir / "student" if (adir / "student" / "adapter_config.json").exists() else adir
            client = self._ensure_client()
            client.load_lora(name, str(path.resolve()))
            if self._active_adapter:                # keep the server's adapter set small
                client.unload_lora(self._active_adapter)
                prev = out_root / self._active_adapter
                if prev.exists():
                    import shutil
                    shutil.rmtree(prev, ignore_errors=True)
        if accelerator is not None:
            accelerator.wait_for_everyone()
        self._active_adapter = name

    def _sync_full(self, model, accelerator):
        """Full-FT weight push: write the student's weights (HF safetensors, same
        names as the hub checkpoint) to a node-local directory, then ask the vLLM
        server to reload them in place (vLLM's worker ``reload_weights`` through the
        dev-mode ``/collective_rpc`` route; the server must run with
        VLLM_SERVER_DEV_MODE=1) and drop its prefix / multimodal caches. The served
        model name does not change."""
        import os
        import time as _t
        is_main = accelerator is None or accelerator.is_main_process
        if is_main:
            srv = self.sampling["server"]
            wdir = Path(srv.get("weights_dir") or
                        f"/tmp/caad_sync_{os.environ.get('SLURM_JOB_ID', os.getpid())}")
            wdir.mkdir(parents=True, exist_ok=True)
            t0 = _t.time()
            core = accelerator.unwrap_model(model) if accelerator else model
            core.save_pretrained(str(wdir), safe_serialization=True, max_shard_size="5GB")
            t1 = _t.time()
            self._ensure_client().reload_weights(str(wdir.resolve()))
            _LOG.info("full weight sync #%d: save %.1fs, vLLM reload %.1fs",
                      self._sync_count, t1 - t0, _t.time() - t1)
        if accelerator is not None:
            accelerator.wait_for_everyone()


def _free_server_gpu(model, port, gpu, need_frac, timeout=300):
    """Kill whatever is left of the vLLM server on ``gpu``: the ``vllm serve``
    process tree for our port AND any other process holding that GPU (an engine
    subprocess orphaned by a crash survives its API server), except this
    trainer, which keeps the frozen teacher there. Then wait until the GPU has
    ``need_frac`` of its memory free."""
    import os
    import signal
    import subprocess
    import time

    def sh(*cmd):
        return subprocess.run(cmd, capture_output=True, text=True).stdout

    me = os.getpid()
    pids = {int(x) for x in sh("pgrep", "-f", f"serve {model} --port {port}").split()}
    frontier = set(pids)
    while frontier:                                       # all descendants
        kids = {int(x) for p in frontier for x in sh("pgrep", "-P", str(p)).split()}
        frontier = kids - pids
        pids |= kids
    uuid = next((l.split(",")[1].strip() for l in sh("nvidia-smi", "--query-gpu=index,uuid",
                 "--format=csv,noheader").splitlines() if l.split(",")[0].strip() == gpu), None)
    if uuid:
        for l in sh("nvidia-smi", "--query-compute-apps=pid,gpu_uuid", "--format=csv,noheader").splitlines():
            pid, u = [x.strip() for x in l.split(",")[:2]]
            if u == uuid and pid.isdigit():
                pids.add(int(pid))
    pids.discard(me)
    _LOG.warning("freeing server GPU %s: killing %s", gpu, sorted(pids))
    for sig in (signal.SIGTERM, signal.SIGKILL):
        for pid in pids:
            try:
                os.kill(pid, sig)
            except (ProcessLookupError, PermissionError):
                pass
        time.sleep(10)
    deadline = time.time() + timeout
    while time.time() < deadline:                        # wait for the memory to be released
        q = sh("nvidia-smi", "-i", gpu, "--query-gpu=memory.free,memory.total", "--format=csv,noheader,nounits")
        try:
            free, total = (float(x) for x in q.split(","))
            if free >= need_frac * total + 1024:
                return
        except ValueError:
            pass
        time.sleep(5)
    _LOG.warning("server GPU %s still short of memory after %ds; trying anyway", gpu, timeout)


class VLLMServerClient:
    """Thin HTTP client to a vLLM OpenAI-compatible server.
    Endpoint shapes (`/v1/chat/completions`, `/v1/load_lora_adapter`, `/health`)
    follow vLLM's documented API but VARY BY VERSION — validate against yours."""

    def __init__(self, base_url, base_model):
        self.base_url = base_url.rstrip("/")
        self.base_model = base_model

    def wait_ready(self, timeout=600, interval=5):
        import time
        import requests
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                if requests.get(self.base_url + "/health", timeout=5).status_code == 200:
                    return
            except requests.RequestException:
                pass
            time.sleep(interval)
        raise RuntimeError(f"vLLM server not ready at {self.base_url} after {timeout}s")

    def load_lora(self, name, path):
        """Hot-load a LoRA adapter. Requires the server started with
        VLLM_ALLOW_RUNTIME_LORA_UPDATING=1 and --enable-lora."""
        import requests
        r = requests.post(self.base_url + "/v1/load_lora_adapter",
                          json={"lora_name": name, "lora_path": path}, timeout=120)
        # already-loaded is fine; surface anything else
        if r.status_code >= 400 and "already" not in r.text.lower():
            r.raise_for_status()

    def unload_lora(self, name):
        import requests
        try:
            requests.post(self.base_url + "/v1/unload_lora_adapter",
                          json={"lora_name": name}, timeout=60)
        except requests.RequestException:
            pass                                    # best effort; a stale adapter is harmless

    def reload_weights(self, path):
        """Reload the served model's weights from a local HF checkpoint dir
        (dev-mode routes; server started with VLLM_SERVER_DEV_MODE=1)."""
        import requests
        r = requests.post(self.base_url + "/collective_rpc",
                          json={"method": "reload_weights", "kwargs": {"weights_path": path},
                                "timeout": 900}, timeout=960)
        r.raise_for_status()
        requests.post(self.base_url + "/reset_prefix_cache", timeout=120).raise_for_status()
        # A full-FT run died at step 3414 with "Encoder cache miss" after ~340
        # reloads. reload_weights resets caches inside the worker only, so we
        # re-sync every multimodal cache here, between requests (nothing in
        # flight): /reset_mm_cache = API-server processor cache + engine receiver.
        requests.post(self.base_url + "/reset_mm_cache", timeout=120).raise_for_status()
        # reload_weights clears only the WORKER's encoder cache; the scheduler still
        # lists those encodings as cached and can schedule a reuse the worker no
        # longer has ("Encoder cache miss"). This route resets scheduler + worker.
        requests.post(self.base_url + "/reset_encoder_cache", timeout=120).raise_for_status()

    def chat(self, messages, model, n, temperature, top_p, max_tokens, stop=None, token_ids=False):
        import requests
        # SEAM: multimodal content shape can vary by vLLM version; the standard is
        # {"type": "image_url", "image_url": {"url": "file:///abs/path.png"}}.
        payload_messages = _to_openai_messages(messages)
        r = requests.post(
            self.base_url + "/v1/chat/completions",
            json={"model": model, "messages": payload_messages, "n": n,
                  "temperature": temperature, "top_p": top_p, "max_tokens": max_tokens,
                  **({"stop": list(stop), "include_stop_str_in_output": True} if stop else {}),
                  **({"return_token_ids": True} if token_ids else {})},
            timeout=600)
        r.raise_for_status()
        if not token_ids:
            return [c["message"]["content"] for c in r.json()["choices"]]
        return [Completion(c["message"]["content"], c.get("token_ids")) for c in r.json()["choices"]]


def _to_openai_messages(messages):
    """Convert our internal message format to OpenAI chat content. The image item
    becomes a file:// image_url — adjust to your vLLM version if it expects a
    different multimodal field (e.g. base64). file:// URLs need the server
    started with --allowed-local-media-path."""
    out = []
    for m in messages:
        if isinstance(m["content"], str):
            out.append(m)
            continue
        parts = []
        for c in m["content"]:
            if c["type"] == "image":
                url = c["image"]
                if "://" not in str(url):
                    url = "file://" + str(Path(url).resolve())
                parts.append({"type": "image_url", "image_url": {"url": url}})
            elif c["type"] == "text":
                parts.append({"type": "text", "text": c["text"]})
        out.append({"role": m["role"], "content": parts})
    return out
