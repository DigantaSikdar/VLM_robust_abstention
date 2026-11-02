"""
caad/eval/steer_probe.py
========================
Smoke test: inference-time activation steering of a distilled model back toward
its base model, applied only where decoding is uncertain.

calib  For calibration questions (disjoint from the eval questions), take the
       trained model's own greedy answer and teacher-force it through the BASE
       model (LoRA disabled) and the TRAINED model (LoRA on) on the identical
       input. Per decoder layer l, the steering vector is the mean residual
       difference over the answer tokens:  v_l = mean_t (h_base[l,t] - h_trained[l,t]).
       lambda = 1 therefore moves a trained activation onto the base model's
       average position at that layer.

eval   Greedy decoding with the trained model. At decode step t, if the
       entropy of the step-(t-1) next-token distribution exceeds tau, add
       lambda * v_l to the residual stream output of every steered layer for the
       token being processed (tau < 0: steer every step). Scored on held-out
       VizWiz (unanswerable / answerable) and GQA-C questions.

    python -m caad.eval.steer_probe calib --lora <adapter> --out outputs/steer/rkl_sys
    python -m caad.eval.steer_probe eval  --lora <adapter> --out outputs/steer/rkl_sys \
        --configs base trained 1.0:0.5 2.0:0.5
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import time
from pathlib import Path

import torch

from ..data.vision import build_messages, process_image
from ..eval.tasks.registry import get_task
from ..eval.tasks.vqa_c import extract_answer
from ..trainers.caad import build_inputs

ROOT = "/gscratch/dsikdar/caad_data/vqa_c"
ABSTAIN = ("unanswerable", "cannot", "unable", "not possible", "no image", "not visible",
           "not provided", "not determin", "impossible")


def cal_split(a):
    """Calibration questions: the 'calib' half of the eval benchmarks (default root),
    or every question of a dedicated calibration pool (--root elsewhere)."""
    return "calib" if a.root == ROOT else "all"


def split_of(qid):
    """Deterministic calibration / eval split by question id."""
    return "calib" if int(hashlib.md5(str(qid).encode()).hexdigest(), 16) % 2 == 0 else "eval"


def in_split(qid, split):
    """split 'all' (a dedicated calibration pool) keeps every question."""
    return split == "all" or split_of(qid) == split


def clean_examples(split, n, prompt_format, seed=0):
    """Clean-image views of GQA / VQAv2 / TextVQA (same eval-split questions; the
    GQA and VQAv2 ones are drawn with the same rng as their corrupted views)."""
    rng = random.Random(seed)
    out = []
    for ds in ("gqa", "vqav2", "textvqa"):
        qs = [e for e in get_task(ds).load({"root": ROOT, "prompt_format": prompt_format, "use_corrupted": False})
              if split_of(e["id"]) == split]
        out += [(f"{ds}_clean", e) for e in rng.sample(qs, n)]
    return out


def examples(split, n_viz_each, n_gqa, prompt_format, seed=0, root=ROOT):
    """VizWiz (half labeled unanswerable, half not) + GQA-C, from one split."""
    rng = random.Random(seed)
    viz = [e for e in get_task("vizwiz").load({"root": root, "prompt_format": prompt_format})
           if in_split(e["id"], split)]
    un = [e for e in viz if e.get("category") == "unanswerable"]
    an = [e for e in viz if e.get("category") != "unanswerable"]
    corr = lambda ds: [e for e in get_task(ds).load({"root": root, "prompt_format": prompt_format,
                                                     "use_corrupted": True}) if in_split(e["id"], split)]
    out = [("viz_unans", e) for e in rng.sample(un, n_viz_each)] + \
          [("viz_ans", e) for e in rng.sample(an, n_viz_each)] + \
          [("gqa_c", e) for e in rng.sample(corr("gqa"), n_gqa)] + \
          [("vqav2_c", e) for e in rng.sample(corr("vqav2"), n_gqa)]
    return out


# base model of the family being analysed (InternVL: CAAD_BASE_MODEL=OpenGVLab/InternVL3_5-4B-HF)
BASE_MODEL = os.environ.get("CAAD_BASE_MODEL", "Qwen/Qwen3-VL-4B-Instruct")


def load_policy(spec=None, device="cuda"):
    """The model to analyse: None -> the base model; a dir with adapter_config.json ->
    base + that LoRA adapter (PeftModel); any other dir -> a full fine-tuned checkpoint."""
    from transformers import AutoModelForImageTextToText
    if spec and not (Path(spec) / "adapter_config.json").exists():
        return AutoModelForImageTextToText.from_pretrained(
            spec, dtype=torch.bfloat16, attn_implementation="sdpa", device_map=device).eval()
    base = AutoModelForImageTextToText.from_pretrained(
        BASE_MODEL, dtype=torch.bfloat16, attn_implementation="sdpa", device_map=device).eval()
    if not spec:
        return base
    from peft import PeftModel
    return PeftModel.from_pretrained(base, spec).eval()


def load_model(lora):
    """(model, processor) for a LoRA adapter dir or a full fine-tuned checkpoint dir."""
    from transformers import AutoProcessor
    return load_policy(lora), AutoProcessor.from_pretrained(BASE_MODEL)


def decoder_layers(model):
    for name, mod in model.named_modules():
        if name.endswith("language_model.layers") and isinstance(mod, torch.nn.ModuleList):
            return mod
    raise RuntimeError("decoder layers not found")


def text_inputs(proc, system, prompt, completion, device):
    """No-image prompt (system + question, no image tokens) + completion + end-of-turn.
    Returns the batch and the prompt length."""
    msgs = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    if system:
        msgs.insert(0, {"role": "system", "content": [{"type": "text", "text": system}]})
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    tok = proc.tokenizer
    p_ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids
    c_ids = tok(completion + "<|im_end|>", return_tensors="pt", add_special_tokens=False).input_ids
    ids = torch.cat([p_ids, c_ids], 1).to(device)
    return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}, p_ids.shape[1]


@torch.inference_mode()
def cmd_calib_noimg(a):
    """Contrastive pairs from NO-IMAGE responses. A calibration question asked without
    its image is kept when the base model refuses and the trained model answers anyway.
    Both texts go through the TRAINED model (the model being steered):
        positive = no-image prompt + base model's refusal
        negative = no-image prompt + trained model's answer
        v_l      = mean_pairs [ mean_t h_l(positive) - mean_t h_l(negative) ]"""
    model, proc = load_model(a.lora)
    load = lambda f: {(r["dataset"], r["question_id"]): r["completion"] for r in map(json.loads, open(f))
                      if r["view"] == "noimage" and r["completion"]}
    base_r, tr_r = load(a.base_noimg), load(a.trained_noimg)
    said = lambda c: any(w in extract_answer(c).lower() for w in ABSTAIN)
    rng = random.Random(1)
    ex = []
    for ds in ("vizwiz", "gqa", "vqav2", "textvqa"):
        qs = [e for e in get_task(ds).load({"root": a.root, "prompt_format": a.prompt_format})
              if in_split(e["id"], cal_split(a)) and (ds, e["id"]) in base_r and (ds, e["id"]) in tr_r
              and said(base_r[(ds, e["id"])]) and not said(tr_r[(ds, e["id"])])]
        print(f"[calib-noimg] {ds}: {len(qs)} pairs (base refuses, trained answers)", flush=True)
        ex += rng.sample(qs, min(a.n_per_ds, len(qs)))
    L = len(decoder_layers(model))
    acc = torch.zeros(L, model.config.text_config.hidden_size, dtype=torch.float64, device="cuda")
    norm_t = torch.zeros(L, dtype=torch.float64, device="cuda")
    t0 = time.time()

    def mean_h(e, resp):
        batch, plen = text_inputs(proc, e["system"], e["prompt"], resp, "cuda")
        sl = slice(plen - 1, batch["input_ids"].shape[1] - 1)      # positions that predict the response
        hs = model(**batch, output_hidden_states=True).hidden_states   # trained model (LoRA on)
        return torch.stack([hs[l + 1][0, sl].double().mean(0) for l in range(L)])

    for i, e in enumerate(ex):
        hp = mean_h(e, base_r[(e["dataset"], e["id"])])
        hn = mean_h(e, tr_r[(e["dataset"], e["id"])])
        acc += hp - hn
        norm_t += hn.norm(dim=-1)
        if (i + 1) % 100 == 0:
            print(f"[calib-noimg] {i + 1}/{len(ex)} {time.time() - t0:.0f}s", flush=True)
    v = (acc / len(ex)).float().cpu()
    stats = {"n_pairs": len(ex), "v_norm": v.norm(dim=-1).tolist(), "h_norm": (norm_t / len(ex)).tolist()}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"v": v, "stats": stats}, out / "steer_vectors_noimg.pt")
    (out / "calib_stats_noimg.json").write_text(json.dumps(stats, indent=2))
    print("relative |v|/|h| per layer:", [round(x / y, 3) for x, y in zip(stats["v_norm"], stats["h_norm"])],
          flush=True)


@torch.inference_mode()
def cmd_calib(a):
    model, proc = load_model(a.lora)
    from peft import PeftModel
    base_model = None if isinstance(model, PeftModel) else load_policy(None)   # full FT: load the base too
    res = {}
    for f in a.results:
        for r in map(json.loads, open(f)):
            if r["view"] in ("corrupted", "real"):
                res[(r["dataset"], r["question_id"])] = r["completion"]
    if a.contrast:
        # targeted direction: calibration VizWiz questions labeled unanswerable on which the
        # base model abstains and the trained model answers anyway
        said = lambda c: "unanswerable" in c.lower().split("<answer>")[-1]
        base = {(r["dataset"], r["question_id"]): r["completion"] for f in a.base_results
                for r in map(json.loads, open(f)) if r["view"] == "real"}
        ex = [("viz_unans", e) for e in get_task("vizwiz").load({"root": a.root, "prompt_format": a.prompt_format})
              if in_split(e["id"], cal_split(a)) and e.get("category") == "unanswerable"
              and (e["dataset"], e["id"]) in base and (e["dataset"], e["id"]) in res
              and said(base[(e["dataset"], e["id"])]) and not said(res[(e["dataset"], e["id"])])]
    else:
        ex = examples(cal_split(a), a.n_viz_each, a.n_gqa, a.prompt_format, seed=1, root=a.root)
    ex = [(g, e) for g, e in ex if res.get((e["dataset"], e["id"]))]
    L = len(decoder_layers(model))
    acc = torch.zeros(L, model.config.text_config.hidden_size, dtype=torch.float64, device="cuda")
    norm_t = torch.zeros(L, dtype=torch.float64, device="cuda")
    n_tok, t0 = 0, time.time()
    for i, (g, e) in enumerate(ex):
        batch, cmask, plen = build_inputs(proc, e["system"], e["prompt"], e["image"],
                                          res[(e["dataset"], e["id"])], "cuda")
        hs_t = model(**batch, output_hidden_states=True).hidden_states            # trained (LoRA on)
        if base_model is None:
            with model.disable_adapter():
                hs_b = model(**batch, output_hidden_states=True).hidden_states    # base (LoRA off)
        else:                                                                     # full FT: separate base
            hs_b = base_model(**{k: v.to(base_model.device) if torch.is_tensor(v) else v
                                 for k, v in batch.items()}, output_hidden_states=True).hidden_states
            hs_b = tuple(h.to(hs_t[0].device) for h in hs_b)
        sl = slice(plen - 1, batch["input_ids"].shape[1] - 1)   # positions that predict answer tokens
        for l in range(L):                                       # hidden_states[l+1] = output of layer l
            d = (hs_b[l + 1][0, sl] - hs_t[l + 1][0, sl]).double()
            acc[l] += d.sum(0)
            norm_t[l] += hs_t[l + 1][0, sl].double().norm(dim=-1).sum()
        n_tok += sl.stop - sl.start
        if (i + 1) % 50 == 0:
            print(f"[calib] {i + 1}/{len(ex)} {time.time() - t0:.0f}s", flush=True)
    v = (acc / n_tok).float().cpu()
    stats = {"n_examples": len(ex), "n_tokens": n_tok,
             "v_norm": v.norm(dim=-1).tolist(), "h_trained_norm": (norm_t / n_tok).tolist()}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"v": v, "stats": stats}, out / f"steer_vectors{a.tag}.pt")
    (out / f"calib_stats{a.tag}.json").write_text(json.dumps(stats, indent=2))
    print("relative |v|/|h| per layer:",
          [round(x / y, 3) for x, y in zip(stats["v_norm"], stats["h_trained_norm"])], flush=True)


class Steerer:
    """Forward hooks that add lam * v[l] at decode steps while `active` is set."""

    def __init__(self, model, v, layers):
        """v: dict layer -> vector actually added at that layer (times lam)."""
        self.v, self.lam, self.active = v, 0.0, False
        self.handles = [decoder_layers(model)[l].register_forward_hook(self._hook(l)) for l in layers]

    def _hook(self, l):
        def f(mod, inp, out):
            hs = out[0] if isinstance(out, tuple) else out
            if self.active and self.lam and hs.shape[1] == 1 and l in self.v:   # decode step only
                hs = hs + self.lam * self.v[l].to(hs.device, hs.dtype)
                return (hs,) + tuple(out[1:]) if isinstance(out, tuple) else hs
            return out
        return f


class EntropyGate:
    """LogitsProcessor: after each step, turn steering on for the next step if the
    current next-token entropy exceeds tau (tau < 0: always on)."""

    def __init__(self, steerer, tau):
        self.s, self.tau, self.on, self.n = steerer, tau, 0, 0

    def __call__(self, input_ids, scores):
        p = scores.float().softmax(-1)
        H = float(-(p * p.clamp_min(1e-12).log()).sum(-1)[0])
        self.s.active = self.tau < 0 or H > self.tau
        self.on += self.s.active
        self.n += 1
        return scores


@torch.inference_mode()
def cmd_eval(a):
    from transformers import LogitsProcessorList
    model, proc = load_model(a.lora)
    V = {tag: torch.load(Path(a.out) / f"steer_vectors{tag}.pt")["v"].cuda() for tag in ("", "_contrast", "_noimg")
         if (Path(a.out) / f"steer_vectors{tag}.pt").exists()}
    L = next(iter(V.values())).shape[0]
    steer = Steerer(model, {}, list(range(L)))

    def plan(dirtag, mode, layer):
        """Per-layer vectors to add. incremental: each layer's own increment of the
        cumulative base-minus-trained gap (lam = 1 maps the trained mean onto the base
        mean at the last steered layer); single: the cumulative gap at one layer."""
        v = V[dirtag]
        if mode == "single":
            return {layer: v[layer]}
        lo, hi = a.first_layer, L - a.skip_last
        return {l: v[l] - (v[l - 1] if l > lo else 0) for l in range(lo, hi)}
    ex = clean_examples("eval", a.n_gqa, a.prompt_format, seed=2) if a.clean else \
        examples("eval", a.n_viz_each, a.n_gqa, a.prompt_format, seed=2)
    out = Path(a.out)
    for cfg in a.configs:
        f = out / f"eval_{cfg.replace(':', '_')}.jsonl"
        done = {json.loads(l)["id"] for l in open(f)} if f.exists() else set()
        t0 = time.time()
        with f.open("a") as fh:
            for i, (g, e) in enumerate(ex):
                if e["id"] in done:
                    continue
                if a.no_image:           # same prompt, image withheld (no image tokens at all)
                    msgs = [{"role": "user", "content": [{"type": "text", "text": e["prompt"]}]}]
                    if e["system"]:
                        msgs.insert(0, {"role": "system", "content": [{"type": "text", "text": e["system"]}]})
                    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
                    inp = proc(text=[text], return_tensors="pt").to("cuda")
                else:
                    msgs = build_messages(e["image"], e["prompt"], e["system"])
                    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
                    inp = process_image(proc, text, msgs).to("cuda")
                gate = None
                if cfg in ("base", "trained"):
                    steer.lam, steer.active = 0.0, False
                    procs = None
                else:          # <dir: all|contrast>:<mode: inc|single>:<layer>:<lam>:<tau>
                    d, mode, layer, lam, tau = cfg.split(":")
                    steer.v = plan({"all": "", "contrast": "_contrast", "noimg": "_noimg"}[d],
                                   "single" if mode == "single" else "inc",
                                   int(layer))
                    lam, tau = float(lam), float(tau)
                    steer.lam, steer.active = lam, False
                    gate = EntropyGate(steer, tau)
                    procs = LogitsProcessorList([gate])
                kw = dict(do_sample=False, max_new_tokens=a.max_new_tokens, logits_processor=procs)
                if cfg == "base":
                    with model.disable_adapter():
                        gen = model.generate(**inp, **kw)
                else:
                    gen = model.generate(**inp, **kw)
                steer.active = False
                comp = proc.decode(gen[0, inp["input_ids"].shape[1]:], skip_special_tokens=True)
                task = get_task(e["dataset"])
                ans = extract_answer(comp).lower()
                fh.write(json.dumps({"id": e["id"], "group": g, "dataset": e["dataset"], "config": cfg,
                                     "correct": bool(task.score(e, comp)), "completion": comp,
                                     "abstain": any(w in ans for w in ABSTAIN), "no_image": a.no_image,
                                     "steered_frac": gate.on / max(gate.n, 1) if gate else 0.0,
                                     "gen_tokens": int(gen.shape[1] - inp["input_ids"].shape[1])}) + "\n")
                fh.flush()
                if (i + 1) % 50 == 0:
                    print(f"[eval {cfg}] {i + 1}/{len(ex)} {time.time() - t0:.0f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("calib_noimg")
    s.add_argument("--root", default=ROOT, help="calibration pool (default: calib half of the eval data)")
    s.add_argument("--lora", required=True, help="LoRA adapter dir or full fine-tuned checkpoint dir")
    s.add_argument("--out", required=True)
    s.add_argument("--prompt-format", default="think_answer")
    s.add_argument("--base-noimg", required=True, help="base model's no-image results.jsonl")
    s.add_argument("--trained-noimg", required=True, help="trained model's no-image results.jsonl")
    s.add_argument("--n-per-ds", type=int, default=150)
    for name in ("calib", "eval"):
        s = sub.add_parser(name)
        s.add_argument("--lora", required=True)
        s.add_argument("--out", required=True)
        s.add_argument("--prompt-format", default="think_answer")
        s.add_argument("--n-viz-each", type=int, default=150)
        s.add_argument("--n-gqa", type=int, default=150)
    sub.choices["calib"].add_argument("--results", nargs="+", required=True,
                                      help="the trained model's eval results.jsonl (its greedy answers)")
    sub.choices["calib"].add_argument("--contrast", action="store_true",
                                      help="only unanswerable VizWiz where base abstains and trained answers")
    sub.choices["calib"].add_argument("--base-results", nargs="*", default=[])
    sub.choices["calib"].add_argument("--tag", default="")
    sub.choices["calib"].add_argument("--root", default=ROOT,
                                      help="calibration pool (default: calib half of the eval data)")
    e = sub.choices["eval"]
    e.add_argument("--configs", nargs="+", required=True,
                   help="base | trained | <all|contrast>:<inc|single>:<layer>:<lambda>:<tau>")
    e.add_argument("--first-layer", type=int, default=4)
    e.add_argument("--skip-last", type=int, default=2)
    e.add_argument("--max-new-tokens", type=int, default=384)
    e.add_argument("--no-image", action="store_true", help="withhold the image (text-only requests)")
    e.add_argument("--clean", action="store_true", help="clean-image GQA / VQAv2 / TextVQA instead")
    a = ap.parse_args()
    {"calib": cmd_calib, "calib_noimg": cmd_calib_noimg, "eval": cmd_eval}[a.cmd](a)


if __name__ == "__main__":
    main()
