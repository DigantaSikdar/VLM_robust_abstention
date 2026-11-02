"""
caad/eval/steer_full.py
=======================
Full-scale evaluation of inference-time activation steering (vectors from
caad.eval.steer_probe calib / calib_noimg), batched.

Views (the whole benchmark, no held-out halves: the steering vectors were built
from training-split pools that share no question or image with these sets):
  image    VizWiz real photos (split by label: vizwiz_unans / vizwiz_ans) and the
           corrupted views of GQA, VQAv2, TextVQA
  clean    clean views of GQA, VQAv2, TextVQA
  noimg    the same questions with the image withheld (no image tokens)

Configs: trained (no steering) | <dir>:single:<layer>:<lambda>:<tau>, dir in
{all, contrast, noimg, probe} (vectors steer_vectors{,_contrast,_noimg,_probe}.pt in --vectors).
tau < 0 steers every decode step; otherwise a row is steered at step t when the
entropy of its step t-1 next-token distribution exceeds tau (per-row gate).
--norm-ref DIR rescales lambda per direction so that ||lambda * v[layer]|| equals
the reference model's ||lambda * v_ref[layer]|| (same steering magnitude).
--norm-scale S multiplies lambda once more, for a model whose residual stream is on a
different scale (S = its mean ||h_layer|| / the reference model's, e.g. Gemma 3).

    python -m caad.eval.steer_full --lora <adapter> --vectors outputs/steer_full/rkl \
        --out outputs/steer_full/rkl/eval --view image --configs trained all:single:18:8:-1 \
        --shard 0 --num-shards 4
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from .steer_probe import ABSTAIN, ROOT, decoder_layers, load_model
from .tasks.registry import get_task
from .tasks.vqa_c import extract_answer
from ..data.vision import batch_images

DIRS = {"all": "", "contrast": "_contrast", "noimg": "_noimg", "probe": "_probe"}


def examples(view, prompt_format, root=ROOT, datasets=("gqa", "vqav2", "textvqa")):
    out = []
    if view in ("image", "noimg"):
        for e in get_task("vizwiz").load({"root": root, "prompt_format": prompt_format}):
            out.append(("vizwiz_unans" if e.get("category") == "unanswerable" else "vizwiz_ans", e))
    for ds in datasets:
        corrupted = view == "image"
        for e in get_task(ds).load({"root": root, "prompt_format": prompt_format, "use_corrupted": corrupted}):
            out.append((f"{ds}_{'corrupted' if corrupted else 'clean'}", e))
    return out


class Steerer:
    """Adds lam * v[l] at decode steps, only for rows whose gate is open."""

    def __init__(self, model):
        self.v, self.lam, self.mask = {}, 0.0, None
        for l, layer in enumerate(decoder_layers(model)):
            layer.register_forward_hook(self._hook(l))

    def _hook(self, l):
        def f(mod, inp, out):
            hs = out[0] if isinstance(out, tuple) else out
            if self.mask is not None and self.lam and hs.shape[1] == 1 and l in self.v:
                hs = hs + self.lam * self.v[l].to(hs.device, hs.dtype) * self.mask[:, None, None].to(hs.dtype)
                return (hs,) + tuple(out[1:]) if isinstance(out, tuple) else hs
            return out
        return f


class BatchGate:
    """Per-row entropy gate (LogitsProcessor): opens a row's steering for the next
    step when its current next-token entropy exceeds tau (tau < 0: always open)."""

    def __init__(self, steerer, tau):
        self.s, self.tau, self.history = steerer, tau, []

    def __call__(self, input_ids, scores):
        if self.tau < 0:
            m = torch.ones(scores.shape[0], dtype=torch.bool, device=scores.device)
        else:
            p = scores.float().softmax(-1)
            m = -(p * p.clamp_min(1e-12).log()).sum(-1) > self.tau
        self.s.mask = m
        self.history.append(m)
        return scores


def norms_at(vdir, tag, layer):
    p = Path(vdir) / f"steer_vectors{tag}.pt"
    return float(torch.load(p)["v"][layer].norm())


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lora", required=True)
    ap.add_argument("--vectors", required=True, help="dir with steer_vectors*.pt for this model")
    ap.add_argument("--out", required=True)
    ap.add_argument("--view", required=True, choices=["image", "clean", "noimg"])
    ap.add_argument("--configs", nargs="+", required=True)
    ap.add_argument("--norm-ref", default=None, help="vectors dir whose steering magnitude to match")
    ap.add_argument("--norm-scale", type=float, default=1.0,
                    help="extra lambda factor: residual-norm ratio to the --norm-ref model")
    ap.add_argument("--prompt-format", default="think_answer")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--max-new-tokens", type=int, default=384)
    ap.add_argument("--limit", type=int, default=None, help="smoke test: first N examples of this shard")
    ap.add_argument("--per-group", type=int, default=None, help="fixed random subset: N examples per group (seed 0)")
    ap.add_argument("--root", default=ROOT, help="question pool (e.g. the steer_calib training-split pool for validation)")
    ap.add_argument("--datasets", default="gqa,vqav2,textvqa", help="benchmarks besides VizWiz")
    ap.add_argument("--exclude", default=None, help="file of question ids to leave out (e.g. those the vectors were built from)")
    a = ap.parse_args()

    from PIL import Image
    from transformers import LogitsProcessorList
    model, proc = load_model(a.lora)
    proc.tokenizer.padding_side = "left"
    pad = proc.tokenizer.pad_token_id
    V = {d: torch.load(Path(a.vectors) / f"steer_vectors{t}.pt")["v"].cuda()
         for d, t in DIRS.items() if (Path(a.vectors) / f"steer_vectors{t}.pt").exists()}
    steer = Steerer(model)
    ex = sorted(examples(a.view, a.prompt_format, a.root, a.datasets.split(",")), key=lambda x: (x[0], str(x[1]["id"])))
    if a.exclude:
        drop = set(open(a.exclude).read().split())
        ex = [x for x in ex if str(x[1]["id"]) not in drop]
    if a.per_group:                                            # same subset for every model and config
        import random
        by = {}
        for x in ex:
            by.setdefault(x[0], []).append(x)
        ex = [x for g in sorted(by) for x in random.Random(0).sample(by[g], min(a.per_group, len(by[g])))]
    ex = ex[a.shard::a.num_shards][:a.limit]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    for cfg in a.configs:
        lam, tau = 0.0, -1.0
        if cfg != "trained":
            d, mode, layer, lam, tau = cfg.split(":")
            assert mode == "single", "steer_full supports single-layer steering"
            layer, lam, tau = int(layer), float(lam), float(tau)
            if a.norm_ref:                                     # match the reference steering magnitude
                lam *= norms_at(a.norm_ref, DIRS[d], layer) / norms_at(a.vectors, DIRS[d], layer)
            lam *= a.norm_scale
            steer.v = {layer: V[d][layer]}
        f = out / f"{a.view}__{cfg.replace(':', '_')}__shard{a.shard}of{a.num_shards}.jsonl"
        done = {json.loads(l)["id"] for l in open(f)} if f.exists() else set()
        todo = [x for x in ex if x[1]["id"] not in done]
        t0 = time.time()
        print(f"[{cfg}] view={a.view} lambda={lam:.3f} {len(todo)}/{len(ex)} to go", flush=True)
        with f.open("a") as fh:
            def run(chunk):
                """Generate for a chunk; on CUDA OOM split it in half and retry."""
                texts, imgs = [], []
                for g, e in chunk:
                    content = [{"type": "text", "text": e["prompt"]}]
                    if a.view != "noimg":
                        content.insert(0, {"type": "image"})
                        with Image.open(e["image"]) as im:
                            imgs.append([im.convert("RGB")])
                    msgs = [{"role": "user", "content": content}]
                    if e["system"]:
                        msgs.insert(0, {"role": "system", "content": [{"type": "text", "text": e["system"]}]})
                    texts.append(proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
                oom = False
                try:
                    inp = proc(text=texts, images=batch_images(proc, imgs), padding=True, return_tensors="pt").to("cuda")
                    gate, procs = None, None
                    if cfg != "trained":
                        steer.lam, steer.mask = lam, None
                        gate = BatchGate(steer, tau)
                        procs = LogitsProcessorList([gate])
                    gen = model.generate(**inp, do_sample=False, max_new_tokens=a.max_new_tokens,
                                         logits_processor=procs, pad_token_id=pad)
                except torch.OutOfMemoryError:
                    oom = True          # recover OUTSIDE the except block: the traceback pins the tensors
                if oom:
                    steer.mask = None
                    inp = gen = gate = procs = None
                    del texts, imgs
                    import gc
                    gc.collect()
                    torch.cuda.empty_cache()
                    if len(chunk) == 1:
                        raise RuntimeError(f"OOM on a single example: {chunk[0][1]['id']}")
                    h = len(chunk) // 2
                    print(f"[oom] splitting a batch of {len(chunk)}", flush=True)
                    return run(chunk[:h]) + run(chunk[h:])
                steer.mask = None
                new = gen[:, inp["input_ids"].shape[1]:]
                hist = torch.stack(gate.history, 1) if gate and gate.history else None   # (B, steps)
                res = []
                for j in range(len(chunk)):
                    n_tok = int((new[j] != pad).sum())
                    frac = float(hist[j, :n_tok].float().mean()) if hist is not None and n_tok else 0.0
                    res.append((proc.decode(new[j], skip_special_tokens=True), n_tok, frac))
                del inp, gen, new
                return res

            for i in range(0, len(todo), a.batch):
                chunk = todo[i:i + a.batch]
                for (g, e), (comp, n_tok, frac) in zip(chunk, run(chunk)):
                    ans = extract_answer(comp).lower()
                    fh.write(json.dumps({"id": e["id"], "group": g, "dataset": e["dataset"], "view": a.view,
                                         "config": cfg, "lambda": lam, "correct": bool(get_task(e["dataset"]).score(e, comp)),
                                         "abstain": any(w in ans for w in ABSTAIN), "steered_frac": frac,
                                         "gen_tokens": n_tok, "completion": comp}) + "\n")
                fh.flush()
                n = i + len(chunk)
                if (i // a.batch) % 10 == 0:
                    print(f"[{cfg}] {n}/{len(todo)} {n / (time.time() - t0):.2f} ex/s", flush=True)


if __name__ == "__main__":
    main()
