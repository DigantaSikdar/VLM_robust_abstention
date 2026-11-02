"""
caad/eval/probe_switch.py
=========================
Alt 1 baseline (probe-triggered refusal) on the full benchmark, on exactly the items
and prompts of caad.eval.steer_full, so it sits next to the steering configs.

extract   One forward pass per question (no generation): the last-prompt-token hidden
          state at --layers, for one steer_full view (image | clean | noimg) and shard.
report    Fit the answerability probe on <probe-dir>/vizwiz_train.npz (same fit as the
          VizWiz-only probe baseline: layer 24, standardized, L2 0.1), score every item,
          and replace the model's own answer (steer_full "trained" outputs) with
          "unanswerable" when the probe logit > threshold. Prints accuracy and refusal
          rate per group for the model, each probe threshold and each steering config.

    python -m caad.eval.probe_switch extract --lora <ckpt> --view image --out outputs/probe_switch/frkl
    python -m caad.eval.probe_switch report --tag frkl --probe-dir outputs/probe_fullft/frkl \
        --feats outputs/probe_switch/frkl --steer outputs/steer_full/frkl/eval
"""

from __future__ import annotations

import argparse
import glob
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from .probe_answerability import fit_lr, score
from .steer_full import examples
from .steer_probe import ABSTAIN, decoder_layers, load_model
from .tasks.registry import get_task
from .tasks.vqa_c import extract_answer
from ..data.vision import batch_images

REF = "<answer>unanswerable</answer>"


@torch.inference_mode()
def cmd_extract(a):
    from PIL import Image
    model, proc = load_model(a.lora)
    proc.tokenizer.padding_side = "left"
    grab, layers = {}, decoder_layers(model)
    for l in a.layers:
        layers[l - 1].register_forward_hook(          # layer l = output of decoder block l-1 (0 = embeddings)
            lambda mod, inp, out, l=l: grab.__setitem__(l, (out[0] if isinstance(out, tuple) else out)[:, -1, :].float()))
    ex = sorted(examples(a.view, a.prompt_format), key=lambda x: (x[0], str(x[1]["id"])))
    ex = ex[a.shard::a.num_shards]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    f = out / f"{a.view}__shard{a.shard}of{a.num_shards}.npz"
    if f.exists():
        return
    feats, meta, t0 = [], [], time.time()
    for i in range(0, len(ex), a.batch):
        chunk, texts, imgs = ex[i:i + a.batch], [], []
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
        inp = proc(text=texts, images=batch_images(proc, imgs), padding=True, return_tensors="pt").to("cuda")
        grab.clear()
        model(**inp, logits_to_keep=1)
        feats.append(torch.stack([grab[l] for l in a.layers], 1).cpu().numpy())
        meta += [{"id": str(e["id"]), "group": g} for g, e in chunk]
        if (i // a.batch) % 100 == 0:
            print(f"[{a.view} {a.shard}] {i + len(chunk)}/{len(ex)} {time.time() - t0:.0f}s", flush=True)
    np.savez(f, X=np.concatenate(feats), layers=np.array(a.layers), meta=json.dumps(meta))


def cmd_report(a):
    z = np.load(Path(a.probe_dir) / "vizwiz_train.npz", mmap_mode="r")
    Xtr = np.asarray(z["X"][:, a.layer], np.float32)
    ytr = np.array([m["unanswerable"] for m in json.loads(str(z["meta"]))])
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    w, b = fit_lr((Xtr - mu) / sd, ytr, a.l2)

    logit = {}                                         # (view, id) -> probe logit
    for f in sorted(glob.glob(f"{a.feats}/*__shard*.npz")):
        z = np.load(f)
        col = list(z["layers"]).index(a.layer)
        s = np.asarray(score((z["X"][:, col] - mu) / sd, w, b))
        view = Path(f).name.split("__")[0]
        for m, v in zip(json.loads(str(z["meta"])), s):
            logit[(view, m["id"])] = float(v)

    tasks, ex = {}, {}
    for view in ("image", "clean", "noimg"):
        for g, e in examples(view, a.prompt_format):
            ex[(view, str(e["id"]))] = e
    rows = defaultdict(lambda: [0, 0, 0])              # (view, group, method) -> n, correct, refused
    for f in sorted(glob.glob(f"{a.steer}/*__shard*.jsonl")):
        view, cfg = Path(f).name.split("__")[:2]
        for r in map(json.loads, open(f)):
            key = (view, str(r["id"]))
            methods = [(cfg, r["completion"])]
            if cfg == "trained" and key in logit:
                methods += [(f"probe>{t:g}", REF if logit[key] > t else r["completion"]) for t in a.thresholds]
            e = ex[key]
            t = tasks.setdefault(e["dataset"], get_task(e["dataset"]))
            for m, comp in methods:
                c = rows[(view, r["group"], m)]
                c[0] += 1
                c[1] += bool(t.score(e, comp))
                c[2] += any(k in extract_answer(comp).lower() for k in ABSTAIN)
    out = {f"{v}|{g}|{m}": {"n": n, "acc": 100 * c / n, "refuse": 100 * r / n}
           for (v, g, m), (n, c, r) in sorted(rows.items())}
    Path(a.feats, "report.json").write_text(json.dumps(out, indent=1))
    for k, d in out.items():
        print(f"{a.tag:6s} {k:55s} n={d['n']:6d} acc {d['acc']:5.1f} refuse {d['refuse']:5.1f}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract")
    e.add_argument("--lora", required=True)
    e.add_argument("--view", required=True, choices=["image", "clean", "noimg"])
    e.add_argument("--out", required=True)
    e.add_argument("--layers", type=int, nargs="+", default=[18, 24])
    e.add_argument("--prompt-format", default="think_answer")
    e.add_argument("--shard", type=int, default=0)
    e.add_argument("--num-shards", type=int, default=1)
    e.add_argument("--batch", type=int, default=8)
    r = sub.add_parser("report")
    r.add_argument("--tag", required=True)
    r.add_argument("--probe-dir", required=True)
    r.add_argument("--feats", required=True)
    r.add_argument("--steer", required=True)
    r.add_argument("--layer", type=int, default=24)
    r.add_argument("--l2", type=float, default=0.1)
    r.add_argument("--thresholds", type=float, nargs="+", default=[0, 1, 2, 3])
    r.add_argument("--prompt-format", default="think_answer")
    a = ap.parse_args()
    {"extract": cmd_extract, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    main()
