"""
caad/eval/answer_commitment.py
==============================
E17: how committed is each model at the moment it writes its answer?

For VizWiz val questions (all labeled unanswerable + a sample of answerable ones),
teacher-force a reasoning prefix ending in "<answer>" and read the model's
distribution over the first answer token. Two prefixes per question:

  own    the model's own greedy response up to "<answer>"   (what it actually did)
  base   the BASE model's response up to "<answer>"           (same reasoning for all models;
                                                               the base mostly concluded "cannot determine")

Per item: entropy of the full next-token distribution, abstention mass (summed
probability of tokens that begin an abstention: unanswerable / cannot / unable /
insufficient / unclear ...), and the probability of the top token.

    python -m caad.eval.answer_commitment --tag rkl --lora <adapter> \
        --own "outputs/steer_full/rkl/eval/image__trained__shard*of8.jsonl" --out outputs/commitment
"""

from __future__ import annotations

import argparse
import glob
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from .tasks.registry import get_task

ROOT = "/gscratch/dsikdar/caad_data/vqa_c"
ABSTAIN_WORDS = ["unanswerable", "Unanswerable", "cannot", "Cannot", "unable", "Unable", "insufficient",
                 "Insufficient", "unclear", "Unclear", "unknown", "Unknown", "not determinable", "Not determinable",
                 "indeterminate", "Indeterminate", "impossible", "Impossible"]


KEEP = {"un", "cannot", "unknown", "unable", "unclear", "insufficient", "impossible"}   # "un" = start of "unanswerable"


def abstain_ids(tok):
    """First tokens of abstention answers, kept only when the token itself is clearly
    abstention-like (drops generic fragments such as 'im', 'ind', 'ins', 'not')."""
    ids = set()
    for w in ABSTAIN_WORDS:
        for pre in ("", " "):
            t = tok(pre + w, add_special_tokens=False).input_ids
            if t and tok.decode([t[0]]).strip().lower() in KEEP:
                ids.add(t[0])
    return sorted(ids)


def load_completions(pattern, view="real"):
    out = {}
    for f in glob.glob(pattern):
        for r in map(json.loads, open(f)):
            if r["dataset"] == "vizwiz" and r.get("view", view) in (view, "image"):
                out[r.get("question_id", r.get("id"))] = r["completion"]
    return out


def prefix(comp):
    i = comp.find("<answer>")
    return None if i < 0 else comp[:i + len("<answer>")]


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--lora", default=None)
    ap.add_argument("--own", required=True, help="glob of this model's VizWiz results (completions)")
    ap.add_argument("--base", default="outputs/eval/base4b-sys-extra/results.jsonl")
    ap.add_argument("--n-ans", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--out", default="outputs/commitment")
    a = ap.parse_args()

    from PIL import Image
    from transformers import AutoProcessor
    from .steer_probe import load_policy
    model = load_policy(a.lora)                    # None = base; LoRA adapter dir; or full checkpoint dir
    proc = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-4B-Instruct")
    proc.tokenizer.padding_side = "left"
    AB = torch.tensor(abstain_ids(proc.tokenizer), device="cuda")

    own, base = load_completions(a.own), load_completions(a.base)
    viz = get_task("vizwiz").load({"root": ROOT, "prompt_format": "think_answer"})
    un = [e for e in viz if e.get("category") == "unanswerable"]
    an = random.Random(0).sample([e for e in viz if e.get("category") != "unanswerable"], a.n_ans)
    jobs = []
    for e in un + an:
        for kind, src in (("own", own), ("base", base)):
            p = prefix(src.get(e["id"], ""))
            if p is not None:
                jobs.append((e, kind, p, src[e["id"]]))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    f = out / f"{a.tag}.jsonl"
    t0 = time.time()
    with f.open("w") as fh:
        for i in range(0, len(jobs), a.batch):
            chunk = jobs[i:i + a.batch]
            texts, imgs = [], []
            for e, kind, p, _ in chunk:
                msgs = [{"role": "system", "content": [{"type": "text", "text": e["system"]}]},
                        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": e["prompt"]}]}]
                texts.append(proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True) + p)
                with Image.open(e["image"]) as im:
                    imgs.append(im.convert("RGB"))
            inp = proc(text=texts, images=imgs, padding=True, return_tensors="pt").to("cuda")
            logits = model(**inp, logits_to_keep=1).logits[:, -1].float()
            lp = logits.log_softmax(-1)
            p = lp.exp()
            H = -(p * lp).sum(-1)
            mass = p[:, AB].sum(-1)
            top = p.max(-1)
            for j, (e, kind, pref, comp) in enumerate(chunk):
                ans = comp[len(pref):].split("</answer>")[0].strip().lower()
                fh.write(json.dumps({"id": e["id"], "unanswerable": e.get("category") == "unanswerable",
                                     "prefix": kind, "entropy": float(H[j]), "abstain_mass": float(mass[j]),
                                     "top_p": float(top.values[j]),
                                     "top_token": proc.tokenizer.decode([int(top.indices[j])]),
                                     "answered_abstain": "unanswerable" in ans or "cannot" in ans}) + "\n")
            if (i // a.batch) % 50 == 0:
                print(f"[{a.tag}] {i + len(chunk)}/{len(jobs)} {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
