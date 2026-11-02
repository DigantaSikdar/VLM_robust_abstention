"""
caad/eval/answer_entropy_probe.py
=================================
Is a model uncertain when it answers a question it should have refused?

Takes a model's own VizWiz evaluation outputs (results.jsonl, greedy decoding,
real photo) and re-scores each completion with the same model, prompt and image
(teacher forcing). Per completion it records the mean next-token entropy over

  * all completion tokens (reasoning + answer),
  * the answer span inside <answer>...</answer>,
  * the first answer token (where the model commits to an answer),

plus the probability of that first answer token. Completions are grouped by
gold label x behaviour:

  unans_answered   gold "unanswerable", model gave an answer   (hallucinated answer)
  unans_abstained  gold "unanswerable", model said unanswerable
  ans_correct      answerable question, correct answer
  ans_wrong        answerable question, wrong answer (not "unanswerable")

    python -m caad.eval.answer_entropy_probe --results outputs/eval/t8b_s4_rkl_sys-extra/results.jsonl \
        --lora .../checkpoints/step_1000/student --out outputs/answer_entropy/rkl_sys
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from ..eval.tasks.vqa_c import VizWiz
from ..trainers.caad import build_inputs

GROUPS = ["unans_answered", "unans_abstained", "ans_correct", "ans_wrong"]


def group_of(r):
    said_un = "unanswerable" in (r.get("normalized") or "")
    if r["category"] == "unanswerable":
        return "unans_abstained" if said_un else "unans_answered"
    if said_un:
        return None                                   # answerable but abstained: not part of the question
    return "ans_correct" if r["correct"] else "ans_wrong"


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, help="the model's VizWiz results.jsonl")
    ap.add_argument("--model", default="Qwen/Qwen3-VL-4B-Instruct")
    ap.add_argument("--lora", default=None)
    ap.add_argument("--root", default="/gscratch/dsikdar/caad_data/vqa_c")
    ap.add_argument("--prompt-format", default="think_answer")
    ap.add_argument("--per-group", type=int, default=250)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    from transformers import AutoModelForImageTextToText, AutoProcessor
    model = AutoModelForImageTextToText.from_pretrained(
        a.model, dtype=torch.bfloat16, attn_implementation="sdpa", device_map="cuda").eval()
    if a.lora:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, a.lora).eval()
    proc = AutoProcessor.from_pretrained(a.model)
    tok = proc.tokenizer
    end = "<|im_end|>"

    req = {e["id"]: e for e in VizWiz.load({"root": a.root, "prompt_format": a.prompt_format})}
    rows = [r for r in map(json.loads, open(a.results)) if r["dataset"] == "vizwiz" and r["completion"]]
    by = defaultdict(list)
    for r in rows:
        g = group_of(r)
        if g and re.search(r"<answer>.*?</answer>", r["completion"], re.S):
            by[g].append(r)
    rng = random.Random(a.seed)
    picks = [(g, r) for g in GROUPS for r in rng.sample(by[g], min(a.per_group, len(by[g])))]
    print({g: f"{min(a.per_group, len(by[g]))}/{len(by[g])}" for g in GROUPS}, flush=True)

    out_rows = []
    for i, (g, r) in enumerate(picks):
        e = req[r["question_id"]]
        comp = r["completion"]
        batch, cmask, plen = build_inputs(proc, e["system"], e["prompt"], e["image"], comp, "cuda")
        logits = model(**batch).logits[0, plen - 1:-1].float()          # predictions for completion tokens
        lp = logits.log_softmax(-1)
        H = -(lp.exp() * lp).sum(-1).cpu().numpy()
        ids = batch["input_ids"][0, plen:]
        p_tok = lp.gather(-1, ids[:, None]).squeeze(-1).exp().cpu().numpy()
        offs = tok(comp + end, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]
        assert len(offs) == len(H), (len(offs), len(H))
        m = list(re.finditer(r"<answer>(.*?)</answer>", comp, re.S))[-1]
        s, t = m.start(1), m.end(1)
        span = [j for j, (o0, o1) in enumerate(offs) if o0 >= s and o1 <= t and o1 > o0]
        span = [j for j in span if comp[offs[j][0]:offs[j][1]].strip()] or span
        if not span:
            continue
        out_rows.append({"group": g, "question_id": r["question_id"], "answer": m.group(1).strip(),
                         "H_all": float(H.mean()), "H_answer": float(H[span].mean()),
                         "H_answer_max": float(H[span].max()), "H_first": float(H[span[0]]),
                         "p_first": float(p_tok[span[0]]), "n_tokens": int(len(H))})
        if (i + 1) % 100 == 0:
            print(f"[probe] {i + 1}/{len(picks)}", flush=True)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "rows.jsonl").open("w") as fh:
        for x in out_rows:
            fh.write(json.dumps(x) + "\n")
    S = summarize(out_rows)
    S["config"] = vars(a)
    (out / "summary.json").write_text(json.dumps(S, indent=2))
    print(json.dumps(S, indent=2))


def auroc(pos, neg):
    """P(score of a random positive > random negative); ties count half."""
    pos, neg = np.asarray(pos), np.asarray(neg)
    if not len(pos) or not len(neg):
        return None
    gt = (pos[:, None] > neg[None, :]).mean()
    eq = (pos[:, None] == neg[None, :]).mean()
    return float(gt + 0.5 * eq)


def summarize(rows):
    S = {"groups": {}}
    for g in GROUPS:
        x = [r for r in rows if r["group"] == g]
        if x:
            S["groups"][g] = {"n": len(x), **{f"{k}_{f}": float(getattr(np, f)([r[k] for r in x]))
                                              for k in ("H_all", "H_answer", "H_first", "p_first")
                                              for f in ("mean", "median")}}
    # can the model's own uncertainty tell a hallucinated answer from a correct one?
    hall = [r for r in rows if r["group"] == "unans_answered"]
    corr = [r for r in rows if r["group"] == "ans_correct"]
    S["auroc_hallucinated_vs_correct"] = {k: auroc([r[k] for r in hall], [r[k] for r in corr])
                                         for k in ("H_all", "H_answer", "H_first")}
    S["auroc_hallucinated_vs_correct"]["1-p_first"] = auroc([1 - r["p_first"] for r in hall],
                                                           [1 - r["p_first"] for r in corr])
    return S


if __name__ == "__main__":
    main()
