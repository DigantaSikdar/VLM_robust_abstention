#!/usr/bin/env python3
"""Pick real examples for the qualitative teaser (fig1_qual.tex). Run on the cluster, in the
repo root, with caad importable. Reproduces the LATCH decision of caad.eval.probe_switch.

robustness  corrupted, answerable: base wrong (no refusal), fine-tuned right, LATCH silent.
abstention  VizWiz unanswerable: base refuses, fine-tuned answers, LATCH fires.

    python paper/cvpr2027/figures/src/select_teaser_examples.py \
        --tag fgrpo --probe-dir outputs/probe_fullft/fgrpo --feats outputs/probe_switch/fgrpo \
        --steer outputs/steer_full/fgrpo/eval --base outputs/eval/<base-run>/results.jsonl \
        [outputs/eval/<base-extra-run>/results.jsonl ...] --out teaser_candidates

Writes <out>/candidates.json and copies each candidate's image into <out>/.
"""
import argparse
import glob
import json
import shutil
from pathlib import Path

import numpy as np

from caad.eval.probe_answerability import fit_lr, score
from caad.eval.steer_full import examples
from caad.eval.steer_probe import ABSTAIN
from caad.eval.tasks.registry import get_task
from caad.eval.tasks.vqa_c import extract_answer


def refused(comp):
    return any(k in extract_answer(comp).lower() for k in ABSTAIN)


def probe_logits(a):
    z = np.load(Path(a.probe_dir) / "vizwiz_train.npz", mmap_mode="r")
    Xtr = np.asarray(z["X"][:, a.layer], np.float32)
    ytr = np.array([m["unanswerable"] for m in json.loads(str(z["meta"]))])
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    w, b = fit_lr((Xtr - mu) / sd, ytr, a.l2)
    out = {}
    for f in sorted(glob.glob(f"{a.feats}/image__shard*.npz")):
        z = np.load(f)
        col = list(z["layers"]).index(a.layer)
        s = np.asarray(score((z["X"][:, col] - mu) / sd, w, b))
        for m, v in zip(json.loads(str(z["meta"])), s):
            out[str(m["id"])] = float(v)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--probe-dir", required=True)
    ap.add_argument("--feats", required=True)
    ap.add_argument("--steer", required=True)
    ap.add_argument("--base", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--layer", type=int, default=24)
    ap.add_argument("--l2", type=float, default=0.1)
    ap.add_argument("--tau", type=float, default=2.0)
    ap.add_argument("--prompt-format", default="think_answer")
    ap.add_argument("--max-gold-words", type=int, default=2)
    ap.add_argument("--max-question-words", type=int, default=9)
    ap.add_argument("--n", type=int, default=25, help="candidates kept per side")
    a = ap.parse_args()

    logit = probe_logits(a)
    ex = {str(e["id"]): (g, e) for g, e in examples("image", a.prompt_format)}
    base = {}
    for f in a.base:
        for r in map(json.loads, open(f)):
            if r["view"] == "corrupted" or r["dataset"] == "vizwiz":
                base[str(r["question_id"])] = r

    tasks, rob, abst, seen = {}, [], [], 0
    for f in sorted(glob.glob(f"{a.steer}/image__trained__shard*.jsonl")):
        for r in map(json.loads, open(f)):
            i = str(r["id"])
            if i not in ex or i not in logit or i not in base:
                continue
            seen += 1
            g, e = ex[i]
            t = tasks.setdefault(e["dataset"], get_task(e["dataset"]))
            b = base[i]["completion"]
            ft = r["completion"]
            if len(str(e.get("question", "")).split()) > a.max_question_words:
                continue
            row = {"id": i, "group": g, "dataset": e["dataset"], "image": e["image"],
                   "question": e.get("question"), "gold": e.get("gold"),
                   "corruption": e.get("corruption"), "probe_logit": round(logit[i], 3),
                   "base": extract_answer(b), "fine_tuned": extract_answer(ft),
                   "base_completion": b, "fine_tuned_completion": ft}
            if g.endswith("_corrupted"):
                if len(str(e.get("gold", "")).split()) > a.max_gold_words:
                    continue
                if not t.score(e, b) and not refused(b) and t.score(e, ft) and logit[i] <= a.tau:
                    row["latch"] = row["fine_tuned"]
                    rob.append(row)
            elif g == "vizwiz_unans":
                if refused(b) and not refused(ft) and logit[i] > a.tau:
                    row["latch"] = "unanswerable"
                    abst.append(row)

    # hardest corruptions first for robustness; most confident probe first for abstention
    rob.sort(key=lambda r: (-(r["corruption"] or {}).get("severity", 0), r["probe_logit"]))
    abst.sort(key=lambda r: -r["probe_logit"])
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    res = {"robustness": rob[:a.n], "abstention": abst[:a.n]}
    (out / "candidates.json").write_text(json.dumps(res, indent=1))
    for side, rows in res.items():
        print(f"\n=== {side}: {len(rob if side == 'robustness' else abst)} match, showing {len(rows)}")
        for k, r in enumerate(rows):
            dst = out / f"{side}_{k:02d}{Path(r['image']).suffix}"
            shutil.copy(r["image"], dst)
            print(f"[{k:02d}] {r['dataset']} {r['id']} logit={r['probe_logit']} "
                  f"corr={(r['corruption'] or {}).get('name')} | Q: {r['question']} | gold: {r['gold']}"
                  f" | base: {r['base']} | ft: {r['fine_tuned']} | latch: {r['latch']} | {dst.name}")
    print(f"\nchecked {seen} items; wrote {out}/candidates.json")


if __name__ == "__main__":
    main()
