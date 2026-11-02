#!/usr/bin/env python3
"""Summarize the steering sweep (caad.eval.steer_probe eval): per config, accuracy on
held-out VizWiz unanswerable (= abstaining correctly), VizWiz answerable, GQA-C and
VQAv2-C, how often steering was on, and paired McNemar tests against the unsteered
trained model.  python scripts/steer_summary.py [outputs/steer/rkl_sys]"""
import glob
import json
import math
import sys

D = sys.argv[1] if len(sys.argv) > 1 else "outputs/steer/rkl_sys"
G = ["viz_unans", "viz_ans", "gqa_c", "vqav2_c"]


def mc(a, b, ks):
    g = sum(b[k] and not a[k] for k in ks); l = sum(a[k] and not b[k] for k in ks); n = g + l
    if not ks:
        return 0.0, 1.0
    return 100 * (g - l) / len(ks), (math.erfc(((abs(g - l) - 1) / math.sqrt(n)) / math.sqrt(2)) if n else 1.0)


R = {}
for f in sorted(glob.glob(f"{D}/eval_*.jsonl")):
    rows = [json.loads(l) for l in open(f)]
    if rows:
        R[rows[0]["config"]] = {r["id"]: r for r in rows}
ref = R.get("trained")
_seen = {x["group"] for X in R.values() for x in X.values()}
G = [g for g in G if g in _seen] + sorted(g for g in _seen if g not in G)
order = ["base", "trained"] + sorted(k for k in R if k not in ("base", "trained"))
print(f"{'config':28s} {'n':>4s} " + " ".join(f"{g:>9s}" for g in G) + "   corr-ans  steer%  no-tag%  tokens"
      "   | vs trained: unans (p)        corr-ans (p)")
for c in order:
    if c not in R:
        continue
    X = R[c]
    acc = {g: [x["correct"] for x in X.values() if x["group"] == g] for g in G}
    cell = lambda g: f"{100 * sum(acc[g]) / len(acc[g]):9.1f}" if acc[g] else f"{'—':>9s}"
    ca = acc.get("gqa_c", []) + acc.get("vqav2_c", [])
    steer = 100 * sum(x["steered_frac"] for x in X.values()) / len(X)
    notag = 100 * sum("<answer>" not in x["completion"] for x in X.values()) / len(X)
    tok = sum(x["gen_tokens"] for x in X.values()) / len(X)
    line = f"{c:28s} {len(X):4d} " + " ".join(cell(g) for g in G) + \
        f"   {100 * sum(ca) / max(len(ca), 1):8.1f}  {steer:6.1f}  {notag:7.1f}  {tok:6.0f}"
    if ref and c != "trained":
        A = {k: v["correct"] for k, v in ref.items()}; B = {k: v["correct"] for k, v in X.items()}
        ku = [k for k in B if k in A and X[k]["group"] == "viz_unans"]
        kc = [k for k in B if k in A and X[k]["group"] in ("gqa_c", "vqav2_c")]
        (du, pu), (dc, pc) = mc(A, B, ku), mc(A, B, kc)
        line += f"   | {du:+6.1f} (p={pu:.2g})   {dc:+6.1f} (p={pc:.2g})"
    print(line)

# abstention ("unanswerable" / "cannot determine" ...) per group; without an image this is
# the correct behaviour for every question, with an image only for VizWiz-unanswerable
if any("abstain" in x for X in R.values() for x in X.values()):
    print(f"\n{'config':28s} " + " ".join(f"{'abst ' + g:>14s}" for g in G) + "   abst all   | vs trained: abst all (p)")
    for c in order:
        if c not in R or not all("abstain" in x for x in R[c].values()):
            continue
        X = R[c]
        cell = lambda g: (lambda v: f"{100 * sum(v) / len(v):14.1f}" if v else f"{'—':>14s}")(
            [x["abstain"] for x in X.values() if x["group"] == g])
        line = f"{c:28s} " + " ".join(cell(g) for g in G) + f"   {100 * sum(x['abstain'] for x in X.values()) / len(X):8.1f}"
        if ref and c != "trained" and all("abstain" in x for x in ref.values()):
            ks = [k for k in X if k in ref]
            d, p = mc({k: ref[k]["abstain"] for k in ks}, {k: X[k]["abstain"] for k in ks}, ks)
            line += f"   | {d:+6.1f} (p={p:.2g})"
        print(line)
