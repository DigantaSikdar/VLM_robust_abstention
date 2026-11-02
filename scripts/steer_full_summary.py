#!/usr/bin/env python3
"""
Full steering evaluation summary (caad.eval.steer_full output): per model and view,
accuracy and abstention per group, and paired McNemar tests of each steered config
against the unsteered trained model on the same questions.

    python scripts/steer_full_summary.py outputs/steer_full/rkl/eval outputs/steer_full/grpo/eval
"""
import glob
import json
import math
import sys
from collections import defaultdict


def mc(a, b):
    g = sum(y and not x for x, y in zip(a, b)); l = sum(x and not y for x, y in zip(a, b)); n = g + l
    if not a:
        return 0.0, 1.0
    return 100 * (g - l) / len(a), (math.erfc(((abs(g - l) - 1) / math.sqrt(n)) / math.sqrt(2)) if n else 1.0)


def main():
    for d in sys.argv[1:]:
        R = defaultdict(dict)                       # (view, config) -> id -> row
        for f in glob.glob(f"{d}/*.jsonl"):
            for r in map(json.loads, open(f)):
                R[(r["view"], r["config"])][r["id"]] = r
        print(f"\n######## {d}")
        for view in ("image", "clean", "noimg"):
            cfgs = sorted({c for v, c in R if v == view}, key=lambda c: (c != "trained", c))
            if not cfgs:
                continue
            groups = sorted({r["group"] for c in cfgs for r in R[(view, c)].values()})
            ref = R.get((view, "trained"), {})
            metric = "abstain" if view == "noimg" else "correct"
            print(f"\n== view={view}  ({'abstention %, higher = better' if view == 'noimg' else 'accuracy %'})")
            print(f"{'config':26s} {'n':>6s} " + " ".join(f"{g[:16]:>16s}" for g in groups) + "   pooled   vs trained (p)")
            for c in cfgs:
                X = R[(view, c)]
                cells = []
                for g in groups:
                    v = [r[metric] for r in X.values() if r["group"] == g]
                    cells.append(f"{100 * sum(v) / len(v):16.2f}" if v else f"{'—':>16s}")
                pooled = 100 * sum(r[metric] for r in X.values()) / max(len(X), 1)
                line = f"{c:26s} {len(X):6d} " + " ".join(cells) + f"   {pooled:6.2f}"
                if c != "trained" and ref:
                    ks = [k for k in X if k in ref]
                    dd, p = mc([ref[k][metric] for k in ks], [X[k][metric] for k in ks])
                    line += f"   {dd:+6.2f} (p={p:.2g})"
                print(line)
            if view == "image":                     # abstention where it matters / where it hurts
                print("   abstention % with the image:")
                for c in cfgs:
                    X = R[(view, c)]
                    ab = {g: [r["abstain"] for r in X.values() if r["group"] == g] for g in groups}
                    print(f"   {c:26s} " + " ".join(f"{g[:16]}={100 * sum(v) / len(v):5.1f}" for g, v in ab.items() if v))


if __name__ == "__main__":
    main()
