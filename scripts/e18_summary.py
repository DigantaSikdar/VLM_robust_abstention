#!/usr/bin/env python3
"""E18 collateral curve: for each model and direction, abstention gained on VizWiz
unanswerable vs accuracy lost on answerable groups, across lambda (same subset).

    python scripts/e18_summary.py outputs/e18/rkl outputs/e18/grpo
"""
import glob
import json
import math
import sys
from collections import defaultdict

ANS = ["vizwiz_ans", "gqa_corrupted", "vqav2_corrupted", "textvqa_corrupted"]


def mc(a, b):
    g = sum(y and not x for x, y in zip(a, b)); l = sum(x and not y for x, y in zip(a, b)); n = g + l
    return math.erfc(((abs(g - l) - 1) / math.sqrt(n)) / math.sqrt(2)) if n else 1.0


def main():
    for d in sys.argv[1:]:
        R = defaultdict(dict)
        for f in glob.glob(f"{d}/*.jsonl"):
            for r in map(json.loads, open(f)):
                R[r["config"]][r["id"]] = r
        ref = R.get("trained", {})
        print(f"\n######## {d}   (n per config: {len(ref)})")
        print(f"{'config':24s} {'lam':>6s} {'unans acc':>9s} {'unans abst':>10s} " + " ".join(f"{g[:12]:>12s}" for g in ANS)
              + f" {'ans pooled':>10s} {'Δans':>6s} {'p':>7s} {'Δunans':>7s} {'ratio':>6s} {'no-tag%':>7s}")
        def key(c):
            if c == "trained":
                return ("", 0)
            p = c.split(":"); return (p[0], float(p[3]))
        for c in sorted(R, key=key):
            X = R[c]
            un = [r for r in X.values() if r["group"] == "vizwiz_unans"]
            accs = []
            for g in ANS:
                v = [r["correct"] for r in X.values() if r["group"] == g]
                accs.append(100 * sum(v) / len(v) if v else float("nan"))
            ks = [k for k in X if X[k]["group"] in ANS and k in ref]
            pooled = 100 * sum(X[k]["correct"] for k in ks) / max(len(ks), 1)
            base_pooled = 100 * sum(ref[k]["correct"] for k in ks) / max(len(ks), 1)
            uk = [r["id"] for r in un if r["id"] in ref]
            u_acc = 100 * sum(r["correct"] for r in un) / max(len(un), 1)
            u_ref = 100 * sum(ref[k]["correct"] for k in uk) / max(len(uk), 1)
            dans, dun = pooled - base_pooled, u_acc - u_ref
            p = mc([ref[k]["correct"] for k in ks], [X[k]["correct"] for k in ks]) if c != "trained" else 1.0
            notag = 100 * sum("<answer>" not in r["completion"] for r in X.values()) / len(X)
            lam = X[next(iter(X))]["lambda"]
            ratio = f"{dun / -dans:6.1f}" if dans < -0.05 else "   inf" if dun > 0 else "     —"
            print(f"{c:24s} {lam:6.2f} {u_acc:9.2f} {100*sum(r['abstain'] for r in un)/max(len(un),1):10.2f} "
                  + " ".join(f"{x:12.2f}" for x in accs) + f" {pooled:10.2f} {dans:+6.2f} {p:7.2g} {dun:+7.2f} {ratio} {notag:7.1f}")


if __name__ == "__main__":
    main()
