#!/usr/bin/env python3
"""E17 summary: first-answer-token entropy / abstention mass per model, prefix and subgroup.

    python scripts/commitment_summary.py outputs/commitment
"""
import json
import sys
from pathlib import Path

import numpy as np


def main():
    d = Path(sys.argv[1] if len(sys.argv) > 1 else "outputs/commitment")
    print(f"{'model':5s} {'prefix':5s} {'group':22s} {'n':>5s} {'H mean':>7s} {'H med':>6s} {'abst mass':>9s} {'P(mass>.5)':>10s} {'top_p':>6s}")
    for tag in ("base", "rkl", "grpo"):
        f = d / f"{tag}.jsonl"
        if not f.exists():
            continue
        R = [json.loads(l) for l in open(f)]
        own = {r["id"]: r for r in R if r["prefix"] == "own"}
        for pre in ("own", "base"):
            rows = [r for r in R if r["prefix"] == pre]
            groups = {
                "unans (all)": [r for r in rows if r["unanswerable"]],
                "unans, own=answered": [r for r in rows if r["unanswerable"] and r["id"] in own and not own[r["id"]]["answered_abstain"]],
                "unans, own=abstained": [r for r in rows if r["unanswerable"] and r["id"] in own and own[r["id"]]["answered_abstain"]],
                "answerable": [r for r in rows if not r["unanswerable"]],
            }
            for g, x in groups.items():
                if not x:
                    continue
                H = np.array([r["entropy"] for r in x]); M = np.array([r["abstain_mass"] for r in x]); T = np.array([r["top_p"] for r in x])
                print(f"{tag:5s} {pre:5s} {g:22s} {len(x):5d} {H.mean():7.3f} {np.median(H):6.3f} {M.mean():9.3f} {100*(M>.5).mean():9.1f}% {T.mean():6.3f}")
        print()


if __name__ == "__main__":
    main()
