#!/usr/bin/env python3
"""
Compare TextVQA-C / GQA-C evaluation runs (outputs of caad.eval.vllm_eval).

    python scripts/compare_evals.py --ref outputs/eval/s500_s4_jsd-student \
        base=outputs/eval/Qwen3-VL-4B-Instruct jsd=outputs/eval/s500_s4_jsd-student ...

Prints accuracy per dataset/view for every run, the <answer> extraction rate,
paired McNemar tests of each run against --ref (same questions, same images),
and corrupted-view breakdowns by severity / category / seen-vs-unseen.
"""
import argparse
import json
import math
from pathlib import Path

DS, VIEWS = ("textvqa", "gqa"), ("clean", "corrupted")


def load(path):
    p = Path(path)
    summ = json.loads((p / "summary.json").read_text())
    res = {(r["dataset"], r["question_id"], r["view"]): r["correct"]
           for r in map(json.loads, (p / "results.jsonl").open())}
    return summ, res


def mcnemar(ref, other, keys):
    g = sum(other[k] and not ref[k] for k in keys)
    l = sum(ref[k] and not other[k] for k in keys)
    n = g + l
    p = math.erfc(((abs(g - l) - 1) / math.sqrt(n)) / math.sqrt(2)) if n else 1.0
    return g, l, min(p, 1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True, help="eval dir every run is tested against")
    ap.add_argument("runs", nargs="+", help="name=eval_dir")
    a = ap.parse_args()
    runs = dict(r.split("=", 1) for r in a.runs)
    data = {k: load(v) for k, v in runs.items()}
    ref_s, ref_r = load(a.ref)
    w = max(len(k) for k in runs) + 2

    print("<answer> extraction:", " | ".join(f"{k} {100 * s['overall']['extraction_rate']:.2f}%"
                                              for k, (s, _) in data.items()))
    for ds in DS:
        for v in VIEWS:
            print(f"\n{ds} {v}")
            keys = [k for k in ref_r if k[0] == ds and k[2] == v]
            for name, (s, r) in data.items():
                acc = 100 * s[ds][v]["accuracy"]
                if Path(runs[name]).resolve() == Path(a.ref).resolve():
                    print(f"  {name:<{w}} {acc:6.2f}   (reference)")
                    continue
                g, l, p = mcnemar(ref_r, r, keys)
                print(f"  {name:<{w}} {acc:6.2f}   vs ref {acc - 100 * ref_s[ds][v]['accuracy']:+5.2f}"
                      f"  fixed {g:4d} broke {l:4d}  p={p:.2g}")
        for grp in ("by_severity", "by_category", "by_seen"):
            keys = list(ref_s[ds][grp])
            print(f"  {ds} corrupted {grp}: " + "   ".join(
                f"{k}: " + "/".join(f"{100 * s[ds][grp][k]['accuracy']:.1f}" for s, _ in data.values())
                for k in keys) + f"   [{'/'.join(data)}]")


if __name__ == "__main__":
    main()
