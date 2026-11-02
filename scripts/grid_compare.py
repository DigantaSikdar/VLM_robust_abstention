#!/usr/bin/env python3
"""
Teacher x loss grid at 1000 steps (severity-4 GQA training):
  teachers: t4e (4B self-EMA, clean image), t8b (Qwen3-VL-8B frozen, clean image)
  losses:   jsd, rkl, rkl+FKL(H<0.2), rkl+FKL(H<0.2|>0.8), rkl+FKL(|dH|>0.2)
Prints accuracy on all benchmarks, paired tests of each loss vs RKL within a
teacher, and of 8B vs 4B for each loss, plus pooled degraded/clean summaries.

    python scripts/grid_compare.py
"""
import json
import math
import os

E = "outputs/eval"
LOSSES = [("JSD", "jsd"), ("RKL", "rkl"), ("RKL+FKL H<0.2", "rkl_fkl_lowH0.2"),
          ("RKL+FKL H<0.2|>0.8", "rkl_fkl_low0.2_high0.8"), ("RKL+FKL |dH|>0.2", "rkl_fkl_gap0.2")]
TEACHERS = [("4B self-EMA", "t4e"), ("8B frozen", "t8b")]
CELLS = [("textvqa", "clean"), ("textvqa", "corrupted"), ("gqa", "clean"), ("gqa", "corrupted"),
         ("vqav2", "clean"), ("vqav2", "corrupted"), ("vizwiz", "real")]
DEGRADED = [("gqa", "corrupted"), ("textvqa", "corrupted"), ("vqav2", "corrupted"), ("vizwiz", "real")]
CLEAN = [("gqa", "clean"), ("textvqa", "clean"), ("vqav2", "clean")]


def load(name, dirs=None):
    acc, res = {}, {}
    for d in dirs or (f"{name}-student", f"{name}-extra"):
        p = f"{E}/{d}"
        if os.path.exists(f"{p}/summary.json"):
            s = json.load(open(f"{p}/summary.json"))
            for ds, v in CELLS:
                if ds in s and v in s[ds]:
                    acc[(ds, v)] = 100 * s[ds][v]["accuracy"]
            acc["tag"] = acc.get("tag", []) + [100 * s["overall"]["extraction_rate"]]
        if os.path.exists(f"{p}/results.jsonl"):
            res.update({(r["dataset"], r["question_id"], r["view"]): r["correct"]
                        for r in map(json.loads, open(f"{p}/results.jsonl"))})
    return acc, res


def paired(ra, rb, cells):
    ks = [k for k in ra if (k[0], k[2]) in cells and k in rb]
    if not ks:
        return None
    g = sum(rb[k] and not ra[k] for k in ks); l = sum(ra[k] and not rb[k] for k in ks); n = g + l
    p = math.erfc(((abs(g - l) - 1) / math.sqrt(n)) / math.sqrt(2)) if n else 1.0
    return 100 * (g - l) / len(ks), p


def fmt(t):
    return "   —   " if t is None else f"{t[0]:+.2f}{'*' if t[1] < 0.05 else ' '}(p={t[1]:.2g})"


def main():
    base, _ = load(None, dirs=("Qwen3-VL-4B-Instruct", "base-extra"))
    D = {(t, l): load(f"{t}_s4_{l}") for _, t in TEACHERS for _, l in LOSSES}
    print(f"{'teacher':12s} {'loss':20s} " + " ".join(f"{ds[:6] + '/' + v[:4]:>11s}" for ds, v in CELLS))
    print(f"{'base model':12s} {'(no training)':20s} " + " ".join(f"{base.get(c, float('nan')):11.2f}" for c in CELLS))
    for tn, t in TEACHERS:
        for ln, l in LOSSES:
            acc = D[(t, l)][0]
            print(f"{tn:12s} {ln:20s} " + " ".join(f"{acc[c]:11.2f}" if c in acc else f"{'—':>11s}" for c in CELLS))
    print("\n== each loss vs RKL, same teacher (paired; pooled over independent question sets)")
    for tn, t in TEACHERS:
        rr = D[(t, "rkl")][1]
        for ln, l in LOSSES:
            if l == "rkl":
                continue
            rb = D[(t, l)][1]
            print(f"   {tn:12s} {ln:20s} degraded {fmt(paired(rr, rb, DEGRADED))} | clean {fmt(paired(rr, rb, CLEAN))} | "
                  f"GQA-C {fmt(paired(rr, rb, [('gqa', 'corrupted')]))} | VizWiz {fmt(paired(rr, rb, [('vizwiz', 'real')]))}")
    print("\n== 8B teacher vs 4B self-EMA teacher, same loss")
    for ln, l in LOSSES:
        ra, rb = D[("t4e", l)][1], D[("t8b", l)][1]
        print(f"   {ln:20s} degraded {fmt(paired(ra, rb, DEGRADED))} | clean {fmt(paired(ra, rb, CLEAN))} | "
              f"TextVQA-C {fmt(paired(ra, rb, [('textvqa', 'corrupted')]))} | VizWiz {fmt(paired(ra, rb, [('vizwiz', 'real')]))}")
    print("\n== <answer>-tag rate (main eval / VQAv2+VizWiz eval)")
    for tn, t in TEACHERS:
        print("   " + tn + ": " + " | ".join(f"{ln} " + "/".join(f"{x:.1f}" for x in D[(t, l)][0].get("tag", [])) for ln, l in LOSSES))


if __name__ == "__main__":
    main()
