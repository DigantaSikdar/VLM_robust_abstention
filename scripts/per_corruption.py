#!/usr/bin/env python3
"""
Per-corruption comparison of evaluation runs on the corrupted view, with paired
McNemar tests against a reference run and Benjamini-Hochberg FDR across all
(run, corruption, dataset) tests. A "noise" run (same recipe as the reference,
different seed/hardware) calibrates how many differences appear by chance.

    python scripts/per_corruption.py --ref a6k_rkl=outputs/eval/a6k_s4_rkl-student \
        --noise h100_rkl=outputs/eval/s500_s4_rkl-student \
        gap=outputs/eval/a6k_s4_rkl_fkl_gap0.2-student ...
"""
import argparse
import json
import math
from collections import defaultdict

CATS = {"noise": ["gaussian_noise", "shot_noise", "impulse_noise", "speckle_noise"],
        "blur": ["defocus_blur", "glass_blur", "motion_blur", "zoom_blur", "gaussian_blur"],
        "weather": ["snow", "frost", "fog", "brightness", "spatter"],
        "digital": ["contrast", "elastic_transform", "pixelate", "jpeg_compression", "saturate"]}
ORDER = [c for cs in CATS.values() for c in cs]


def load(path):
    out = {}
    for l in open(f"{path}/results.jsonl"):
        r = json.loads(l)
        if r["view"] == "corrupted":
            out[(r["dataset"], r["question_id"])] = (r["correct"], r["corruption"]["name"])
    return out


def mcnemar(g, l):
    n = g + l
    if n == 0:
        return 1.0
    k = min(g, l)                                   # exact two-sided binomial
    p = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n * 2
    return min(p, 1.0)


def bh(pvals):
    m = len(pvals); order = sorted(range(m), key=lambda i: pvals[i]); q = [0.0] * m; prev = 1.0
    for rank, i in reversed(list(enumerate(order, 1))):
        prev = min(prev, pvals[i] * m / rank); q[i] = prev
    return q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--noise", required=True)
    ap.add_argument("--base", default="base=outputs/eval/Qwen3-VL-4B-Instruct")
    ap.add_argument("runs", nargs="+")
    a = ap.parse_args()
    rname, rpath = a.ref.split("=", 1)
    ref = load(rpath)
    base_name, base_path = a.base.split("=", 1)
    base = load(base_path)
    runs = dict(x.split("=", 1) for x in [a.noise] + a.runs)
    data = {k: load(v) for k, v in runs.items()}
    noise_name = a.noise.split("=", 1)[0]

    tests = []                                       # (run, ds, corr, delta, g, l, p)
    acc = defaultdict(dict)
    for ds in ("gqa", "textvqa"):
        for c in ORDER:
            keys = [k for k, v in ref.items() if k[0] == ds and v[1] == c]
            n = len(keys)
            acc[(ds, c)]["n"] = n
            acc[(ds, c)][base_name] = 100 * sum(base[k][0] for k in keys) / n
            acc[(ds, c)][rname] = 100 * sum(ref[k][0] for k in keys) / n
            for name, d in data.items():
                g = sum(d[k][0] and not ref[k][0] for k in keys)
                l = sum(ref[k][0] and not d[k][0] for k in keys)
                acc[(ds, c)][name] = 100 * sum(d[k][0] for k in keys) / n
                tests.append([name, ds, c, 100 * (g - l) / n, g, l, mcnemar(g, l)])
    qs = bh([t[6] for t in tests])
    for t, q in zip(tests, qs):
        t.append(q)
    T = {(t[0], t[1], t[2]): t for t in tests}

    names = list(data)
    for ds in ("gqa", "textvqa"):
        print(f"\n== {ds.upper()} corrupted: accuracy, and change vs {rname} (paired; * p<0.05, ** FDR q<0.10)")
        hdr = f"{'corruption':18s} {'n':>4} {base_name:>6} {rname:>8} | " + " ".join(f"{n_[:16]:>16}" for n_ in names)
        print(hdr)
        for cat, cs in CATS.items():
            for c in cs:
                row = acc[(ds, c)]
                cells = []
                for n_ in names:
                    _, _, _, dlt, g, l, p, q = T[(n_, ds, c)]
                    mark = "**" if q < 0.10 else ("*" if p < 0.05 else "")
                    cells.append(f"{dlt:+6.1f}{mark:<2} ({g:>3}/{l:<3})")
                print(f"{c:18s} {row['n']:4d} {row[base_name]:6.1f} {row[rname]:8.1f} | " + " ".join(f"{x:>16}" for x in cells))
    print("\n== nominal p<0.05 and FDR q<0.10 counts per run (out of 38 corruption x dataset cells)")
    for n_ in names:
        ts = [t for t in tests if t[0] == n_]
        up = sum(t[6] < 0.05 and t[3] > 0 for t in ts); dn = sum(t[6] < 0.05 and t[3] < 0 for t in ts)
        fdr = sum(t[7] < 0.10 for t in ts)
        pos = sum(t[3] > 0 for t in ts); neg = sum(t[3] < 0 for t in ts)
        tag = "  <- same recipe as reference: this is the chance level" if n_ == noise_name else ""
        print(f"   {n_:24s} better in {pos:2d} / worse in {neg:2d} cells | p<0.05: {up} up, {dn} down | FDR q<0.10: {fdr}{tag}")
    print(f"\n== training gain over {base_name} per corruption ({rname}), GQA / TextVQA:")
    print("   " + ", ".join(f"{c} {acc[('gqa', c)][rname] - acc[('gqa', c)][base_name]:+.1f}/{acc[('textvqa', c)][rname] - acc[('textvqa', c)][base_name]:+.1f}" for c in ORDER))


if __name__ == "__main__":
    main()
