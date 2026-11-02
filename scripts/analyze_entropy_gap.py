#!/usr/bin/env python3
"""
Does the student-teacher entropy gap (dH = H_student - H_teacher) pick out the
tokens worth a forward-KL push? Uses entropy_probe outputs (tokens.npz,
rollouts.jsonl). Compares gates at equal token budgets on: share of content
words, forward-KL mass captured, and how many teacher/student top-1
disagreements (where the student picks a different word) each gate catches.

    python scripts/analyze_entropy_gap.py outputs/probe/s4_base_tokens
"""
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from analyze_probe_tokens import category  # noqa: E402


def rank(x):
    r = np.empty(len(x)); r[np.argsort(x)] = np.arange(len(x)); return r


def main(path):
    from transformers import AutoTokenizer
    root = Path(path)
    D = dict(np.load(root / "tokens.npz", allow_pickle=True))
    meta = [json.loads(l) for l in open(root / "rollouts.jsonl")]
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-VL-4B-Instruct")
    vocab = {i: tok.decode([i]) for i in set(D["tok"].tolist()) | set(D["t_arg"].tolist()) | set(D["s_arg"].tolist())}
    strs = np.array([vocab[i] for i in D["tok"]], dtype=object)
    cats = np.array([category(s, False) for s in strs], dtype=object)
    Ht, Hs, f, agree = D["H_t"], D["H_s"], D["fkl"], D["agree"]
    dH = Hs - Ht
    n = len(Ht)
    flip = agree == 0
    content = cats == "content"
    rng = np.random.default_rng(0)

    print(f"tokens {n}")
    print("dH = H_s - H_t percentiles:", {p: round(float(np.percentile(dH, p)), 4) for p in (1, 5, 10, 25, 50, 75, 90, 95, 99)})
    print(f"student MORE uncertain (dH>0.05): {100 * (dH > 0.05).mean():.1f}% | about equal (|dH|<=0.05): {100 * (abs(dH) <= 0.05).mean():.1f}% | student LESS uncertain (dH<-0.05): {100 * (dH < -0.05).mean():.1f}%")
    print(f"rank corr(dH, FKL) = {np.corrcoef(rank(dH), rank(f))[0, 1]:.3f} | rank corr(|dH|, FKL) = {np.corrcoef(rank(abs(dH)), rank(f))[0, 1]:.3f} | rank corr(H_t, FKL) = {np.corrcoef(rank(Ht), rank(f))[0, 1]:.3f}")

    print(f"\nteacher/student top-1 disagreements: {flip.sum()} tokens ({100 * flip.mean():.1f}%), {100 * f[flip].sum() / f.sum():.1f}% of all FKL")
    for name, m in [("all flips", flip), ("flips where teacher confident (H_t<0.2)", flip & (Ht < 0.2))]:
        print(f"   {name}: n={m.sum()}, student dH>0.05 in {100 * (dH[m] > 0.05).mean():.0f}%, |dH|<=0.05 in {100 * (abs(dH[m]) <= 0.05).mean():.0f}%, student MORE confident (dH<-0.05) in {100 * (dH[m] < -0.05).mean():.0f}%; median H_s {np.median(Hs[m]):.3f}")

    def topk(score, frac):
        k = int(frac * n); m = np.zeros(n, bool); m[np.argsort(-score)[:k]] = True; return m
    gates = {
        "top dH (student more uncertain)": lambda fr: topk(dH, fr),
        "top |dH|": lambda fr: topk(abs(dH), fr),
        "lowest H_t (confident teacher)": lambda fr: topk(-Ht + rng.random(n) * 1e-9, fr),
        "highest H_t (EOPD-like)": lambda fr: topk(Ht, fr),
        "random": lambda fr: topk(rng.random(n), fr),
        "ORACLE: highest FKL": lambda fr: topk(f, fr),
    }
    for fr in (0.10, 0.20):
        print(f"\n== budget {int(100 * fr)}% of tokens ({int(fr * n)} tokens)")
        print(f"   {'gate':34s} {'content%':>8} {'function%':>9} {'FKL captured':>12} {'flips caught':>12} {'confident-teacher flips caught':>30}")
        cf = flip & (Ht < 0.2)
        for name, g in gates.items():
            m = g(fr)
            print(f"   {name:34s} {100 * content[m].mean():8.1f} {100 * (cats[m] == 'function').mean():9.1f} {100 * f[m].sum() / f.sum():11.1f}% {100 * (m & flip).sum() / flip.sum():11.1f}% {100 * (m & cf).sum() / cf.sum():29.1f}%")

    # what the top-dH tokens are
    m = topk(dH, 0.10)
    print("\ntop-10% dH tokens, most frequent:", ", ".join(f"{k}({v})" for k, v in Counter(s.strip().lower() for s in strs[m]).most_common(20)))
    print("top-10% dH tokens by category:", {k: f"{100 * v / m.sum():.0f}%" for k, v in Counter(cats[m]).most_common()})
    by_style = {}
    for st in sorted(set(D["style"].tolist())):
        sm = D["style"] == st
        by_style[st] = (float(np.mean(dH[sm])), float(np.mean(dH[sm & content])))
    print("mean dH by corruption (all tokens / content tokens):", {k: f"{v[0]:+.3f}/{v[1]:+.3f}" for k, v in sorted(by_style.items(), key=lambda kv: -kv[1][1])})

    # examples: highest-dH tokens, and confident flips the dH gate would miss
    def ctx(j, w=10):
        r = D["rollout"][j]; idx = np.where(D["rollout"] == r)[0]; p = int(np.where(idx == j)[0][0])
        return meta[r]["style"], "".join(strs[idx[max(0, p - w):p]])[-70:]
    print("\nhighest-dH tokens (student confused, teacher sure):")
    for j in np.argsort(-dH)[:10]:
        st, c = ctx(j)
        print(f"   [{st}] …{c!r} → {strs[j]!r}  H_t {Ht[j]:.3f} H_s {Hs[j]:.3f} FKL {f[j]:.3f} teacher {vocab[D['t_arg'][j]]!r} student {vocab[D['s_arg'][j]]!r}")
    miss = np.where(flip & (Ht < 0.2) & (dH < 0.05))[0]
    print(f"\nconfident-teacher flips with dH < 0.05 (a dH gate misses these): {len(miss)} of {(flip & (Ht < 0.2)).sum()}")
    for j in miss[np.argsort(-f[miss])][:10]:
        st, c = ctx(j)
        print(f"   [{st}] …{c!r} → teacher {vocab[D['t_arg'][j]]!r} vs student {vocab[D['s_arg'][j]]!r}  H_t {Ht[j]:.3f} H_s {Hs[j]:.3f} FKL {f[j]:.2f}")


if __name__ == "__main__":
    main(sys.argv[1])
