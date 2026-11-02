#!/usr/bin/env python3
"""
Cold start (SFT on 8B answers, clean images) vs no cold start, 8B teacher,
1000 steps on severity-4 GQA. Accuracy on every benchmark and paired McNemar
tests (pooled over independent question sets), as in grid_compare.py.

    python scripts/coldstart_compare.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from grid_compare import CELLS, CLEAN, DEGRADED, fmt, load, paired  # noqa: E402

RUNS = [("base model", None), ("SFT cold start only", "sft_coldstart_8b"),
        ("4B self RKL", "t4e_s4_rkl"),
        ("8B RKL", "t8b_s4_rkl"), ("8B RKL + cold start", "t8b_s4_rkl_coldstart"),
        ("8B RKL+FKL H<0.2", "t8b_s4_rkl_fkl_lowH0.2"),
        ("8B RKL+FKL H<0.2 + cold start", "t8b_s4_rkl_fkl_lowH0.2_coldstart")]
PAIRS = [("8B RKL", "8B RKL + cold start"), ("8B RKL+FKL H<0.2", "8B RKL+FKL H<0.2 + cold start"),
         ("8B RKL + cold start", "8B RKL+FKL H<0.2 + cold start"),
         ("SFT cold start only", "8B RKL + cold start"),
         ("4B self RKL", "8B RKL + cold start"), ("4B self RKL", "8B RKL+FKL H<0.2 + cold start"),
         ("base model", "SFT cold start only"), ("base model", "8B RKL + cold start"),
         ("base model", "8B RKL+FKL H<0.2 + cold start")]


def main():
    D = {n: load(None, dirs=("Qwen3-VL-4B-Instruct", "base-extra")) if r is None else load(r) for n, r in RUNS}
    print(f"{'run':32s} " + " ".join(f"{ds[:6] + '/' + v[:4]:>11s}" for ds, v in CELLS) + "   tag(main/extra)")
    for n, _ in RUNS:
        acc = D[n][0]
        print(f"{n:32s} " + " ".join(f"{acc[c]:11.2f}" if c in acc else f"{'—':>11s}" for c in CELLS)
              + "   " + "/".join(f"{x:.1f}" for x in acc.get("tag", [])))
    print("\n== paired tests: B vs A (positive = B better)")
    for a, b in PAIRS:
        ra, rb = D[a][1], D[b][1]
        print(f"   {b:32s} vs {a:22s} degraded {fmt(paired(ra, rb, DEGRADED))} | clean {fmt(paired(ra, rb, CLEAN))} | "
              f"GQA-C {fmt(paired(ra, rb, [('gqa', 'corrupted')]))} | TextVQA-C {fmt(paired(ra, rb, [('textvqa', 'corrupted')]))} | "
              f"VQAv2-C {fmt(paired(ra, rb, [('vqav2', 'corrupted')]))} | VizWiz {fmt(paired(ra, rb, [('vizwiz', 'real')]))}")


if __name__ == "__main__":
    main()
