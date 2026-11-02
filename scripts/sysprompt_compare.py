#!/usr/bin/env python3
"""
8B-teacher distillation under the system prompt (prompt_format think_answer,
"option 2"), 1000 steps, severity-4 GQA, vs the same runs under the original
inline prompt. Each model is tested against the base 4B under ITS OWN prompt.

    python scripts/sysprompt_compare.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from grid_compare import CELLS, CLEAN, DEGRADED, fmt, load, paired  # noqa: E402

RUNS = [("4B base, inline", ("Qwen3-VL-4B-Instruct", "base-extra")),
        ("8B base, inline", ("Qwen3-VL-8B-Instruct", "base8b-extra")),
        ("8B RKL, inline", "t8b_s4_rkl"), ("8B RKL+FKL H<0.2, inline", "t8b_s4_rkl_fkl_lowH0.2"),
        ("4B self RKL, inline", "t4e_s4_rkl"),
        ("4B base, system", ("base4b-sys", "base4b-sys-extra")),
        ("8B base, system", ("base8b-sys", "base8b-sys-extra")),
        ("8B RKL, system", "t8b_s4_rkl_sys"), ("8B RKL+FKL H<0.2, system", "t8b_s4_rkl_fkl_lowH0.2_sys")]
PAIRS = [("4B base, system", "8B RKL, system"), ("4B base, system", "8B RKL+FKL H<0.2, system"),
         ("8B RKL, system", "8B RKL+FKL H<0.2, system"),
         ("8B RKL, inline", "8B RKL, system"), ("8B RKL+FKL H<0.2, inline", "8B RKL+FKL H<0.2, system"),
         ("4B self RKL, inline", "8B RKL, system"), ("4B self RKL, inline", "8B RKL+FKL H<0.2, system"),
         ("4B base, inline", "4B base, system"), ("4B base, system", "8B base, system")]


def main():
    D = {n: load(None, dirs=r) if isinstance(r, tuple) else load(r) for n, r in RUNS}
    print(f"{'run':28s} " + " ".join(f"{ds[:6] + '/' + v[:4]:>11s}" for ds, v in CELLS) + "   tag(main/extra)")
    for n, _ in RUNS:
        acc = D[n][0]
        print(f"{n:28s} " + " ".join(f"{acc[c]:11.2f}" if c in acc else f"{'—':>11s}" for c in CELLS)
              + "   " + "/".join(f"{x:.1f}" for x in acc.get("tag", [])))
    print("\n== paired tests: B vs A (positive = B better)")
    for a, b in PAIRS:
        ra, rb = D[a][1], D[b][1]
        print(f"   {b:26s} vs {a:24s} degraded {fmt(paired(ra, rb, DEGRADED))} | clean {fmt(paired(ra, rb, CLEAN))} | "
              f"GQA-C {fmt(paired(ra, rb, [('gqa', 'corrupted')]))} | TextVQA-C {fmt(paired(ra, rb, [('textvqa', 'corrupted')]))} | "
              f"VQAv2-C {fmt(paired(ra, rb, [('vqav2', 'corrupted')]))} | VizWiz {fmt(paired(ra, rb, [('vizwiz', 'real')]))}")


if __name__ == "__main__":
    main()
