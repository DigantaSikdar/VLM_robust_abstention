#!/usr/bin/env python3
"""
System-prompt (prompt_format think_answer) grid only: base 4B / 8B and
distilled students (4B self-EMA teacher vs frozen 8B teacher; RKL vs
RKL + FKL H<0.2), 1000 steps on severity-4 GQA, all trained and evaluated
under the system prompt.

    python scripts/sysprompt_grid.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from grid_compare import CELLS, CLEAN, DEGRADED, fmt, load, paired  # noqa: E402

RUNS = [("4B base", ("base4b-sys", "base4b-sys-extra")), ("8B base", ("base8b-sys", "base8b-sys-extra")),
        ("4B self, RKL", "t4e_s4_rkl_sys"), ("4B self, RKL+FKL H<0.2", "t4e_s4_rkl_fkl_lowH0.2_sys"),
        ("8B teacher, RKL", "t8b_s4_rkl_sys"), ("8B teacher, RKL+FKL H<0.2", "t8b_s4_rkl_fkl_lowH0.2_sys")]
PAIRS = [("4B base", "4B self, RKL"), ("4B base", "4B self, RKL+FKL H<0.2"),
         ("4B base", "8B teacher, RKL"), ("4B base", "8B teacher, RKL+FKL H<0.2"),
         ("4B self, RKL", "4B self, RKL+FKL H<0.2"), ("8B teacher, RKL", "8B teacher, RKL+FKL H<0.2"),
         ("4B self, RKL", "8B teacher, RKL"), ("4B self, RKL+FKL H<0.2", "8B teacher, RKL+FKL H<0.2")]


def main():
    D = {n: load(None, dirs=r) if isinstance(r, tuple) else load(r) for n, r in RUNS}
    print(f"{'model':26s} " + " ".join(f"{ds[:6] + '/' + v[:4]:>11s}" for ds, v in CELLS) + "   tag(main/extra)")
    for n, _ in RUNS:
        acc = D[n][0]
        print(f"{n:26s} " + " ".join(f"{acc[c]:11.2f}" if c in acc else f"{'—':>11s}" for c in CELLS)
              + "   " + "/".join(f"{x:.1f}" for x in acc.get("tag", [])))
    print("\n== paired tests: B vs A (positive = B better)")
    for a, b in PAIRS:
        ra, rb = D[a][1], D[b][1]
        print(f"   {b:26s} vs {a:24s} degraded {fmt(paired(ra, rb, DEGRADED))} | clean {fmt(paired(ra, rb, CLEAN))} | "
              f"GQA-C {fmt(paired(ra, rb, [('gqa', 'corrupted')]))} | TextVQA-C {fmt(paired(ra, rb, [('textvqa', 'corrupted')]))} | "
              f"VQAv2-C {fmt(paired(ra, rb, [('vqav2', 'corrupted')]))} | VizWiz {fmt(paired(ra, rb, [('vizwiz', 'real')]))}")


if __name__ == "__main__":
    main()
