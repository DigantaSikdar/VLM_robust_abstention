#!/usr/bin/env python3
"""
System-prompt (think_answer) cold start: SFT of the 4B on the 8B's system-prompt
answers (clean images), then 8B-teacher distillation (RKL, RKL + FKL H<0.2),
vs the same distillation without the cold start. Everything trained and
evaluated under the system prompt.

    python scripts/sysprompt_coldstart_compare.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from grid_compare import CELLS, CLEAN, DEGRADED, fmt, load, paired  # noqa: E402

RUNS = [("4B base", ("base4b-sys", "base4b-sys-extra")), ("8B base", ("base8b-sys", "base8b-sys-extra")),
        ("SFT cold start only", "sft_coldstart_8b_sys"),
        ("RKL", "t8b_s4_rkl_sys"), ("RKL + cold start", "t8b_s4_rkl_sys_coldstart"),
        ("RKL+FKL H<0.2", "t8b_s4_rkl_fkl_lowH0.2_sys"),
        ("RKL+FKL H<0.2 + cold start", "t8b_s4_rkl_fkl_lowH0.2_sys_coldstart"),
        ("RKL+FKL H>0.8", "t8b_s4_rkl_fkl_high0.8_sys"),
        ("RKL+FKL H>0.8 + cold start", "t8b_s4_rkl_fkl_high0.8_sys_coldstart"),
        ("RKL+FKL H<0.2|>0.8", "t8b_s4_rkl_fkl_low0.2_high0.8_sys"),
        ("RKL+FKL H<0.2|>0.8 + cold start", "t8b_s4_rkl_fkl_low0.2_high0.8_sys_coldstart"),
        ("GRPO (8 rollouts, labels)", "grpo_s4_sys")]
PAIRS = [("RKL", "RKL + cold start"), ("RKL+FKL H<0.2", "RKL+FKL H<0.2 + cold start"),
         ("RKL + cold start", "RKL+FKL H<0.2 + cold start"), ("4B base", "SFT cold start only"),
         ("SFT cold start only", "RKL + cold start"),
         ("4B base", "RKL + cold start"), ("4B base", "RKL+FKL H<0.2 + cold start"),
         ("4B base", "RKL+FKL H<0.2|>0.8 + cold start"), ("RKL + cold start", "RKL+FKL H<0.2|>0.8 + cold start"),
         ("RKL+FKL H<0.2 + cold start", "RKL+FKL H<0.2|>0.8 + cold start"),
         ("RKL+FKL H<0.2|>0.8", "RKL+FKL H<0.2|>0.8 + cold start"), ("RKL", "RKL+FKL H<0.2|>0.8"),
         ("RKL+FKL H<0.2", "RKL+FKL H<0.2|>0.8"), ("4B base", "RKL+FKL H<0.2|>0.8"),
         ("RKL+FKL H>0.8", "RKL+FKL H>0.8 + cold start"), ("RKL", "RKL+FKL H>0.8"),
         ("RKL + cold start", "RKL+FKL H>0.8 + cold start"), ("4B base", "RKL+FKL H>0.8"),
         ("4B base", "GRPO (8 rollouts, labels)"), ("RKL+FKL H<0.2", "GRPO (8 rollouts, labels)"),
         ("RKL", "GRPO (8 rollouts, labels)")]


def main():
    D = {n: load(None, dirs=r) if isinstance(r, tuple) else load(r) for n, r in RUNS}
    print(f"{'model':32s} " + " ".join(f"{ds[:6] + '/' + v[:4]:>11s}" for ds, v in CELLS) + "   tag(main/extra)")
    for n, _ in RUNS:
        acc = D[n][0]
        print(f"{n:32s} " + " ".join(f"{acc[c]:11.2f}" if c in acc else f"{'—':>11s}" for c in CELLS)
              + "   " + "/".join(f"{x:.1f}" for x in acc.get("tag", [])))
    print("\n== paired tests: B vs A (positive = B better)")
    for a, b in PAIRS:
        ra, rb = D[a][1], D[b][1]
        print(f"   {b:32s} vs {a:28s} degraded {fmt(paired(ra, rb, DEGRADED))} | clean {fmt(paired(ra, rb, CLEAN))} | "
              f"GQA-C {fmt(paired(ra, rb, [('gqa', 'corrupted')]))} | TextVQA-C {fmt(paired(ra, rb, [('textvqa', 'corrupted')]))} | "
              f"VQAv2-C {fmt(paired(ra, rb, [('vqav2', 'corrupted')]))} | VizWiz {fmt(paired(ra, rb, [('vizwiz', 'real')]))}")
    print("\n== overlap ratio, 4B student vs 8B teacher, system prompt (step 0)")
    for n, d in (("base 4B", "t8b_sys_s0"), ("after SFT", "t8b_sft_sys_s0")):
        f = Path(f"outputs/overlap/{d}/summary.json")
        if f.exists():
            s = json.loads(f.read_text())
            print(f"   {n:10s} overlap {s['overlap_ratio']:.4f} | abs entropy gap {s['abs_entropy_gap']:.4f} | "
                  f"top1 agree {s['top1_agree']:.4f} | overlap adv {s['overlap_adv_mean_over_v']:.4f}")
    f = Path("/gscratch/dsikdar/caad_data/coldstart_gqa30k/filter_stats_think_answer.json")
    if f.exists():
        print("\n== cold-start data filter:", f.read_text())


if __name__ == "__main__":
    main()
