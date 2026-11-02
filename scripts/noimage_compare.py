#!/usr/bin/env python3
"""
How much of each system-prompt model's accuracy depends on seeing the image:
accuracy with the image (clean / corrupted; VizWiz real) vs the same prompt
with the image withheld (view "noimage"), per dataset. Per question it also
splits the clean-image accuracy into answers that need the image (right only
with it) and answers the text alone gets (right without it too).

    python scripts/noimage_compare.py
"""
import json
from pathlib import Path

E = Path("outputs/eval")
MODELS = [("4B base", ["base4b-sys", "base4b-sys-extra"], "noimg-base4b-sys"),
          ("8B base", ["base8b-sys", "base8b-sys-extra"], "noimg-base8b-sys"),
          ("SFT cold start only", ["sft_coldstart_8b_sys-student", "sft_coldstart_8b_sys-extra"], "noimg-sft_coldstart_8b_sys"),
          ("8B RKL", ["t8b_s4_rkl_sys-student", "t8b_s4_rkl_sys-extra"], "noimg-t8b_s4_rkl_sys"),
          ("8B RKL+FKL H<0.2", ["t8b_s4_rkl_fkl_lowH0.2_sys-student", "t8b_s4_rkl_fkl_lowH0.2_sys-extra"],
           "noimg-t8b_s4_rkl_fkl_lowH0.2_sys"),
          ("8B RKL + cold start", ["t8b_s4_rkl_sys_coldstart-student", "t8b_s4_rkl_sys_coldstart-extra"],
           "noimg-t8b_s4_rkl_sys_coldstart"),
          ("8B RKL+FKL H<0.2 + cold start", ["t8b_s4_rkl_fkl_lowH0.2_sys_coldstart-student",
                                             "t8b_s4_rkl_fkl_lowH0.2_sys_coldstart-extra"],
           "noimg-t8b_s4_rkl_fkl_lowH0.2_sys_coldstart"),
          ("8B RKL+FKL H>0.8", ["t8b_s4_rkl_fkl_high0.8_sys-student", "t8b_s4_rkl_fkl_high0.8_sys-extra"],
           "noimg-t8b_s4_rkl_fkl_high0.8_sys"),
          ("8B RKL+FKL H>0.8 + cold start", ["t8b_s4_rkl_fkl_high0.8_sys_coldstart-student",
                                            "t8b_s4_rkl_fkl_high0.8_sys_coldstart-extra"],
           "noimg-t8b_s4_rkl_fkl_high0.8_sys_coldstart"),
          ("8B RKL+FKL H<0.2|>0.8", ["t8b_s4_rkl_fkl_low0.2_high0.8_sys-student", "t8b_s4_rkl_fkl_low0.2_high0.8_sys-extra"],
           "noimg-t8b_s4_rkl_fkl_low0.2_high0.8_sys"),
          ("8B RKL+FKL H<0.2|>0.8 + cold start", ["t8b_s4_rkl_fkl_low0.2_high0.8_sys_coldstart-student",
                                                  "t8b_s4_rkl_fkl_low0.2_high0.8_sys_coldstart-extra"],
           "noimg-t8b_s4_rkl_fkl_low0.2_high0.8_sys_coldstart"),
          ("GRPO", ["grpo_s4_sys-student", "grpo_s4_sys-extra"], "noimg-grpo_s4_sys")]
DS = ["textvqa", "gqa", "vqav2", "vizwiz"]


def rows(dirs):
    out = {}
    for d in dirs:
        f = E / d / "results.jsonl"
        if f.exists():
            for r in map(json.loads, open(f)):
                out[(r["dataset"], r["question_id"], r["view"])] = r["correct"]
    return out


def acc(R, ds, view):
    v = [c for (d, _, w), c in R.items() if d == ds and w == view]
    return 100 * sum(v) / len(v) if v else None


def main():
    f = lambda x: f"{x:6.2f}" if x is not None else "    — "
    print("Accuracy with the image vs with the image withheld (same system prompt)")
    print(f"{'model':30s} | " + " | ".join(f"{d:^29s}" for d in DS))
    print(f"{'':30s} | " + " | ".join(("  real  no-img  drop          " if d == "vizwiz" else " clean  corr  no-img  drop    ") for d in DS))
    for name, with_dirs, no_dir in MODELS:
        W, N = rows(with_dirs), rows([no_dir])
        cells = []
        for d in DS:
            n = acc(N, d, "noimage")
            if d == "vizwiz":
                w = acc(W, d, "real")
                cells.append(f"{f(w)} {f(n)} {f(w - n) if None not in (w, n) else '    — '}         ")
            else:
                c, k = acc(W, d, "clean"), acc(W, d, "corrupted")
                cells.append(f"{f(c)} {f(k)} {f(n)} {f(c - n) if None not in (c, n) else '    — '}")
        print(f"{name:30s} | " + " | ".join(cells))
    print("\nPer question, clean image (VizWiz: real photo): % right only WITH the image | right also without it "
          "| right only WITHOUT it")
    for name, with_dirs, no_dir in MODELS:
        W, N = rows(with_dirs), rows([no_dir])
        if not N:
            print(f"{name:30s} (no-image eval not finished)")
            continue
        parts = []
        for d in DS:
            v = "real" if d == "vizwiz" else "clean"
            ks = [(d, q) for (dd, q, w) in W if dd == d and w == v and (d, q, "noimage") in N]
            if not ks:
                parts.append(f"{d}: —")
                continue
            img = sum(W[(d, q, v)] and not N[(d, q, "noimage")] for d, q in ks)
            both = sum(W[(d, q, v)] and N[(d, q, "noimage")] for d, q in ks)
            txt = sum(N[(d, q, "noimage")] and not W[(d, q, v)] for d, q in ks)
            parts.append(f"{d}: {100 * img / len(ks):5.1f} | {100 * both / len(ks):5.1f} | {100 * txt / len(ks):4.1f}")
        print(f"{name:30s} " + "   ".join(parts))


if __name__ == "__main__":
    main()
