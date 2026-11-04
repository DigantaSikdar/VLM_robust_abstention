"""Per-corruption accuracy gain over the base model (Qwen3-VL-4B, full fine-tuning).

    python make_per_corruption.py <repo outputs/eval dir> <out_dir>   -> per_corruption_gain.{pdf,png,csv}

Gain = accuracy(fine-tuned) - accuracy(base) on the same corrupted questions, pooled over
TextVQA, GQA and VQAv2 (all five severities, ~1,190 questions per corruption type).
Error bars: 95% CI of the paired difference (normal approximation).
"""
import csv
import json
import math
import os
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

EVAL, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)
MODELS = [("RKL distillation", "fullft_t8b_rkl_sys_1ep_r2", "#eb6834"),
          ("GRPO", "fullft_grpo_sys_1ep_v2", "#2a78d6")]
BASE = "base4b-sys"
FAMILIES = ["blur", "noise", "digital", "weather"]
INK, MUTED, GRID = "#1f2328", "#5a6470", "#e3e6ea"


def load(name):
    out = {}
    for s in ("", "-extra", "-student"):
        f = f"{EVAL}/{name}{s}/results.jsonl"
        if os.path.exists(f):
            for r in map(json.loads, open(f)):
                if r["view"] == "corrupted" and r["dataset"] != "vizwiz":
                    out[(r["dataset"], str(r["question_id"]))] = r
    return out


base = load(BASE)
meta = {}                                     # corruption name -> (family, seen in training)
for r in base.values():
    c = r["corruption"]
    meta[c["name"]] = (c["category"], bool(c.get("seen_in_training")))
gain = defaultdict(dict)                      # name -> model -> (mean, half-width of 95% CI, n)
for label, run, _ in MODELS:
    rows = load(run)
    diffs = defaultdict(list)
    for k, b in base.items():
        if k in rows:
            diffs[b["corruption"]["name"]].append(float(rows[k]["correct"]) - float(b["correct"]))
    for name, d in diffs.items():
        n = len(d)
        m = sum(d) / n
        sd = math.sqrt(sum((x - m) ** 2 for x in d) / (n - 1))
        gain[name][label] = (100 * m, 100 * 1.96 * sd / math.sqrt(n), n)

# order: by family, then by GRPO gain (largest first)
order = []
for fam in FAMILIES:
    names = [n for n in meta if meta[n][0] == fam]
    order += sorted(names, key=lambda n: -gain[n]["GRPO"][0])

with open(f"{OUT}/per_corruption_gain.csv", "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["corruption", "family", "seen_in_training", "model", "gain_points", "ci95_half_width", "n"])
    for n in order:
        for label, _, _ in MODELS:
            g, h, k = gain[n][label]
            w.writerow([n, meta[n][0], meta[n][1], label, f"{g:.2f}", f"{h:.2f}", k])

# ---------------------------------------------------------------- plot
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Liberation Sans", "Arial", "DejaVu Sans"],
                     "font.size": 7, "axes.linewidth": 0.6, "pdf.fonttype": 42, "ps.fonttype": 42})
fig, ax = plt.subplots(figsize=(3.3, 4.1))
bar_h, gap, fam_gap = 0.36, 0.06, 0.7
y, ys, fam_spans = 0.0, {}, []
for fam in FAMILIES:
    start = y
    for n in [n for n in order if meta[n][0] == fam]:
        ys[n] = y
        y += 2 * bar_h + gap + 0.28
    fam_spans.append((fam, start, y - 0.28 - gap))
    y += fam_gap - 0.28

for j, (label, _, col) in enumerate(MODELS):
    yy = [ys[n] + j * (bar_h + gap) for n in order]
    g = [gain[n][label][0] for n in order]
    h = [gain[n][label][1] for n in order]
    ax.barh(yy, g, height=bar_h, color=col, label=label, edgecolor="white", linewidth=0.6, zorder=2)
    ax.errorbar(g, yy, xerr=h, fmt="none", ecolor=INK, elinewidth=0.5, capsize=1.2, capthick=0.5, zorder=3)

pretty = lambda n: n.replace("_", " ").replace("jpeg compression", "JPEG").replace("elastic transform", "elastic")
ax.set_yticks([ys[n] + (bar_h + gap) / 2 for n in order])
ax.set_yticklabels([pretty(n) + (" †" if meta[n][1] else "") for n in order], color=INK)
ax.tick_params(axis="y", length=0, pad=2)
ax.tick_params(axis="x", length=2, width=0.6, colors=MUTED)
ax.invert_yaxis()
ax.axvline(0, color=INK, linewidth=0.7, zorder=1)
ax.grid(axis="x", color=GRID, linewidth=0.5, zorder=0)
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
ax.spines["bottom"].set_color(MUTED)
ax.set_xlabel("Accuracy gain over base (points)", color=INK)
xmax = max(gain[n][l][0] + gain[n][l][1] for n in order for l, _, _ in MODELS)
xmin = min(0, min(gain[n][l][0] - gain[n][l][1] for n in order for l, _, _ in MODELS))
ax.set_xlim(xmin - 0.5, xmax + 3.2)
for fam, a, b in fam_spans:                    # family labels at the right edge
    ax.text(xmax + 3.0, (a + b) / 2, fam, ha="right", va="center", color=MUTED, fontsize=7, style="italic")
for _, a, b in fam_spans[1:]:
    ax.axhline(a - fam_gap / 2 - 0.05, color=GRID, linewidth=0.6, zorder=0)
from matplotlib.lines import Line2D  # noqa: E402
hs, ls = ax.get_legend_handles_labels()
hs.append(Line2D([], [], linestyle="none"))
ls.append("† seen in training")
leg = ax.legend(hs, ls, loc="lower center", bbox_to_anchor=(0.4, 1.0), ncol=3, frameon=False, fontsize=7,
                handlelength=1.0, handletextpad=0.4, columnspacing=1.0, borderaxespad=0.2)
leg.get_texts()[-1].set_color(MUTED)
fig.tight_layout(pad=0.3)
fig.savefig(f"{OUT}/per_corruption_gain.pdf")
fig.savefig(f"{OUT}/per_corruption_gain.png", dpi=300)
print("wrote", OUT)
for n in order:
    print(f"{n:18s} {meta[n][0]:8s} seen={meta[n][1]!s:5s} " +
          "  ".join(f"{l}: {gain[n][l][0]:+5.1f} ±{gain[n][l][1]:.1f}" for l, _, _ in MODELS))
