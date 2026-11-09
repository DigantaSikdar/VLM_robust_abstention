"""Fig. per-corruption: accuracy gain over the base model per corruption type.

Style follows the plots of arXiv:2609.12278, as in the teaser: Computer Modern serif,
full black frame, light grid, no tick marks, lowercase labels, frameless legend below. Both recipes are
robust fine-tuning, so they share the red family used for fine-tuning in Fig. 1
(GRPO = red, RKL = orange).

Data: figures/data/per_corruption_gain.csv, written by
docs/iclr2027/figures/per_corruption/make_per_corruption.py from outputs/eval
(Qwen3-VL-4B, full fine-tuning; gain = fine-tuned - base accuracy on the same corrupted
questions, pooled over TextVQA, GQA, VQAv2; 95% CI of the paired difference).

Run:  python3 per_corruption_plot.py   ->  ../per_corruption_gain.pdf
"""
import csv
import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, "..", "data", "per_corruption_gain.csv")
OUT = os.path.join(HERE, "..", "per_corruption_gain")

INK, MUTED, GRID = "#000000", "#666666", "#E3E3E3"
MODELS = [("RKL distillation", "#EE9B4A"), ("GRPO", "#DC5B52")]
FAMILIES = ["blur", "noise", "digital", "weather"]

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["cmr10"],
    "mathtext.fontset": "cm",
    "axes.formatter.use_mathtext": True,
    "axes.unicode_minus": False,
    "font.size": 8,
    "axes.linewidth": 0.8,
    "axes.edgecolor": "#000000",
    "xtick.major.size": 0,
    "xtick.major.pad": 3,
    "legend.frameon": False,
    "pdf.fonttype": 42,
})

gain, meta, order = defaultdict(dict), {}, []
for r in csv.DictReader(open(CSV)):
    n = r["corruption"]
    if n not in meta:
        meta[n] = (r["family"], r["seen_in_training"] == "True")
        order.append(n)                     # CSV is already ordered: family, then GRPO gain
    gain[n][r["model"]] = (float(r["gain_points"]), float(r["ci95_half_width"]))

fig, ax = plt.subplots(figsize=(3.3, 4.1))
bar_h, pair_gap, row_gap, fam_gap = 0.36, 0.04, 0.30, 0.55
y, ys, spans = 0.0, {}, []
for fam in FAMILIES:
    start = y
    for n in [n for n in order if meta[n][0] == fam]:
        ys[n] = y
        y += 2 * bar_h + pair_gap + row_gap
    spans.append((fam, start, y - row_gap))
    y += fam_gap

for j, (label, col) in enumerate(MODELS):
    yy = [ys[n] + j * (bar_h + pair_gap) for n in order]
    g = [gain[n][label][0] for n in order]
    h = [gain[n][label][1] for n in order]
    ax.barh(yy, g, height=bar_h, color=col, edgecolor="none", zorder=2)
    ax.errorbar(g, yy, xerr=h, fmt="none", ecolor="#333333", elinewidth=0.6, capsize=1.3,
                capthick=0.6, zorder=3)

pretty = {"jpeg_compression": "JPEG", "elastic_transform": "elastic"}
ax.set_yticks([ys[n] + (bar_h + pair_gap) / 2 for n in order])
ax.set_yticklabels([pretty.get(n, n.replace("_", " ")) + (r"$^\dagger$" if meta[n][1] else "")
                    for n in order], color=INK)
ax.tick_params(axis="y", length=0, pad=3)
ax.invert_yaxis()
ax.axvline(0, color="#000000", linewidth=0.8, zorder=4)
ax.grid(axis="x", color=GRID, linewidth=0.6, zorder=0)
ax.set_xlabel("accuracy gain over base (points)")

xmax = max(gain[n][l][0] + gain[n][l][1] for n in order for l, _ in MODELS)
xmin = min(0, min(gain[n][l][0] - gain[n][l][1] for n in order for l, _ in MODELS))
ax.set_xlim(xmin - 0.4, xmax + 3.4)
ax.set_xticks([0, 5, 10, 15])
ax.set_ylim(y - fam_gap + 0.1, -0.5)
for fam, a, b in spans:
    ax.text(xmax + 3.3, (a + b) / 2, fam, ha="right", va="center", color=MUTED,
            fontsize=8)
for _, a, _ in spans[1:]:
    ax.axhline(a - fam_gap / 2 - row_gap / 2, color="#BDBDBD", linewidth=0.6, zorder=0)

handles = [Patch(color=c) for _, c in MODELS] + [Line2D([], [], ls="none")]
labels = [l for l, _ in MODELS] + [r"$^\dagger$seen in training"]
leg = ax.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.45, -0.088), ncol=3,
                handlelength=1.0, handleheight=0.8, handletextpad=0.4, columnspacing=1.1,
                borderaxespad=0.0)
leg.get_texts()[-1].set_color(MUTED)

fig.tight_layout(pad=0.2)
fig.savefig(OUT + ".pdf", bbox_inches="tight", pad_inches=0.02)
fig.savefig(OUT + ".png", dpi=300, bbox_inches="tight", pad_inches=0.02)
print(OUT + ".pdf")
