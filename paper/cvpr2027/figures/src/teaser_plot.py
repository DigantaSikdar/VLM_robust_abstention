"""Right panel of Fig. 1 (teaser): corrupted accuracy vs. correct refusal, for RKL and GRPO
on three backbones.

Style follows the plots of Palafox & Fridovich-Keil (arXiv:2609.12278): Computer
Modern serif, full black frame, light grid, no tick marks, lowercase axis labels,
frameless legend under the axes. Semantic palette: blue = ours, orange = RKL, red = GRPO,
gray = base.

Numbers: RKL and GRPO (full fine-tuning). Corrupted accuracy from Tab. accuracy;
correct refusal (CR, VizWiz unanswerable) from Tab. refusal; LATCH at
tau_p = 2 (p_c = 0.88).

Run:  python3 teaser_plot.py   ->  teaser_plot.pdf (embedded by fig1.tex)
"""
import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerTuple

HERE = os.path.dirname(os.path.abspath(__file__))

# Computer Modern (matplotlib's bundled cmr10, TrueType -> embeds as Type 42); tick
# labels and minus signs go through mathtext in the same face.
FONT = ["cmr10"]

PALETTE = {
    "blue_main": "#3775BA",
    "red_strong": "#DC5B52",
    "red_2": "#E9A6A1",
    "orange": "#EE9B4A",
    "orange_2": "#F3BE8B",
    "gray": "#767676",
    "ink": "#1F2430",
}

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": FONT,
    "mathtext.fontset": "cm",
    "axes.formatter.use_mathtext": True,
    "axes.unicode_minus": False,
    "font.size": 8,
    "axes.linewidth": 0.8,
    "axes.edgecolor": "#000000",
    "axes.grid": True,
    "grid.color": "#E3E3E3",
    "grid.linewidth": 0.6,
    "axes.axisbelow": True,
    "xtick.major.size": 0,
    "ytick.major.size": 0,
    "xtick.major.pad": 3,
    "ytick.major.pad": 3,
    "axes.labelpad": 2,
    "legend.frameon": False,
    "pdf.fonttype": 42,
})

# backbone: base (acc, CR), then for each recipe (acc, CR, +LATCH acc, +LATCH CR)
DATA = {
    "Qwen3-VL": ((64.26, 71.6), (66.38, 49.6, 66.42, 65.4), (71.84, 44.9, 71.43, 61.2)),
    "InternVL3.5": ((61.35, 58.6), (66.64, 34.0, 66.45, 60.0), (68.00, 25.9, 67.95, 56.6)),
    "Gemma-3": ((55.44, 28.2), (58.94, 11.8, 58.58, 50.2), (60.65, 10.9, 59.38, 43.2)),
}
# Each channel has one job: line + label color = backbone, fill = recipe, star with a blue
# outline = + LATCH. Dashed = base -> fine-tuned, solid = fine-tuned -> + LATCH.
GLOW_MAX, GLOW_MIN, GLOW_STEPS, GLOW_ALPHA = 15.0, 8.0, 12, 0.05
FAMILY = {"Qwen3-VL": "#16808A", "InternVL3.5": "#9C4F96", "Gemma-3": "#5E7F2B"}
LABEL_OFF = {"Qwen3-VL": (0, 3.5), "InternVL3.5": (-3, 3.5), "Gemma-3": (0, 3.5)}

fig, ax = plt.subplots(figsize=(3.17, 1.86))

# reversed: Qwen last, so its shorter line stays on top where it meets InternVL at x = 66.4
for name, ((ab, cb), rkl, grpo) in reversed(list(DATA.items())):
    fam = FAMILY[name]
    for (af, cf, as_, cs), col in ((grpo, PALETTE["red_strong"]), (rkl, PALETTE["orange"])):
        ax.plot([ab, af], [cb, cf], color=fam, lw=0.85, ls=(0, (3, 1.8)), zorder=2)
        ax.plot([af, as_], [cf, cs], color=fam, lw=1.0, zorder=2)
        ax.plot(af, cf, "o", ms=4.8, mfc=col, mec="white", mew=0.5, zorder=3)
        # glow in the backbone's color: stacked translucent stars add up to a star-shaped gradient
        for ms in np.linspace(GLOW_MAX, GLOW_MIN, GLOW_STEPS):
            ax.plot(as_, cs, "*", ms=ms, mfc=fam, mec="none", alpha=GLOW_ALPHA, zorder=2.5)
        # + LATCH: blue outline = ours, fill = the recipe it was applied to
        ax.plot(as_, cs, "*", ms=7.6, mfc=col, mec=PALETTE["blue_main"], mew=0.75, zorder=4)
    ax.plot(ab, cb, "o", ms=4.8, mfc="white", mec=PALETTE["gray"], mew=1.2, zorder=3)
    ax.annotate(name, (ab, cb), xytext=LABEL_OFF[name], textcoords="offset points",
                ha="center", va="bottom", fontsize=7.5, color=fam)

ax.set_xlim(53, 74)
ax.set_ylim(5, 82)
ax.set_xticks([55, 60, 65, 70])
ax.set_yticks([20, 40, 60, 80])
ax.set_xlabel("corrupted accuracy (%)")
ax.set_ylabel("correct refusal (%)")

handles = [
    plt.Line2D([], [], ls="", marker="o", ms=5, mfc="white", mec=PALETTE["gray"], mew=1.2),
    plt.Line2D([], [], ls="", marker="o", ms=5, mfc=PALETTE["orange"], mec="white"),
    plt.Line2D([], [], ls="", marker="o", ms=5, mfc=PALETTE["red_strong"], mec="white"),
    tuple(plt.Line2D([], [], ls="", marker="*", ms=7, mfc=c, mec=PALETTE["blue_main"], mew=0.7)
          for c in (PALETTE["orange"], PALETTE["red_strong"])),
]
fig.subplots_adjust(left=0.115, right=0.992, top=0.985, bottom=0.285)
fig.legend(handles, ["Base", "Distillation (RKL)", "GRPO", "+ LATCH (ours)"],
           loc="lower center", bbox_to_anchor=(0.5, -0.025), ncol=4, fontsize=7.4,
           handletextpad=0.05, columnspacing=0.7, borderaxespad=0.0,
           handler_map={tuple: HandlerTuple(ndivide=None, pad=0.0)})

out = os.path.join(HERE, "teaser_plot.pdf")
fig.savefig(out)
fig.savefig(os.path.join(HERE, "teaser_plot.png"), dpi=300)
print(out)
