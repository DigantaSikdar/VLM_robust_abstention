"""Render the demo gallery. ALL DATA HERE IS SYNTHETIC; do not cite or reuse these numbers.

Run:  python3 make_demos.py   ->  demos/*.pdf, demos/*.png, and demos/lint_report.txt
"""
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker  # noqa: F401
import numpy as np

import paperplot as pp

OUT = Path(__file__).resolve().parent / "demos"
rng = np.random.default_rng(7)
report = {}


def finish(fig, name):
    report[name] = pp.save(fig, OUT / name)
    plt.close(fig)


def learning_curves():
    fig, axes = plt.subplots(1, 4, figsize=pp.size("double", h=1.62), sharex=True, sharey=True)
    x = np.linspace(0, 1, 40)
    params = {
        "ours": (0.96, 11, 0.20), "a": (0.78, 9, 0.30), "b": (0.66, 8, 0.36), "c": (0.54, 7, 0.45)}
    for i, (ax, name) in enumerate(zip(axes, ["Dataset A", "Dataset B", "Dataset C", "Dataset D"])):
        for role, (top, k, mid) in params.items():
            shift = 0.05 * i if role != "ours" else 0.015 * i
            runs = (top - shift) / (1 + np.exp(-k * (x - mid - 0.02 * i)))
            runs = runs + rng.normal(0, 0.025, (8, x.size)).cumsum(1) * 0.20
            m = runs.mean(0)
            s = 1.96 * runs.std(0, ddof=1) / np.sqrt(len(runs))
            pp.line(ax, x, m, role, lo=m - s, hi=m + s)
        pp.line(ax, x, np.full_like(x, 1.0), "oracle")
        ax.set_title(name)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.1)
        ax.set_xticks([0, 0.5, 1], ["0", "0.5", "1M"])
        ax.set_yticks([0, 0.5, 1])
    axes[0].set_ylabel("normalized return")
    pp.shared_xlabel(axes, "environment steps")
    pp.legend(axes, [pp.key("ours", "Ours"), pp.key("a", "Baseline A"), pp.key("b", "Baseline B"),
                     pp.key("c", "Baseline C"), pp.key("oracle", "Oracle")])
    finish(fig, "01_learning_curves")


def risk_coverage():
    fig, ax = plt.subplots(figsize=pp.size("single", h=2.3))
    c = np.linspace(0.2, 1, 60)
    curves = {"ours": (0.015, 0.145, 2.8), "a": (0.05, 0.14, 1.6), "b": (0.09, 0.135, 1.1)}
    names = {"ours": "Ours", "a": "Baseline A", "b": "Baseline B"}
    handles = []
    for role, (lo, top, p) in curves.items():
        y = lo + (top + (0.03 if role != "ours" else 0) - lo) * (c - 0.2) ** p / 0.8 ** p
        band = 0.006 + 0.012 * (1 - c)
        pp.line(ax, c, y * 100, role, lo=(y - band) * 100, hi=(y + band) * 100)
        aurc = np.trapezoid(y, c) / 0.8
        handles.append(pp.key(role, f"{names[role]} (AURC {aurc:.3f})"))
    pp.line(ax, c, np.maximum(0, 1 - (1 - 0.12) / c) * 100, "oracle")
    ax.vlines(0.8, 0, 17.5, color=pp.MUTED, lw=0.5, zorder=1)
    ax.set(xlim=(0.2, 1.0), ylim=(0, 25), xlabel="coverage (fraction of questions answered)",
           ylabel="risk on answered (%)", yticks=[0, 5, 10, 15, 20])
    target = mpl.lines.Line2D([], [], color=pp.MUTED, lw=0.5, label="target coverage")
    pp.legend(ax, handles + [pp.key("oracle", "Oracle"), target])
    finish(fig, "02_risk_coverage")


def grouped_bars():
    fig, ax = plt.subplots(figsize=pp.size("single", h=2.0))
    data = {
        "ours": ([84.1, 76.8, 69.4], [1.0, 1.3, 1.5]), "a": ([79.3, 71.2, 58.7], [1.2, 1.5, 1.8]),
        "b": ([76.0, 66.5, 52.9], [1.4, 1.2, 2.1]), "c": ([70.2, 60.1, 47.0], [1.1, 1.7, 1.9])}
    w = 0.2
    groups = np.arange(3)
    for i, (role, (v, e)) in enumerate(data.items()):
        pos = groups + (i - 1.5) * (w + 0.015)
        pp.bar(ax, pos, v, w, role, err=e)
        if role == "ours":
            for p, val, er in zip(pos, v, e):
                ax.text(p, val + er + 1.2, f"{val:.1f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(groups, ["Dataset A", "Dataset B", "Dataset C"])
    ax.set(ylim=(0, 118), ylabel="accuracy (%)", yticks=range(0, 101, 20))
    ax.grid(axis="x", visible=False)
    pp.legend(ax, [pp.key("ours", "Ours", "patch"), pp.key("a", "Baseline A", "patch"),
                   pp.key("b", "Baseline B", "patch"), pp.key("c", "Baseline C", "patch")])
    finish(fig, "03_grouped_bars")


def aggregate_intervals():
    names = ["Ours", "Baseline A", "Baseline B", "Baseline C", "Baseline D"]
    roles = ["ours", "a", "b", "c", "d"]
    centers = [0.93, 0.74, 0.68, 0.55, 0.47]
    runs = [np.clip(rng.normal(m, 0.15, 60), 0, 1.5) for m in centers]
    metrics = [("IQM", pp.iqm), ("mean", np.mean), ("optimality gap (lower is better)", lambda z: np.mean(np.maximum(1 - z, 0)))]
    fig, axes = plt.subplots(1, 3, figsize=pp.size("double", h=1.28), sharey=True)
    for ax, (title, f) in zip(axes, metrics):
        for k, (r, role) in enumerate(zip(runs, roles)):
            point, lo, hi = pp.boot_ci(r, f, seed=k)
            col = pp.ROLES[role].color
            ax.plot([lo, hi], [k, k], color=col, lw=3.4, alpha=0.35, solid_capstyle="butt", zorder=2)
            ax.plot([point], [k], ls="none", marker=pp.ROLES[role].marker, color=col, ms=4.2, zorder=3)
        ax.set_title(title)
        ax.grid(axis="y", visible=False)
        ax.margins(x=0.12)
        ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(0.2))
    axes[0].set_yticks(range(5), names)
    axes[0].invert_yaxis()
    axes[0].set_ylim(4.55, -0.55)
    pp.shared_xlabel(axes, "normalized score, 95% bootstrap CI")
    finish(fig, "04_aggregate_intervals")


def dumbbell():
    names = ["zoom blur", "snow", "defocus", "gaussian", "contrast", "JPEG", "motion blur", "fog", "frost",
             "brightness"]
    base = np.array([41.2, 46.8, 44.1, 39.0, 49.5, 58.3, 47.0, 55.1, 53.0, 69.4])
    gain = np.array([17.1, 14.8, 13.9, 13.0, 11.2, 9.4, 8.8, 7.1, 6.3, 3.1])
    fig, ax = plt.subplots(figsize=pp.size("single", h=2.45))
    y = np.arange(len(names))
    ax.hlines(y, base, base + gain, color=pp.ROLES["base"].color, lw=1.0, zorder=2)
    ax.plot(base, y, ls="none", marker="o", mfc="white", mec=pp.ROLES["base"].color, mew=0.9, ms=4.2, zorder=3)
    ax.plot(base + gain, y, ls="none", marker="o", color=pp.ROLES["ours"].color, ms=4.6, zorder=3)
    for yy, b, g in zip(y[[0]], base[[0]], gain[[0]]):
        pp.label_end(ax, b, yy, "Base", dx=-5, ha="right")
        pp.label_end(ax, b + g, yy, "Ours", dx=5)
    for yy, b, g in zip(y, base, gain):
        ax.text(83.5, yy, f"+{g:.1f}", ha="right", va="center", fontsize=7)
    ax.text(83.5, -1.05, "gain", ha="right", va="center", fontsize=7, color=pp.MUTED)
    ax.set_yticks(y, names)
    ax.set(xlim=(32, 84), ylim=(len(names) - 0.4, -0.65), xlabel="accuracy on corrupted images (%)",
           xticks=[40, 50, 60, 70])
    ax.grid(axis="y", visible=False)
    finish(fig, "05_dumbbell")


def reliability():
    fig, (top, bot) = plt.subplots(2, 1, figsize=pp.size("single", h=2.6), sharex=True,
                                   gridspec_kw={"height_ratios": [3, 1]})
    conf = np.linspace(0.05, 0.95, 10)
    acc_a = np.clip(conf - 0.16 * np.sin(np.pi * conf) ** 1.2, 0, 1)
    acc_o = np.clip(conf + 0.018 * np.sin(5 * conf), 0, 1)
    top.plot([0, 1], [0, 1], color=pp.ROLES["oracle"].color, ls=pp.ROLES["oracle"].ls, lw=0.8, zorder=1)
    pp.line(top, conf, acc_a, "a", marker=True, lo=acc_a - 0.03, hi=acc_a + 0.03)
    pp.line(top, conf, acc_o, "ours", marker=True, lo=acc_o - 0.018, hi=acc_o + 0.018)
    pp.label_end(top, conf[6], acc_o[6], "Ours (ECE 0.013)", dx=-5, dy=7, ha="right")
    pp.label_end(top, conf[4], acc_a[4], "Baseline A (ECE 0.112)", dx=8, dy=-10, ha="left", va="top")
    pp.label_end(top, 0.93, 0.975, "perfect", dx=-2, dy=0, ha="right", va="bottom", color=pp.MUTED)
    top.set(xlim=(0, 1), ylim=(0, 1.05), ylabel="empirical accuracy")
    h = np.array([2, 3, 4, 5, 7, 9, 13, 17, 19, 21], float)
    bot.bar(conf, h, width=0.085, color=pp.tint("ours", pp.OURS_BAR_SHADE), zorder=2)
    bot.set(ylim=(0, 25), ylabel="% samples", xlabel="predicted confidence")
    bot.set_xticks(np.arange(0, 1.01, 0.2))
    bot.grid(axis="x", visible=False)
    finish(fig, "06_reliability")


def heatmap():
    models = ["Ours", "Baseline A", "Baseline B", "Baseline C", "Baseline D"]
    corr = ["blur", "noise", "fog", "snow", "JPEG", "pixelate", "contrast", "frost"]
    m = np.array([
        [9.8, 12.4, 6.1, 8.8, 7.0, 6.2, 8.1, 7.5],
        [4.2, 6.9, 1.5, 3.0, 4.8, 3.3, 1.1, 2.0],
        [2.3, 4.0, -1.8, 0.7, 3.5, 1.8, -2.6, 0.4],
        [0.9, 2.6, -3.5, -1.0, 2.2, 0.5, -4.9, -2.2],
        [-1.2, 0.6, -6.1, -4.1, 0.3, -1.5, -7.2, -3.8]])
    fig, ax = plt.subplots(figsize=pp.size("single", h=1.95))
    lim = np.abs(m).max()
    cmap = mpl.colormaps[pp.DIV_CMAP]
    im = ax.imshow(m, cmap=cmap, vmin=-lim, vmax=lim, aspect="auto")
    for (i, j), v in np.ndenumerate(m):
        ax.text(j, i, f"${v:+.1f}$", ha="center", va="center", fontsize=6.5,
                color=pp.ink_on(cmap((v + lim) / (2 * lim))))
    ax.set_xticks(range(len(corr)), corr, rotation=35, ha="right", rotation_mode="anchor")
    ax.set_yticks(range(len(models)), models)
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.02, ticks=[-10, 0, 10])
    cb.outline.set_linewidth(0.5)
    cb.ax.tick_params(length=0, labelsize=7)
    cb.set_label("accuracy gain (pts)", fontsize=7)
    finish(fig, "07_heatmap")


def violin():
    fig, ax = plt.subplots(figsize=pp.size("single", h=2.15))
    roles = ["base", "a", "b", "ours"]
    names = ["Base", "Baseline A", "Baseline B", "Ours"]
    centers = [51, 60, 62, 71]
    shared = rng.normal(0, 3.5, 24)
    data = [c + shared + rng.normal(0, 2.4, 24) for c in centers]
    parts = ax.violinplot(data, positions=range(4), widths=0.72, showextrema=False)
    for body, role in zip(parts["bodies"], roles):
        body.set_facecolor(pp.ROLES[role].color)
        body.set_alpha(0.28)
        body.set_linewidth(0)
    for i, (d, role) in enumerate(zip(data, roles)):
        jitter = np.random.default_rng(i).uniform(-0.12, 0.12, len(d))
        ax.scatter(i + jitter, d, s=3.5, color=pp.ROLES[role].color, linewidths=0, zorder=3)
        ax.plot([i - 0.2, i + 0.2], [np.median(d)] * 2, color=pp.INK, lw=1.0, zorder=4,
                solid_capstyle="butt")

    def bracket(a, b, y, p):
        ax.plot([a, a, b, b], [y - 0.9, y, y, y - 0.9], color=pp.INK, lw=0.6, zorder=5)
        ax.text((a + b) / 2, y + 0.3, pp.p_text(p), ha="center", va="bottom", fontsize=7)

    bracket(2, 3, 83.5, pp.paired_p(data[3], data[2]))
    bracket(0, 3, 90.0, pp.paired_p(data[3], data[0]))
    ax.set_xticks(range(4), names)
    ax.set(ylim=(36, 97), ylabel="accuracy per run (%)", xlim=(-0.55, 3.55))
    ax.grid(axis="x", visible=False)
    finish(fig, "08_violin_paired_test")


def pareto():
    fig, ax = plt.subplots(figsize=pp.size("single", h=2.1))
    x = 10 ** rng.uniform(0.8, 2.75, 42)
    y = np.minimum(88 - 150 / (x ** 0.72) - np.abs(rng.normal(0, 3.5, 42)) - 2, 81.5)
    ax.scatter(x, y, s=8, color=pp.ROLES["base"].color, alpha=0.55, linewidths=0, zorder=2)
    methods = {"ours": (28, 82.0, "Ours"), "a": (95, 79.8, "Baseline A"), "b": (210, 86.0, "Baseline B"),
               "c": (14, 71.2, "Baseline C")}
    place = {"ours": dict(dx=-5, dy=4, ha="right"), "a": dict(dx=-5, dy=-5, ha="right", halo=True),
             "b": dict(dx=0, dy=6, ha="center", va="bottom"), "c": dict(dx=0, dy=6, ha="center", va="bottom")}
    all_x, all_y = np.r_[x, [m[0] for m in methods.values()]], np.r_[y, [m[1] for m in methods.values()]]
    order, front = np.argsort(all_x), []
    best = -np.inf
    for k in order:
        if all_y[k] > best:
            best = all_y[k]
            front.append(k)
    ax.step(all_x[front], all_y[front], where="post", color=pp.ROLES["oracle"].color, lw=0.8, ls=pp.ROLES["oracle"].ls,
            zorder=3)
    for role, (mx, my, label) in methods.items():
        r = pp.ROLES[role]
        ax.plot([mx], [my], ls="none", marker=r.marker, color=r.color, ms=6.5 if role == "ours" else 5, mec="white",
                mew=0.6, zorder=5)
        pp.label_end(ax, mx, my, label, **place[role])
    pp.label_end(ax, 70, 82.0, "Pareto front", dx=0, dy=2, ha="center", va="bottom", color=pp.MUTED)
    ax.set_xscale("log")
    ax.set(xlim=(6, 620), ylim=(60, 92), xlabel="latency per image (ms, log scale)", ylabel="accuracy (%)")
    ax.set_xticks([10, 100], ["10", "100"])
    finish(fig, "09_pareto_scatter")


def ablation():
    fig, ax = plt.subplots(figsize=pp.size("single", h=1.45))
    names = ["Base model", "+ corrupted views", "+ consistency loss", "+ abstention head", "Full (ours)"]
    vals = [52.4, 61.0, 66.2, 69.1, 74.8]
    errs = [1.2, 1.1, 1.0, 1.1, 0.9]
    shades = [0.16, 0.26, 0.38, 0.52, 0.72]
    y = np.arange(5)
    for k, (v, e, a) in enumerate(zip(vals, errs, shades)):
        pp.bar(ax, k, v, 0.68, "ours", err=e, horizontal=True, shade=a)
        ax.text(v + e + 1.0, k, f"{v:.1f}", va="center", fontsize=7)
    ax.set_yticks(y, names)
    ax.set(xlim=(0, 88), ylim=(4.6, -0.6), xlabel="accuracy (%)")
    ax.grid(axis="y", visible=False)
    finish(fig, "10_ablation")


def performance_profile():
    fig, ax = plt.subplots(figsize=pp.size("single", h=2.0))
    tau = np.linspace(0, 2, 80)
    handles = []
    for role, center, name in [("ours", 1.25, "Ours"), ("a", 0.92, "Baseline A"), ("b", 0.70, "Baseline B"),
                               ("c", 0.48, "Baseline C")]:
        scores = rng.normal(center, 0.33, 150)
        y = np.array([(scores > t).mean() for t in tau])
        band = 0.012 + 1.96 * np.sqrt(y * (1 - y) / 150)
        pp.line(ax, tau, y, role, lo=np.clip(y - band, 0, 1), hi=np.clip(y + band, 0, 1))
        handles.append(pp.key(role, name))
    ax.set(xlim=(0, 2), ylim=(0, 1.03), xlabel=r"normalized score $\tau$",
           ylabel=r"fraction of runs with score $>\tau$", xticks=[0, 0.5, 1, 1.5, 2], yticks=[0, 0.5, 1])
    pp.legend(ax, handles)
    finish(fig, "11_performance_profile")


def sensitivity():
    fig, axes = plt.subplots(1, 3, figsize=pp.size("double", h=1.55), sharey=True)
    sweeps = [(r"loss weight $\lambda$", np.logspace(-3, 1, 7), 0.1, [62, 68, 73, 76, 74.5, 69, 60]),
              (r"temperature $\tau$", np.logspace(-2, 1, 7), 0.5, [66, 70.5, 73, 75.8, 75.0, 72, 68]),
              ("rank $k$", np.array([1, 2, 4, 8, 16, 32, 64]), 16, [58, 64, 69, 72.5, 76, 76.5, 76.8])]
    for i, (ax, (label, x, default, y)) in enumerate(zip(axes, sweeps)):
        y = np.array(y)
        err = 1.6 * np.ones_like(y, dtype=float)
        ax.axhspan(65, 68, color=pp.ROLES["a"].color, alpha=0.16, lw=0, zorder=1)
        ax.axhline(66.5, color=pp.ROLES["a"].color, lw=1.2, zorder=2)
        ax.vlines(default, 65, 82, color=pp.MUTED, lw=0.6, zorder=1)
        pp.line(ax, x, y, "ours", lo=y - err, hi=y + err, marker=True)
        ax.set_xscale("log", base=2 if i == 2 else 10)
        ax.set_xlabel(label)
        ax.minorticks_off()
        ax.set_ylim(54, 82)
        pp.tag(ax, "abc"[i])
        pp.label_end(ax, default, 80.4, "default", dx=2, dy=0, color=pp.MUTED, fontsize=7)
    axes[2].set_xticks([1, 4, 16, 64], ["1", "4", "16", "64"])
    axes[0].set_ylabel("accuracy (%)")
    pp.legend(axes, [pp.key("ours", "Ours"), pp.key("a", "Baseline A (no parameter)")])
    finish(fig, "12_sensitivity")


def composition():
    fig, ax = plt.subplots(figsize=pp.size("single", h=1.62))
    names = ["Ours", "Baseline A", "Baseline B", "Base"]
    labels = ["correct", "correct abstain", "over-abstain", "wrong"]
    colors = ["d", "a", "b", "c"]
    m = np.array([[66, 21, 6, 7], [58, 13, 11, 18], [55, 8, 7, 30], [52, 0, 1, 47]], float)
    left = np.zeros(4)
    for j, (lab, role) in enumerate(zip(labels, colors)):
        col = pp.tint(role, 0.55)
        ax.barh(range(4), m[:, j] - 0.5, left=left + 0.25, height=0.66, color=col, zorder=2)
        for i, v in enumerate(m[:, j]):
            if v >= 8:
                ax.text(left[i] + v / 2, i, f"{v:.0f}", ha="center", va="center", fontsize=7, color=pp.ink_on(col))
        left += m[:, j]
    ax.set_yticks(range(4), names)
    ax.set(xlim=(0, 100), ylim=(3.5, -0.5), xlabel="share of questions (%)")
    ax.grid(False)
    pp.legend(ax, [mpl.patches.Patch(color=pp.tint(r, 0.55), label=l) for r, l in zip(colors, labels)])
    finish(fig, "13_composition")


def before_after():
    x = np.linspace(0.2, 1.0, 9)
    series = {"Ours": (0.02 + 0.13 * x ** 3), "Baseline A": (0.05 + 0.13 * x ** 1.5), "Baseline B": (0.09 + 0.1 * x)}
    width = pp.size("single")[0]

    mpl.rcdefaults()
    fig, ax = plt.subplots(figsize=(width, 2.05), dpi=150)
    for name, y in series.items():
        ax.plot(x, y * 100, label=name)
    ax.set_xlabel("Coverage")
    ax.set_ylabel("Selective Risk (%)")
    ax.set_title("Risk vs. Coverage Comparison of All Methods")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "14a_defaults.pdf")
    fig.savefig(OUT / "14a_defaults.png", dpi=300)
    plt.close(fig)

    pp.use("cvpr")
    fig, ax = plt.subplots(figsize=(width, 2.05))
    for (name, y), role, at in zip(series.items(), ["ours", "a", "b"], (4.0, 9.5, 14.4)):
        pp.line(ax, x, y * 100, role, lo=(y - 0.01) * 100, hi=(y + 0.01) * 100, marker=True)
        ax.text(0.22, at, name, ha="left", va="bottom")
    ax.set(xlim=(0.2, 1.0), ylim=(0, 21), xlabel="coverage", ylabel="risk on answered (%)", yticks=[0, 5, 10, 15, 20])
    finish(fig, "14b_house_style")


def scene(blur=9):
    """Invented picture of a bottle on a table, blurred so that the label cannot be read."""
    h, w = 240, 380
    yy, xx = np.mgrid[0:h, 0:w]
    img = np.dstack([0.78 - 0.2 * yy / h, 0.84 - 0.16 * yy / h, 0.92 - 0.1 * yy / h])
    img[yy > 175] = (0.55, 0.40, 0.27)
    bottle = (np.abs(xx - 190) < 34) & (yy > 62) & (yy < 188) | (np.abs(xx - 190) < 13) & (yy > 22) & (yy <= 62)
    img[bottle] = (0.16, 0.42, 0.34)
    label = (np.abs(xx - 190) < 34) & (yy > 100) & (yy < 150)
    img[label] = (0.93, 0.90, 0.78)
    img[label & (np.abs(yy - 118) < 4) & (np.abs(xx - 190) < 24)] = (0.2, 0.2, 0.25)
    img[label & (np.abs(yy - 133) < 3) & (np.abs(xx - 190) < 18)] = (0.2, 0.2, 0.25)
    out = img.copy()
    for axis in (0, 1):  # three box-blur passes per axis
        for _ in range(3):
            c = np.cumsum(out, axis=axis)
            k = 2 * blur + 1
            c = np.concatenate([np.take(c, [0] * (blur + 1), axis=axis) * 0, c,
                                np.repeat(np.take(c, [-1], axis=axis), blur, axis=axis)], axis=axis)
            out = (np.take(c, range(k, c.shape[axis]), axis=axis) - np.take(c, range(c.shape[axis] - k), axis=axis)) / k
    edge = blur + 1
    return np.clip(out[edge:-edge, edge:-edge], 0, 1)


def method_figure():
    fig, ax = pp.canvas("double", h=2.62)
    pp.note(ax, 0.55, 2.0, "Input image", bold=True)
    pp.image(ax, scene(), 0.08, 1.02, 0.95, 0.86)

    # Existing model, in gray with an ink path.
    pp.arrow(ax, [(1.03, 1.68), (1.45, 2.08)])
    pp.trapezoid(ax, 1.45, 2.08, 0.9, 0.86, 0.44, "Vision\nencoder")
    pp.arrow(ax, [(2.35, 2.08), (4.0, 2.08), (4.0, 1.98)])
    pp.box(ax, 4.45, 2.38, 1.72, 0.18, "Question: What does the label say?", bold=False, size=7)
    pp.arrow(ax, [(5.0, 2.38), (5.0, 1.98)])

    # The added module, in blue with a dashed outline and a blue path.
    pp.arrow(ax, [(1.03, 1.25), (1.42, 0.97)], role="ours")
    pp.box(ax, 1.42, 0.42, 0.96, 1.04, role="ours", dashed=True, shade=0.08)
    pp.trapezoid(ax, 1.52, 0.98, 0.76, 0.84, 0.5, "Answer-\nability\nprobe", role="ours", shade=0.42, size=7)
    pp.note(ax, 1.9, 0.27, "Added module", bold=True)
    pp.arrow(ax, [(2.38, 1.25), (2.72, 1.25), (2.72, 1.88), (2.95, 1.88)], role="ours")

    # Tokens and decoder.
    end = pp.tokens(ax, 3.0, 1.8, 4, "ours")
    pp.box(ax, 2.95, 1.76, end - 2.95 + 0.05, 0.23, role="ours", dashed=True, shade=0, radius=0.03, zorder=1)
    end = pp.tokens(ax, 3.6, 1.8, 8, "base")
    pp.tokens(ax, end + 0.13, 1.8, 7, "base", hollow=True)
    pp.arrow(ax, [(4.13, 1.76), (4.13, 1.62)])
    pp.box(ax, 2.95, 1.36, 2.4, 0.26, "Language decoder")

    # Scores, with a real example inside each card.
    pp.arrow(ax, [(3.35, 1.36), (3.35, 1.15)], role="ours")
    pp.arrow(ax, [(4.95, 1.36), (4.95, 1.15)])
    pp.note(ax, 2.95, 0.78, r"$\gamma\,\log\,($", size=8, ha="right")
    pp.score_card(ax, 2.97, 0.42, 0.76, 0.72, [("unclear", "0.70"), ("Acme", "0.20"), ("blue", "0.10")], "ours")
    pp.note(ax, 4.15, 0.78, r"$)\;+\;(1-\gamma)\,\log\,($", size=8)
    pp.score_card(ax, 4.57, 0.42, 0.76, 0.72, [("Acme", "0.50", True), ("unclear", "0.30"), ("blue", "0.20")], "base")
    pp.note(ax, 5.35, 0.78, r"$)$", size=8, ha="left")
    pp.arrow(ax, [(5.43, 0.78), (5.6, 0.78)])
    pp.score_card(ax, 5.6, 0.42, 0.76, 0.72, [("unclear", "0.54"), ("Acme", "0.31"), ("blue", "0.14")], "ours", strong=True)

    # Readout of the final answer.
    pp.note(ax, 6.42, 1.0, "The\nlabel\nis", ha="left", color=pp.MUTED, linespacing=1.05)
    pp.note(ax, 6.42, 0.74, "unclear", bold=True, ha="left")
    pp.arrow(ax, [(3.35, 0.42), (3.35, 0.14), (5.98, 0.14), (5.98, 0.42)], role="ours")
    pp.note(ax, 4.66, 0.27, "Mix the scores before choosing a word", bold=True, size=7)
    finish(fig, "15_method_figure")


def qualitative_example():
    H = 2.16
    fig, ax = pp.canvas("double", h=H)
    left, right, top = 0.04, 2.62, H - 0.04
    pp.image(ax, scene()[10:190], left, top - 1.22, 2.44, 1.22)
    reference = ["The image is too blurred to read the label. A tall green bottle stands on a table in front of a window, but",
                 "no letters can be made out, so the correct response is that the text cannot be read."]
    pp.text_card(ax, left, top - 1.26, 2.44, "Reference answer", reference, "b", "R")

    y = pp.text_card(ax, right, top, 4.2, None, ["Question: What is written on the label of the bottle?"], "c", shade=0.12)
    pp.badge(ax, right + 4.08, top - 0.1, "Q", "c")
    base = ["The label on the bottle reads", ("\"Acme Spring Water\"", "error"), "in", ("dark blue letters,", "error"),
            "which shows that the drink is bottled water. The", ("small mountain logo below the brand name", "error"),
            "suggests that the water comes from a natural spring. The bottle stands on a wooden table next to a window,",
            "and the light from outside falls across its front."]
    y = pp.text_card(ax, right, y - 0.08, 4.2, "Base model after robust fine-tuning", base, "base", "B", shade=0.2)
    ours = ["The label is too blurred for me to read, so I cannot say what is written on it. I can see a tall green bottle",
            "with a pale label, standing on a wooden table in front of a window. A sharper photo of the label would let me",
            "answer the question, because the letters on this one cannot be told apart at all."]
    pp.text_card(ax, right, y - 0.08, 4.2, "Ours", ours, "ours", "O", shade=0.2)
    finish(fig, "16_qualitative_example")


def main():
    OUT.mkdir(exist_ok=True)
    pp.use("cvpr")
    demos = [learning_curves, risk_coverage, grouped_bars, aggregate_intervals, dumbbell, reliability, heatmap, violin,
             pareto, ablation, performance_profile, sensitivity, composition, before_after, method_figure,
             qualitative_example]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for demo in demos:
            demo()
            pp.use("cvpr")
    for name in ("01_learning_curves", "03_grouped_bars"):
        pp.print_check(OUT / f"{name}.png")
    lines = [f"{name}: {'OK' if not issues else '; '.join(issues)}" for name, issues in report.items()]
    other = sorted({str(w.message) for w in caught if "Glyph" in str(w.message) or "font" in str(w.message).lower()})
    (OUT / "lint_report.txt").write_text("\n".join(lines + other) + "\n")
    print("\n".join(lines + other))


if __name__ == "__main__":
    main()
