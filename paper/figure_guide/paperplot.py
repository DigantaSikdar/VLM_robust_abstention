"""paperplot: house style for NeurIPS / ICML / ICLR / CVPR figures.

The rules behind every default are in GUIDELINE.md. Only numpy and matplotlib are required.

    import paperplot as pp
    pp.use("cvpr")
    fig, ax = plt.subplots(figsize=pp.size("single", h=2.0))
    pp.line(ax, x, y, "ours", label="Ours", lo=y_lo, hi=y_hi)
    pp.save(fig, "figures/my_plot")     # writes .pdf and .png, then runs the checks
"""
from __future__ import annotations

import logging
import re
import shutil
import subprocess
import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl
import matplotlib.patheffects as patheffects
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.image import imread, imsave
from matplotlib.collections import PathCollection
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Patch, Polygon, Rectangle
from matplotlib.path import Path as MplPath
from matplotlib.textpath import text_to_path
from matplotlib.transforms import Bbox

# Text widths in inches, measured from each venue's official style file.
# (column width, full text width)
VENUES = {
    "cvpr": (3.28125, 6.875),
    "iccv": (3.28125, 6.875),
    "icml": (3.25, 6.75),
    "neurips": (5.5, 5.5),
    "iclr": (5.5, 5.5),
}

INK = "#1A1A1A"
MUTED = "#666666"
GRID = "#E4E4E4"
SEQ_CMAP = "viridis"
DIV_CMAP = "BrBG"  # teal = better, brown = worse; avoids red/green, which colorblind readers confuse


@dataclass(frozen=True)
class Role:
    color: str
    marker: str
    ls: object
    hatch: str
    lw: float = 1.2


# Fixed slot order: colour follows the method across the whole paper.
# Each slot has a redundant marker, dash and hatch so it survives grayscale printing.
ROLES = {
    "ours": Role("#2E7BD6", "o", "-", "", 1.7),
    "a": Role("#C9302C", "s", "-", "////"),
    "b": Role("#BF8700", "^", (0, (5, 1.5)), "\\\\\\\\"),
    "c": Role("#A23BA6", "D", (0, (3, 1, 1, 1)), "...."),
    "d": Role("#159E86", "v", (0, (1, 1)), "xxxx"),
    "base": Role("#8C8C8C", "o", "-", "", 1.0),
    "oracle": Role("#333333", "", (0, (4, 2)), "", 1.0),
}

# Bars use light shades. Ours keeps a slightly stronger shade so it still reads first.
BAR_SHADE = 0.46
OURS_BAR_SHADE = 0.66

logging.getLogger("fontTools").setLevel(logging.ERROR)  # silences harmless cmr10 timestamp messages

MIN_FONT_PT = 6.0
MIN_LINE_PT = 0.5  # ICML template: lines at least 0.5 pt
_venue = "cvpr"


def use(venue: str = "cvpr", font: str = "cm", base: float = 8.0, tex: bool = False) -> None:
    """Apply the house rcParams. font: 'cm' (serif, for papers) or 'sans' (for slides and posters).

    tex=True renders text through LaTeX, which embeds Type 1 fonts. It needs a LaTeX install.
    """
    global _venue
    if venue not in VENUES:
        raise ValueError(f"unknown venue {venue!r}; choose from {sorted(VENUES)}")
    _venue = venue
    mpl.rcdefaults()
    if font == "cm":
        family = {
            "font.family": "serif",
            "font.serif": ["cmr10"],
            "mathtext.fontset": "cm",
            "axes.formatter.use_mathtext": True,
            "axes.unicode_minus": False,  # cmr10 has no U+2212
        }
    elif font == "sans":
        family = {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "mathtext.fontset": "dejavusans",
        }
    else:
        raise ValueError("font must be 'cm' or 'sans'")
    mpl.rcParams.update(family)
    mpl.rcParams.update({
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],  # qualitative cards use sans at any setting
        "font.size": base,
        "axes.labelsize": base,
        "axes.titlesize": base,
        "legend.fontsize": base - 0.5,
        "xtick.labelsize": base - 1,
        "ytick.labelsize": base - 1,
        "axes.linewidth": 0.6,
        "axes.edgecolor": INK,
        "axes.labelcolor": INK,
        "axes.labelpad": 2.0,
        "axes.titlepad": 3.0,
        "axes.axisbelow": True,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.5,
        "xtick.major.size": 0,
        "ytick.major.size": 0,
        "xtick.minor.size": 0,
        "ytick.minor.size": 0,
        "xtick.major.pad": 2.5,
        "ytick.major.pad": 2.5,
        "xtick.color": INK,
        "ytick.color": INK,
        "lines.linewidth": 1.2,
        "lines.markersize": 3.4,
        "lines.markeredgewidth": 0.0,
        "patch.linewidth": 0.0,
        "hatch.linewidth": 0.5,
        "legend.frameon": False,
        "legend.borderaxespad": 0.6,
        "legend.borderpad": 0.0,
        "legend.handlelength": 2.0,
        "legend.handletextpad": 0.5,
        "legend.columnspacing": 1.2,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "figure.constrained_layout.use": True,
        "figure.constrained_layout.w_pad": 0.02,
        "figure.constrained_layout.h_pad": 0.02,
        "figure.constrained_layout.wspace": 0.03,
        "figure.constrained_layout.hspace": 0.03,
        "pdf.fonttype": 42,  # embedded TrueType, never Type 3
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "axes.prop_cycle": mpl.cycler(color=[ROLES[r].color for r in ("ours", "a", "b", "c", "d")]),
        "text.usetex": tex,
    })


def size(span: str = "single", h: float | None = None, aspect: float = 0.62, frac: float = 1.0) -> tuple:
    """Exact figure size in inches. span: 'single' (one column) or 'double' (full text width).

    Include the PDF with \\includegraphics[width=\\linewidth] (or \\textwidth) so it prints 1:1.
    On one-column venues (NeurIPS, ICLR) 'single' still means a narrow panel: 60% of the text width.
    """
    col, full = VENUES[_venue]
    if col == full and span == "single":
        col = round(0.6 * full, 3)
    width = (col if span == "single" else full) * frac
    return (width, h if h is not None else round(width * aspect, 3))


def line(ax, x, y, role="ours", label=None, lo=None, hi=None, marker=False, markevery=None, **kw):
    """Line with optional uncertainty band. The band is the same hue at 18% opacity, with no edge."""
    r = ROLES[role]
    x, y = np.asarray(x), np.asarray(y)
    if lo is not None:
        ax.fill_between(x, lo, hi, color=r.color, alpha=0.18, linewidth=0, zorder=2)
    opts = dict(color=r.color, ls=r.ls, lw=r.lw, label=label, zorder=4 if role == "ours" else 3,
                solid_capstyle="round")
    if marker and r.marker:
        opts.update(marker=r.marker, markevery=markevery)
    opts.update(kw)
    return ax.plot(x, y, **opts)[0]


def tint(role_or_color, shade=BAR_SHADE):
    """Opaque light shade of a role or colour, mixed with white. shade=1 is the full colour."""
    color = ROLES[role_or_color].color if role_or_color in ROLES else role_or_color
    return tuple(shade * c + (1 - shade) for c in mpl.colors.to_rgb(color))


def _bar_shade(role, shade=None):
    return shade if shade is not None else (OURS_BAR_SHADE if role == "ours" else BAR_SHADE)


def bar(ax, x, h, width, role="ours", err=None, horizontal=False, shade=None, **kw):
    """Light-shaded bar from zero. The hatch uses the full role colour, and whiskers are thin and dark."""
    r = ROLES[role]
    # Mixed with white rather than transparent, so gridlines do not show through the bar.
    opts = dict(color=tint(role, _bar_shade(role, shade)), hatch=r.hatch, edgecolor=r.color, linewidth=0, zorder=2)
    opts.update(kw)
    if horizontal:
        out = ax.barh(x, h, height=width, **opts)
    else:
        out = ax.bar(x, h, width=width, **opts)
    if err is not None:
        e = dict(fmt="none", ecolor=INK, elinewidth=0.6, capsize=1.3, capthick=0.6, zorder=3)
        if horizontal:
            ax.errorbar(h, x, xerr=err, **e)
        else:
            ax.errorbar(x, h, yerr=err, **e)
    return out


def key(role, label, kind="line"):
    """Legend handle for a role. kind: 'line', 'marker', or 'patch'."""
    r = ROLES[role]
    if kind == "patch":
        return Patch(facecolor=tint(role, _bar_shade(role)), hatch=r.hatch, edgecolor=r.color, linewidth=0, label=label)
    if kind == "marker":
        return Line2D([], [], ls="none", marker=r.marker, color=r.color, markersize=4.5, label=label)
    return Line2D([], [], color=r.color, ls=r.ls, lw=r.lw, label=label)


_LEGEND_LOCS = ("best", "upper right", "upper left", "lower right", "lower left", "upper center", "lower center",
                "center right", "center left")


def _legend_hits(ax, leg, renderer):
    """True if the legend box covers a line, marker, band, bar, or text in its axes."""
    leg.draw(renderer)
    b = leg.get_window_extent(renderer)
    box = Bbox.from_extents(b.x0 - 1, b.y0 - 1, b.x1 + 1, b.y1 + 1)
    for ln in ax.get_lines():
        path = ln.get_transform().transform_path(ln.get_path())
        if ln.get_linestyle() not in ("None", "none", "") and ln.get_linewidth() > 0 and path.intersects_bbox(box, filled=False):
            return True
        if ln.get_marker() not in (None, "None", "none", ""):
            if any(box.contains(*v) for v in path.vertices):
                return True
    for patch in ax.patches:
        if patch.get_window_extent(renderer).overlaps(box):
            return True
    for coll in ax.collections:
        if isinstance(coll, PathCollection):
            if any(box.contains(*v) for v in coll.get_offset_transform().transform(coll.get_offsets())):
                return True
        else:
            trans = coll.get_transform()
            if any(trans.transform_path(q).intersects_bbox(box, filled=True) for q in coll.get_paths()):
                return True
    return any(t.get_window_extent(renderer).overlaps(box) for t in ax.texts)


def legend(axes, handles=None, ncols=None, **kw):
    """One legend at the best location: the first free spot inside the plot, checked against the data.

    Call it last, after limits are set. With several panels it uses whichever panel has room.
    If nothing is free, for example on stacked bars that fill the frame, it goes below the panels.
    """
    axes = list(np.atleast_1d(axes).flat)
    fig = axes[0].figure
    if handles is None:
        handles = axes[0].get_legend_handles_labels()[0]
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    opts = dict(frameon=True, facecolor="white", edgecolor="none", framealpha=1.0, borderpad=0.25)
    opts.update(kw)
    for ncol in ([ncols] if ncols else range(1, min(len(handles), 3) + 1)):
        for ax in axes:
            for loc in _LEGEND_LOCS:
                leg = ax.legend(handles=handles, loc=loc, ncol=ncol, **opts)
                if not _legend_hits(ax, leg, renderer):
                    return leg
                leg.remove()
    for ncol in range(len(handles), 0, -1):  # widest row that still fits the figure
        leg = legend_below(fig, handles, ncol=ncol)
        fig.canvas.draw()
        if ncol == 1 or leg.get_window_extent(renderer).width < 0.98 * fig.bbox.width:
            return leg
        leg.remove()


def legend_below(fig, handles, ncol=None):
    """One shared frameless legend below all panels."""
    return fig.legend(handles=handles, loc="outside lower center", ncol=ncol or len(handles), borderaxespad=0.0)


def shared_xlabel(axes, text):
    """One x-label centred under a row of panels. Unlike fig.supxlabel, it cannot hit legend_below."""
    axes = list(axes)
    mid = axes[(len(axes) - 1) // 2]
    mid.set_xlabel(text)
    fig = mid.figure
    fig.draw_without_rendering()
    box = mid.get_position()
    centre = (axes[0].get_position().x0 + axes[-1].get_position().x1) / 2
    mid.xaxis.label.set_x((centre - box.x0) / box.width)


def label_end(ax, x, y, text, dx=3, dy=0, ha="left", va="center", halo=False, **kw):
    """Direct label next to a mark, written in ink rather than the series colour.

    halo=True adds a thin white outline for labels that have to sit over scatter points.
    """
    if halo:
        kw["path_effects"] = [patheffects.withStroke(linewidth=1.6, foreground="white")]
    return ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points", ha=ha, va=va,
                       color=kw.pop("color", INK), annotation_clip=False, **kw)


def tag(ax, letter, x=-0.02, y=1.03):
    """Bold panel tag, placed above the top-left corner of the axes."""
    return ax.text(x, y, rf"$\mathbf{{({letter})}}$", transform=ax.transAxes, ha="right", va="bottom")


def ink_on(color) -> str:
    """Black or white text, whichever is readable on `color`."""
    r, g, b = _to_linear(np.array(mpl.colors.to_rgb(color)))
    return INK if 0.2126 * r + 0.7152 * g + 0.0722 * b > 0.179 else "white"  # WCAG contrast crossover


# Method figures and qualitative examples -----------------------------------------------------

def _bold():
    """Bold text settings. Computer Modern has a separate bold font."""
    serif = mpl.rcParams["font.family"][0] == "serif" and not mpl.rcParams["text.usetex"]
    return {"family": "cmb10"} if serif else {"weight": "bold"}


def canvas(span="double", h=2.5):
    """Blank drawing area at exact print size, with coordinates in inches from the bottom-left corner."""
    w = size(span)[0]
    fig = plt.figure(figsize=(w, h), layout="none")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, w), ylim=(0, h), xticks=[], yticks=[])
    ax.set_axis_off()
    return fig, ax


def note(ax, x, y, text, bold=False, size=7.5, ha="center", va="center", color=INK, **kw):
    """Free text on a canvas."""
    return ax.text(x, y, text, ha=ha, va=va, fontsize=size, color=color, **(_bold() if bold else {}), **kw)


def box(ax, x, y, w, h, text="", role="base", dashed=False, bold=True, size=7.5, shade=0.2, radius=0.05, zorder=2):
    """Rounded component. role='base' is existing machinery, 'ours' is what the paper adds."""
    color = ROLES[role].color
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={radius}", fc=tint(role, shade), ec=color,
                                lw=0.7, ls=(0, (3, 2)) if dashed else "-", zorder=zorder))
    if text:
        note(ax, x + w / 2, y + h / 2, text, bold=bold, size=size, zorder=zorder + 1)


def trapezoid(ax, x, cy, w, left, right, text="", role="base", bold=True, size=7.5, shade=0.2):
    """Encoder-style block, centred at height cy, with the given left and right edge heights."""
    pts = [(x, cy - left / 2), (x + w, cy - right / 2), (x + w, cy + right / 2), (x, cy + left / 2)]
    ax.add_patch(Polygon(pts, closed=True, fc=tint(role, shade), ec=ROLES[role].color, lw=0.7, zorder=2))
    if text:
        note(ax, x + w / 2, cy, text, bold=bold, size=size, zorder=3)


def arrow(ax, points, role=None, lw=1.3, dashed=False):
    """Arrow along a polyline. No role draws the existing path in ink; role='ours' draws the added path."""
    color = ROLES[role].color if role else INK
    ax.add_patch(FancyArrowPatch(path=MplPath(points), arrowstyle="-|>,head_length=3.6,head_width=1.9", mutation_scale=1,
                                 color=color, lw=lw, ls=(0, (3, 2)) if dashed else "-", joinstyle="miter",
                                 capstyle="butt", zorder=1))


def tokens(ax, x, y, n, role="base", w=0.085, h=0.15, gap=0.022, shade=0.55, hollow=False):
    """A row of n small token boxes; returns the x position after the last one."""
    for i in range(n):
        box(ax, x + i * (w + gap), y, w, h, role=role, shade=0 if hollow else shade, radius=0.025)
    return x + n * (w + gap) - gap


def score_card(ax, x, y, w, h, rows, role="base", strong=False, size=7):
    """Card of (label, value) rows so the figure shows real output. Add True as a third item to mark an error."""
    box(ax, x, y, w, h, role=role, shade=0.5 if strong else 0.2, radius=0.06)
    step = h / (len(rows) + 0.3)
    for i, row in enumerate(rows):
        error = len(row) > 2 and row[2]
        note(ax, x + w / 2, y + h - step * (i + 0.65), f"{row[0]}: {row[1]}", bold=True, size=size, zorder=3,
             color=ROLES["a"].color if error else INK)


def image(ax, array, x, y, w, h):
    """Place a picture on a canvas at an exact position."""
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    ax.imshow(array, extent=(x, x + w, y, y + h), aspect="auto", zorder=2, interpolation="bilinear")
    ax.set(xlim=xlim, ylim=ylim)


def badge(ax, x, y, letter, role="base", r=0.075):
    """Small round speaker mark for a question or a model response."""
    ax.add_patch(Circle((x, y), r, fc=ROLES[role].color, ec="white", lw=0.8, zorder=5))
    ax.text(x, y - 0.002, letter, ha="center", va="center", fontsize=6.2, color=ink_on(ROLES[role].color), zorder=6,
            family="sans-serif", weight="bold")


def text_card(ax, x, top, w, title, spans, role="base", letter=None, size=7.0, shade=0.2, pad=0.07):
    """Tinted response card with a bold title and wrapped text; returns the y of its bottom edge.

    spans is a list of strings or (string, style), where style is 'error' (red and underlined) or 'bold'.
    """
    plain, heavy = (FontProperties(family="sans-serif", size=size, weight=w_) for w_ in ("normal", "bold"))

    def width(word, prop):
        return text_to_path.get_text_width_height_descent(word, prop, False)[0] / 72

    space = width("n n", plain) - width("nn", plain)
    words = []
    for span in spans:
        text, style = span if isinstance(span, tuple) else (span, None)
        words += [(word, style) for word in text.split()]
    limit = w - 2 * pad
    lines, current, used = [], [], 0.0
    for word, style in words:
        wide = width(word, heavy if style == "bold" else plain)
        if current and used + wide > limit:
            lines.append(current)
            current, used = [], 0.0
        current.append((word, style, wide))
        used += wide + space
    if current:
        lines.append(current)
    lead = size * 1.27 / 72
    head = lead * 1.25 if title else 0.0
    height = 2 * pad + head + lead * len(lines)
    ax.add_patch(Rectangle((x, top - height), w, height, fc=tint(role, shade), ec="none", zorder=1))
    y = top - pad
    if title:
        ax.text(x + pad + (0.16 if letter else 0), y - lead / 2, title, ha="left", va="center", fontsize=size, color=INK,
                family="sans-serif", weight="bold", zorder=3)
        if letter:
            badge(ax, x + pad + 0.05, y - lead / 2, letter, role)
        y -= head
    error = ROLES["a"].color
    for row in lines:
        cx = x + pad
        for word, style, wide in row:
            ax.text(cx, y - lead / 2, word, ha="left", va="center", fontsize=size, family="sans-serif", zorder=3,
                    color=error if style == "error" else INK, weight="bold" if style == "bold" else "normal")
            if style == "error":
                ax.add_patch(Rectangle((cx, y - lead + 0.006), wide + space, 0.007, fc=error, ec="none", zorder=3))
            cx += wide + space
        y -= lead
    return top - height


# Statistics -----------------------------------------------------------------------------------

def boot_ci(samples, stat=np.mean, n=2000, level=0.95, seed=0):
    """Percentile bootstrap: returns (point, low, high). Resamples along axis 0 (runs/seeds)."""
    rng = np.random.default_rng(seed)
    samples = np.asarray(samples)
    idx = rng.integers(0, len(samples), (n, len(samples)))
    stats = np.array([stat(samples[i]) for i in idx])
    a = 100 * (1 - level) / 2
    return stat(samples), np.percentile(stats, a), np.percentile(stats, 100 - a)


def iqm(x):
    """Interquartile mean: the mean of the middle 50% of values (Agarwal et al., NeurIPS 2021)."""
    x = np.sort(np.ravel(x))
    k = len(x) // 4
    return x[k:len(x) - k].mean()


def paired_p(a, b, n=20000, seed=0):
    """Two-sided paired sign-flip permutation test on the mean difference."""
    rng = np.random.default_rng(seed)
    d = np.asarray(a) - np.asarray(b)
    flips = rng.choice([-1.0, 1.0], (n, len(d)))
    null = np.abs((flips * d).mean(1))
    return (np.sum(null >= abs(d.mean()) - 1e-12) + 1) / (n + 1)


def p_text(p):
    return r"$p<0.001$" if p < 0.001 else rf"$p={p:.3f}$"


# Export and checks ----------------------------------------------------------------------------

def save(fig, stem, png=True, check=True):
    """Write stem.pdf (and stem.png), keeping the exact figure size. Never crops to a tight bbox."""
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    problems = lint(fig) if check else []
    pdf = stem.with_suffix(".pdf")
    fig.savefig(pdf, metadata={"CreationDate": None, "Producer": None})
    if png:
        fig.savefig(stem.with_suffix(".png"), dpi=300)
    if check:
        problems += check_pdf_fonts(pdf)
    for p in problems:
        warnings.warn(f"{stem.name}: {p}", stacklevel=2)
    return problems


def _visible_texts(fig):
    out = []
    for ax in fig.axes:
        for axis, lim in ((ax.xaxis, ax.get_xlim()), (ax.yaxis, ax.get_ylim())):
            low, high = min(lim), max(lim)
            for t, loc in zip(axis.get_ticklabels(), axis.get_ticklocs()):
                if low <= loc <= high:
                    out.append(t)
            out.append(axis.label)
        out += list(ax.texts) + [ax.title, ax._left_title, ax._right_title]
        if ax.get_legend():
            out += list(ax.get_legend().get_texts())
    for leg in fig.legends:
        out += list(leg.get_texts())
    out += list(fig.texts)
    for sup in (fig._supxlabel, fig._supylabel, fig._suptitle):
        if sup is not None:
            out.append(sup)
    unique = {id(t): t for t in out}
    return [t for t in unique.values() if t.get_visible() and t.get_text().strip()]


def lint(fig):
    """Mechanical checks: font size, line width, clipped text, text collisions, text over lines, legends over data."""
    problems = []
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    texts = _visible_texts(fig)
    boxes = [t.get_window_extent(renderer) for t in texts]
    w, h = fig.canvas.get_width_height()
    xticks = {id(t) for ax in fig.axes for t in ax.get_xticklabels()}
    yticks = {id(t) for ax in fig.axes for t in ax.get_yticklabels()}
    for t, b in zip(texts, boxes):
        if t.get_fontsize() < MIN_FONT_PT:
            problems.append(f"font {t.get_fontsize():.1f} pt below {MIN_FONT_PT} pt: {t.get_text()!r}")
        if b.x0 < -0.5 or b.y0 < -0.5 or b.x1 > w + 0.5 or b.y1 > h + 0.5:
            problems.append(f"text outside the figure: {t.get_text()!r}")
    for i, a in enumerate(boxes):
        for j in range(i + 1, len(boxes)):
            if texts[i].get_rotation() % 90 or texts[j].get_rotation() % 90:
                continue  # the axis-aligned box of rotated text is much larger than its ink
            if (id(texts[i]) in xticks and id(texts[j]) in yticks) or (id(texts[i]) in yticks and id(texts[j]) in xticks):
                continue  # x and y tick labels meeting at the origin corner
            if a.overlaps(boxes[j]):
                overlap_x = min(a.x1, boxes[j].x1) - max(a.x0, boxes[j].x0)
                overlap_y = min(a.y1, boxes[j].y1) - max(a.y0, boxes[j].y0)
                if overlap_x > 3 and overlap_y > 3:  # px at figure.dpi; ignores tick labels meeting at a corner
                    problems.append(f"text overlap: {texts[i].get_text()!r} / {texts[j].get_text()!r}")
    for ax in fig.axes:
        frame = ax.get_window_extent(renderer)
        if ax.get_legend() and _legend_hits(ax, ax.get_legend(), renderer):
            problems.append("legend covers data")
        for ln in ax.get_lines():
            if ln.get_linestyle() != "None" and 0 < ln.get_linewidth() < MIN_LINE_PT:
                problems.append(f"line {ln.get_linewidth()} pt below {MIN_LINE_PT} pt")
        for t in ax.texts:
            b = t.get_window_extent(renderer)
            if b.x0 < frame.x1 < b.x1 or b.x0 < frame.x0 < b.x1 or b.y0 < frame.y1 < b.y1 or b.y0 < frame.y0 < b.y1:
                problems.append(f"text crosses the axes frame: {t.get_text()!r}")
            inner = Bbox.from_extents(b.x0 + 1, b.y0 + 1, b.x1 - 1, b.y1 - 1)
            for ln in ax.get_lines():
                if ln.get_linestyle() == "None" or ln.get_linewidth() == 0:
                    continue
                if ln.get_transform().transform_path(ln.get_path()).intersects_bbox(inner, filled=False):
                    problems.append(f"text sits on a line: {t.get_text()!r}")
                    break
    return problems


def check_pdf_fonts(pdf):
    """Flag Type 3 or unembedded fonts in a figure PDF. Skipped if pdffonts is not installed."""
    if not shutil.which("pdffonts"):
        return []
    rows = subprocess.run(["pdffonts", str(pdf)], capture_output=True, text=True).stdout.splitlines()[2:]
    problems = []
    for row in rows:
        m = re.search(r"\s(yes|no)\s+(yes|no)\s+(yes|no)\s+\d+\s+\d+\s*$", row)
        if "Type 3" in row:
            problems.append(f"Type 3 font: {row.split()[0]}")
        elif m and m.group(1) == "no":
            problems.append(f"font not embedded: {row.split()[0]}")
    return problems


def _to_linear(c):
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _to_srgb(c):
    c = np.clip(c, 0, 1)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * c ** (1 / 2.4) - 0.055)


# Machado, Oliveira & Fernandes (2009), severity 1.0.
_DEUTAN = np.array([[0.367322, 0.860646, -0.227968],
                    [0.280085, 0.672501, 0.047413],
                    [-0.011820, 0.042940, 0.968881]])


def print_check(png, out=None):
    """Write the original, grayscale, and deuteranopia versions of a PNG side by side."""
    img = imread(str(png))
    rgb = img[..., :3]
    if img.shape[-1] == 4:
        rgb = rgb * img[..., 3:] + (1 - img[..., 3:])
    linear = _to_linear(rgb.astype(np.float64))
    luma = (linear * np.array([0.2126, 0.7152, 0.0722])).sum(-1)
    gray = np.repeat(_to_srgb(luma)[..., None], 3, axis=-1)
    deutan = _to_srgb(np.stack([(linear * row).sum(-1) for row in _DEUTAN], -1))
    gap = np.ones((rgb.shape[0], 40, 3))
    out = Path(out) if out else Path(png).with_name(Path(png).stem + "_printcheck.png")
    imsave(out, np.clip(np.hstack([rgb, gap, gray, gap, deutan]), 0, 1))
    return out


__all__ = ["use", "size", "line", "bar", "tint", "key", "legend", "legend_below", "canvas", "note", "box", "trapezoid",
           "arrow", "tokens", "score_card", "image", "badge", "text_card", "shared_xlabel", "label_end", "tag", "ink_on", "boot_ci",
           "iqm", "paired_p", "p_text", "save", "lint", "check_pdf_fonts", "print_check", "ROLES", "VENUES",
           "INK", "MUTED", "GRID", "SEQ_CMAP", "DIV_CMAP"]

_ = plt  # keep pyplot imported so `paperplot` can be the only style import in small scripts
