"""Teaser figure v2 (icons + charts, fewer words): problem -> diagnosis -> fix. SVG, PNG, PDF, draw.io.

    python make_teaser_v2.py <photo.jpg> <out_dir>

Every number below comes from the evaluation outputs in Vision_self_distillation/outputs
(see the NUMBERS block); the example responses are verbatim excerpts for VizWiz_val_00002295.
"""
import base64
import html
import io
import sys
from pathlib import Path

from PIL import Image, ImageFont

PHOTO, OUT = Path(sys.argv[1]), Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- palette / type
INK, MUTED, RULE, PANEL = "#1F2328", "#5A6470", "#D3D8DE", "#FAFBFC"
BASE, RKL, GRPO, FIX = "#5B6B7F", "#C8571B", "#2B6CB0", "#0E8A6A"
BAD, GOOD = "#C62828", "#1E7B3A"
FONT_DIR = "/usr/share/fonts/truetype/liberation/"
FAMILY = "Helvetica, Arial, Liberation Sans, sans-serif"
W, H = 880, 452

_fonts = {}


def font(size, b=False, i=False):
    name = "LiberationSans-" + ({(0, 0): "Regular", (1, 0): "Bold", (0, 1): "Italic", (1, 1): "BoldItalic"}[(b, i)])
    key = (name, size)
    if key not in _fonts:
        _fonts[key] = ImageFont.truetype(FONT_DIR + name + ".ttf", size=round(size * 4))
    return _fonts[key]


def width(text, size, b=False, i=False):
    return font(size, b, i).getlength(text) / 4


# ---------------------------------------------------------------- scene graph
svg, cells = [], []
_id = [1]


def nid():
    _id[0] += 1
    return f"c{_id[0]}"


def rect(x, y, w, h, fill="none", stroke="none", sw=1.0, rx=0, dash=False):
    d = ' stroke-dasharray="4 3"' if dash else ""
    svg.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" '
               f'stroke="{stroke}" stroke-width="{sw}"{d}/>')
    style = (f"rounded={1 if rx else 0};absoluteArcSize=1;arcSize={2 * rx};html=1;"
             f"fillColor={fill if fill != 'none' else 'none'};strokeColor={stroke if stroke != 'none' else 'none'};"
             f"strokeWidth={sw};{'dashed=1;dashPattern=4 3;' if dash else ''}")
    cells.append(("v", style, "", x, y, w, h))


def circle(cx, cy, r, fill, stroke="none"):
    svg.append(f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{fill}" stroke="{stroke}"/>')
    cells.append(("v", f"ellipse;html=1;fillColor={fill};strokeColor={stroke};", "", cx - r, cy - r, 2 * r, 2 * r))


def line(x1, y1, x2, y2, color=INK, sw=1.2, arrow=False, dash=False):
    d = ' stroke-dasharray="4 3"' if dash else ""
    m = ' marker-end="url(#ah)"' if arrow else ""
    svg.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{sw}"{d}{m}/>')
    style = (f"html=1;endArrow={'block;endFill=1;endSize=5' if arrow else 'none'};startArrow=none;"
             f"strokeColor={color};strokeWidth={sw};{'dashed=1;dashPattern=4 3;' if dash else ''}")
    cells.append(("e", style, (x1, y1, x2, y2)))


def image(x, y, w, h, uri):
    svg.append(f'<image x="{x}" y="{y}" width="{w}" height="{h}" href="{uri}" preserveAspectRatio="none"/>')
    cells.append(("v", "shape=image;html=1;imageAspect=0;aspect=fixed;image=" + uri.replace(";base64", ""),
                  "", x, y, w, h))


def icon(kind, x, y, s=11):
    """Check mark or cross inside a filled circle (vector in both outputs)."""
    col = GOOD if kind == "ok" else BAD
    path = ("M3.2 6.3 L5.3 8.4 L8.9 3.9" if kind == "ok" else "M3.8 3.8 L8.2 8.2 M8.2 3.8 L3.8 8.2")
    body = (f'<circle cx="6" cy="6" r="6" fill="{col}"/><path d="{path}" fill="none" stroke="#fff" '
            f'stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"/>')
    svg.append(f'<g transform="translate({x},{y}) scale({s / 12})">{body}</g>')
    uri = "data:image/svg+xml," + base64.b64encode(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 12 12">{body}</svg>'.encode()).decode()
    cells.append(("v", f"shape=image;html=1;imageAspect=0;image={uri}", "", x, y, s, s))


# rich text: a paragraph is a list of runs (text, style); style keys b, i, c (colour)
def wrap(runs, size, maxw):
    lines, cur, curw = [], [], 0.0
    for text, st in runs:
        for k, word in enumerate(text.split(" ")):
            if word == "" and k:
                continue
            tok = word if not cur or (curw == 0) else " " + word
            wd = width(tok, size, st.get("b", False), st.get("i", False))
            if cur and curw + wd > maxw:
                lines.append(cur)
                cur, curw = [], 0.0
                tok = word
                wd = width(tok, size, st.get("b", False), st.get("i", False))
            if cur and cur[-1][1] == st:
                cur[-1] = (cur[-1][0] + tok, st)
            else:
                cur.append((tok, st))
            curw += wd
    if cur:
        lines.append(cur)
    return lines


def text(x, y, runs, size=12.5, maxw=None, color=INK, lh=1.25, anchor="start"):
    """Draw (optionally wrapped) rich text with its top-left at (x, y); returns the height."""
    if isinstance(runs, str):
        runs = [(runs, {})]
    lines = wrap(runs, size, maxw) if maxw else [runs]
    asc = size * 0.80
    tsp = []
    for li, ln in enumerate(lines):
        dy = 0 if li == 0 else size * lh
        first = True
        for t, st in ln:
            attrs = []
            if st.get("b"):
                attrs.append('font-weight="bold"')
            if st.get("i"):
                attrs.append('font-style="italic"')
            attrs.append(f'fill="{st.get("c", color)}"')
            pos = f' x="{x}" dy="{dy}"' if first else ""
            tsp.append(f'<tspan{pos} {" ".join(attrs)}>{html.escape(t)}</tspan>')
            first = False
    svg.append(f'<text x="{x}" y="{y + asc}" font-family="{FAMILY}" font-size="{size}" '
               f'text-anchor="{anchor}" xml:space="preserve">{"".join(tsp)}</text>')
    # draw.io: explicit line breaks, no re-wrapping, same metrics family
    hl = []
    for ln in lines:
        seg = ""
        for t, st in ln:
            s = html.escape(t)
            if st.get("b"):
                s = f"<b>{s}</b>"
            if st.get("i"):
                s = f"<i>{s}</i>"
            seg += f'<font color="{st.get("c", color)}">{s}</font>'
        hl.append(seg)
    wbox = max(sum(width(t, size, st.get("b", False), st.get("i", False)) for t, st in ln) for ln in lines) + 2
    hbox = size * (1 + lh * (len(lines) - 1)) + 2
    bx = x - (wbox if anchor == "end" else wbox / 2 if anchor == "middle" else 0)
    align = {"start": "left", "middle": "center", "end": "right"}[anchor]
    style = (f"text;html=1;whiteSpace=nowrap;align={align};verticalAlign=top;spacing=0;spacingTop=-1;"
             f"fontFamily=Helvetica;fontSize={size};fontColor={color};")
    cells.append(("v", style, "<br>".join(hl), bx, y, wbox, hbox))
    return size * (1 + lh * (len(lines) - 1))


B = {"b": True}


# ---------------------------------------------------------------- vector glyphs (24x24 viewBox)
def glyph(body, x, y, s, vb=24):
    svg.append(f'<g transform="translate({x},{y}) scale({s / vb})">{body}</g>')
    uri = "data:image/svg+xml," + base64.b64encode(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{vb}" height="{vb}" viewBox="0 0 {vb} {vb}">{body}</svg>'.encode()).decode()
    cells.append(("v", f"shape=image;html=1;imageAspect=0;image={uri}", "", x, y, s, s))


def g_image(col, corrupt=False):
    body = (f'<rect x="2.5" y="4" width="19" height="16" rx="2.5" fill="#FFFFFF" stroke="{col}" stroke-width="1.6"/>'
            f'<circle cx="8" cy="9" r="1.9" fill="{col}"/>'
            f'<path d="M3.5 18 L9.5 12 L13 15.5 L16 12.5 L20.5 17" fill="none" stroke="{col}" stroke-width="1.6" stroke-linejoin="round"/>')
    if corrupt:   # blur / noise streaks and a crack
        body += (f'<path d="M4 7.5 H11 M13 10 H20 M5 14 H10" stroke="{col}" stroke-width="1" opacity="0.55"/>'
                 f'<path d="M15 4 L13.5 8.5 L16 10 L14 14" fill="none" stroke="{BAD}" stroke-width="1.5" stroke-linejoin="round"/>')
    return body


def g_question(col):
    return (f'<circle cx="12" cy="12" r="9.5" fill="#FFFFFF" stroke="{col}" stroke-width="1.6"/>'
            f'<path d="M9.2 9.3 A2.9 2.9 0 1 1 13.4 11.9 C12.5 12.4 12 13 12 14" fill="none" stroke="{col}" '
            f'stroke-width="1.8" stroke-linecap="round"/><circle cx="12" cy="17" r="1.15" fill="{col}"/>')


def g_lens(col):
    return (f'<circle cx="10" cy="10" r="6.5" fill="#FFFFFF" stroke="{col}" stroke-width="2"/>'
            f'<path d="M15 15 L21 21" stroke="{col}" stroke-width="2.6" stroke-linecap="round"/>'
            f'<path d="M7 11.5 L9 9 L11 10.5 L13 7.5" fill="none" stroke="{col}" stroke-width="1.4" stroke-linejoin="round"/>')


def g_shield(col):
    return (f'<path d="M12 2.5 L19.5 5.5 V11 C19.5 15.8 16.3 19.2 12 21 C7.7 19.2 4.5 15.8 4.5 11 V5.5 Z" fill="{col}"/>'
            f'<path d="M8.4 11.6 L11 14.2 L15.8 9" fill="none" stroke="#FFFFFF" stroke-width="2" '
            f'stroke-linecap="round" stroke-linejoin="round"/>')


def g_tri(col, up=True):
    d = "M12 5 L20 18 H4 Z" if up else "M12 19 L20 6 H4 Z"
    return f'<path d="{d}" fill="{col}"/>'


def g_vec(col):   # steering vector: an arrow with a plus
    return (f'<path d="M3 19 L17 5" stroke="{col}" stroke-width="2.2" stroke-linecap="round"/>'
            f'<path d="M10 4.5 H18.5 V13" fill="none" stroke="{col}" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/>')


def layers(x, y, n, hl, col, w=9, h=46, gap=5):
    """A row of decoder layers (thin vertical blocks); returns the highlighted block's centre x."""
    for k in range(n):
        rect(x + k * (w + gap), y, w, h, fill=col if k == hl else "#E3E7EC", rx=2)
    return x + hl * (w + gap) + w / 2


def header(x, w, letter, kicker, title, col):
    circle(x + 20, PY + 19, 9.5, col)
    text(x + 20, PY + 12.2, [(letter, {"b": True, "c": "#FFFFFF"})], size=12.5, anchor="middle")
    text(x + 36, PY + 13, [(kicker.upper(), {"b": True, "c": col})], size=10.5)
    text(x + 12, PY + 34, [(title, B)], size=15)


# ---------------------------------------------------------------- NUMBERS (sources in comments)
# outputs/eval/*: full fine-tuned Qwen3-VL-4B, accuracy averaged over TextVQA / GQA / VQAv2 (unweighted)
CLEAN = {"base": 72.09, "rkl": 72.16, "grpo": 79.01}
CORR = {"base": 64.26, "rkl": 66.39, "grpo": 71.84}
# VizWiz-val unanswerable (n=1,385) accuracy
UNANS = {"base": 76.0, "rkl": 56.5, "grpo": 60.3}
# outputs/probe_fullft/analysis.json: layer-24 probe, test AUROC
PROBE = {"base": 0.931, "rkl": 0.927, "grpo": 0.930}
# outputs/steer_full/fgrpo/eval: GRPO unsteered vs no-image vector (layer 18, always on); same pipeline both
STEER = {"clean": (78.98, 78.80), "corr": (71.96, 71.72), "unans": (60.1, 81.7)}

W, H = 880, 372
PX = [(8, 322), (344, 230), (588, 284)]
PY, PH = 8, H - 16
PURPLE = "#6B4FA0"
for (x, w) in PX:
    rect(x, PY, w, PH, fill=PANEL, stroke=RULE, sw=1, rx=8)
for (x, w), (x2, _) in zip(PX[:-1], PX[1:]):
    cy = PY + PH / 2
    line(x + w + 2, cy, x2 - 2, cy, color=MUTED, sw=1.6, arrow=True)

im = Image.open(PHOTO).convert("RGB")
im.thumbnail((360, 480))
buf = io.BytesIO()
im.save(buf, "JPEG", quality=82)
uri = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
ASPECT = im.size[1] / im.size[0]

# ---- (a) the problem -------------------------------------------------------------
ax, aw = PX[0]
header(ax, aw, "a", "The problem", "Robustness up, abstention down", BAD)
ph_w = 74
ph_h = round(ph_w * ASPECT)
image(ax + 12, 60, ph_w, ph_h, uri)
rect(ax + 12, 60, ph_w, ph_h, stroke=RULE, sw=0.8)
qx = ax + 98
qh = text(qx, 60, [("Q: ", B), ("“What flavor of jellybeans are in this package?”", {"i": True})],
          size=12.5, maxw=ax + aw - 12 - qx)
glyph(g_question(MUTED), qx, 66 + qh, 12)
text(qx + 16, 66 + qh, [("label: ", {"c": MUTED}), ("unanswerable", {"b": True, "c": MUTED})], size=11)
rows = [("Base", BASE, "unanswerable", GOOD, "ok"), ("RKL", RKL, "Cherry, Lime, …", BAD, "bad"),
        ("GRPO", GRPO, "strawberry", BAD, "bad")]
for k, (name, col, ans, acol, ok) in enumerate(rows):
    y = 108 + k * 18
    circle(qx + 4, y + 6, 4, col)
    text(qx + 13, y, [(name, {"b": True, "c": col})], size=11.5)
    text(qx + 54, y, [("→ ", {"c": MUTED}), (ans, {"b": True, "c": acol})], size=11.5)
    icon(ok, ax + aw - 26, y + 0.5, 12)

# change vs base: a diverging bar chart, one group per test condition
line(ax + 12, 174, ax + aw - 12, 174, color=RULE, sw=0.8)
text(ax + 12, 180, [("Accuracy change vs. base (points)", {"c": MUTED})], size=10.5)
z, sc = 248, 2.9
line(ax + 12, z, ax + aw - 12, z, color=MUTED, sw=1)
text(ax + aw - 12, z - 13, [("base", {"c": MUTED})], size=10, anchor="end")
groups = [("Clean", g_image(INK), CLEAN, True), ("Corrupted", g_image(INK, True), CORR, True),
          ("Unanswerable", g_question(INK), UNANS, False)]
gw = (aw - 24) / 3
for k, (lab, gl, d, up) in enumerate(groups):
    gc = ax + 12 + gw * k + gw / 2 + 8
    glyph(gl, gc - 10, 196, 20)
    for j, (m, col) in enumerate((("rkl", RKL), ("grpo", GRPO))):
        dv = d[m] - d["base"]
        x = gc - 25 + j * 28
        hgt = max(abs(dv) * sc, 1.2)
        rect(x, z - hgt if dv > 0 else z, 22, hgt, fill=col)
        ty = z - hgt - 13 if dv > 0 else z + hgt + 2
        text(x + 11, ty, [(f"{dv:+.1f}", {"b": True, "c": col})], size=10, anchor="middle")
    lw = width(lab, 11.5, True)
    text(gc - 6, 324, [(lab, B)], size=11.5, anchor="middle")
    glyph(g_tri(GOOD if up else BAD, up), gc - 6 + lw / 2 + 3, 324.5, 11)

# ---- (b) diagnosis ----------------------------------------------------------------
bx, bw = PX[1]
header(bx, bw, "b", "Diagnosis", "The model still knows", PURPLE)
# input -> layers -> probe -> "unanswerable"
image(bx + 14, 66, 30, round(30 * ASPECT), uri)
rect(bx + 14, 66, 30, round(30 * ASPECT), stroke=RULE, sw=0.8)
line(bx + 47, 86, bx + 60, 86, color=MUTED, sw=1, arrow=True)
hx = layers(bx + 64, 63, 6, 4, PURPLE)
text(hx, 113, [("ℓ = 24", {"b": True, "c": PURPLE})], size=10, anchor="middle")
line(hx + 6, 86, bx + 150, 86, color=MUTED, sw=1, arrow=True)
glyph(g_lens(PURPLE), bx + 152, 72, 28)
text(bx + 166, 104, [("probe", {"c": MUTED})], size=10, anchor="middle")
line(bx + 182, 86, bx + 193, 86, color=MUTED, sw=1, arrow=True)
glyph(g_question(PURPLE), bx + 196, 76, 20)
icon("ok", bx + 208, 72, 10)

# probe AUROC bars
line(bx + 12, 134, bx + bw - 12, 134, color=RULE, sw=0.8)
text(bx + 12, 140, [("Probe AUROC: is it unanswerable?", {"c": MUTED})], size=10.5)
cb, ch, lo = 262, 88, 0.5
line(bx + 30, cb, bx + bw - 30, cb, color=MUTED, sw=1)
text(bx + bw - 30, cb + 3, [("chance", {"c": MUTED})], size=9.5, anchor="end")
for k, (m, col, lab) in enumerate((("base", BASE, "Base"), ("rkl", RKL, "RKL"), ("grpo", GRPO, "GRPO"))):
    v = PROBE[m]
    hgt = ch * (v - lo) / (1 - lo)
    x = bx + 44 + k * 52
    rect(x, cb - hgt, 34, hgt, fill=col)
    text(x + 17, cb - hgt - 14, [(f"{v:.2f}", B)], size=11, anchor="middle")
    text(x + 17, cb + 16, [(lab, {"b": True, "c": col})], size=11, anchor="middle")
icon("ok", bx + 34, 312, 13)
text(bx + 52, 312, [("Knows it can’t answer", {"b": True, "c": PURPLE})], size=12)
icon("bad", bx + 34, 332, 13)
text(bx + 52, 332, [("…but answers anyway", {"b": True, "c": BAD})], size=12)

# ---- (c) the fix ------------------------------------------------------------------
cx, cw = PX[2]
header(cx, cw, "c", "Our fix", "Restore abstention, keep robustness", FIX)
hx = layers(cx + 18, 88, 7, 3, FIX, h=38)
glyph(g_vec(FIX), hx - 11, 64, 22)
text(hx + 14, 66, [("+ λ·v", {"b": True, "c": FIX})], size=12)
text(hx, 130, [("ℓ = 18", {"b": True, "c": FIX})], size=10, anchor="middle")
tags = [("training-free", g_shield(MUTED)), ("one vector, one layer", g_vec(MUTED))]
for k, (t, gl) in enumerate(tags):
    glyph(gl, cx + 134, 88 + k * 22, 14)
    text(cx + 153, 89 + k * 22, [(t, {"c": INK})], size=11)

line(cx + 12, 146, cx + cw - 12, 146, color=RULE, sw=0.8)
text(cx + 12, 151, [("GRPO accuracy (%)", {"c": MUTED})], size=10.5)
fb, fh = 300, 0.98
gw = (cw - 24) / 3
groups = [("Clean", g_image(INK), "clean", CLEAN["base"], "kept"), ("Corrupted", g_image(INK, True), "corr", CORR["base"], "kept"),
          ("Unanswerable", g_question(INK), "unans", UNANS["base"], "up")]
line(cx + 12, fb, cx + cw - 12, fb, color=MUTED, sw=1)
for k, (lab, gl, key, base, st) in enumerate(groups):
    gc = cx + 12 + gw * k + gw / 2
    glyph(gl, gc - 9, 170, 18)
    for j, (v, col) in enumerate(zip(STEER[key], (GRPO, FIX))):
        x = gc - 27 + j * 29
        hgt = v * fh
        rect(x, fb - hgt, 25, hgt, fill=col)
        text(x + 12.5, fb - hgt - 13, [(f"{v:.1f}", {"b": True, "c": col})], size=10, anchor="middle")
    line(gc - 33, fb - base * fh, gc + 35, fb - base * fh, color=INK, sw=1.1, dash=True)
    lw = width(lab, 10.5, True)
    text(gc - 6, 305, [(lab, B)], size=10.5, anchor="middle")
    glyph(g_shield(FIX) if st == "kept" else g_tri(GOOD), gc - 6 + lw / 2 + 2, 304.5, 11)
ly = 324
rect(cx + 14, ly + 2, 10, 10, fill=GRPO)
text(cx + 28, ly, [("GRPO", {"c": MUTED})], size=10.5)
rect(cx + 70, ly + 2, 10, 10, fill=FIX)
text(cx + 84, ly, [("+ steering", {"b": True, "c": FIX})], size=10.5)
line(cx + 150, ly + 7, cx + 168, ly + 7, color=INK, sw=1.1, dash=True)
text(cx + 172, ly, [("base", {"c": MUTED})], size=10.5)
glyph(g_shield(FIX), cx + 206, ly + 0.5, 12)
text(cx + 221, ly, [("gain kept", {"c": MUTED})], size=10.5)
text(cx + 14, ly + 18, [("Jellybeans  ", {"c": MUTED}), ("→ ", {"c": MUTED}), ("unanswerable", {"b": True, "c": GOOD})], size=11.5)
icon("ok", cx + 14 + width("Jellybeans  → unanswerable", 11.5, True) + 6, ly + 18.5, 12)

# ---------------------------------------------------------------- write SVG / PNG / draw.io
defs = ('<defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" '
        'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="context-stroke"/></marker></defs>')
doc = (f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="{W}" height="{H}" '
       f'viewBox="0 0 {W} {H}"><rect width="{W}" height="{H}" fill="#FFFFFF"/>{defs}' + "".join(svg) + "</svg>")
# cairosvg has no context-stroke: give each arrowed line its own colour via a per-colour marker
cols = sorted({c for c in (MUTED, FIX)})
doc = doc.replace(defs, "<defs>" + "".join(
    f'<marker id="ah{k}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">'
    f'<path d="M0,0 L10,5 L0,10 z" fill="{c}"/></marker>' for k, c in enumerate(cols)) + "</defs>")
for k, c in enumerate(cols):
    doc = doc.replace(f'stroke="{c}" stroke-width="1.6" marker-end="url(#ah)"', f'stroke="{c}" stroke-width="1.6" marker-end="url(#ah{k})"')
    doc = doc.replace(f'stroke="{c}" stroke-width="1" marker-end="url(#ah)"', f'stroke="{c}" stroke-width="1" marker-end="url(#ah{k})"')
    doc = doc.replace(f'stroke="{c}" stroke-width="1.4" marker-end="url(#ah)"', f'stroke="{c}" stroke-width="1.4" marker-end="url(#ah{k})"')
(OUT / "teaser_v2.svg").write_text(doc)

xml = ['<mxfile host="app.diagrams.net"><diagram name="teaser" id="teaser">'
       f'<mxGraphModel dx="{W}" dy="{H}" grid="0" gridSize="5" guides="1" tooltips="1" connect="0" arrows="1" fold="1" '
       f'page="1" pageScale="1" pageWidth="{W}" pageHeight="{H}" math="0" shadow="0"><root>'
       '<mxCell id="0"/><mxCell id="1" parent="0"/>']
for c in cells:
    if c[0] == "v":
        _, style, val, x, y, w, h = c
        xml.append(f'<mxCell id="{nid()}" value="{html.escape(val, quote=True)}" style="{html.escape(style, quote=True)}" '
                   f'vertex="1" parent="1"><mxGeometry x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" height="{h:.2f}" as="geometry"/></mxCell>')
    else:
        _, style, (x1, y1, x2, y2) = c
        xml.append(f'<mxCell id="{nid()}" style="{html.escape(style, quote=True)}" edge="1" parent="1">'
                   f'<mxGeometry relative="1" as="geometry"><mxPoint x="{x1:.2f}" y="{y1:.2f}" as="sourcePoint"/>'
                   f'<mxPoint x="{x2:.2f}" y="{y2:.2f}" as="targetPoint"/></mxGeometry></mxCell>')
xml.append("</root></mxGraphModel></diagram></mxfile>")
(OUT / "teaser_v2.drawio").write_text("".join(xml))
print("wrote", OUT / "teaser_v2.svg", OUT / "teaser_v2.drawio", len(cells), "cells")
import cairosvg
cairosvg.svg2png(bytestring=doc.encode(), write_to=str(OUT / "teaser_v2.png"), scale=3)
cairosvg.svg2pdf(bytestring=doc.encode(), write_to=str(OUT / "teaser_v2.pdf"))
