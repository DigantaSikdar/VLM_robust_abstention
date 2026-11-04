"""Teaser figure: problem -> diagnosis -> fix. One layout, three outputs (SVG, PNG, draw.io).

    python make_teaser.py <photo.jpg> <out_dir>

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

# ---------------------------------------------------------------- NUMBERS (sources in comments)
# outputs/eval/*/summary.json, VizWiz-val unanswerable (n=1,385)
UNANS = {"base": 76.0, "rkl": 56.5, "grpo": 60.3}
# pooled corrupted accuracy gain over base, TextVQA-C + GQA-C + VQAv2-C
CORR_GAIN = {"rkl": 2.0, "grpo": 8.5}
# outputs/probe_fullft/analysis.json: best-layer test AUROC; behaviour link
PROBE = {"base": 0.931, "rkl": 0.927, "grpo": 0.930}
FLAGGED = 77  # % of answered-although-unanswerable items the probe flags (RKL 76, GRPO 77)
# outputs/commitment: refuse rate given the base model's "can't tell" reasoning
COMMIT = {"rkl": 98.4, "grpo": 8.2}
# outputs/steer_full/{frkl,fgrpo}/eval, image view, paired with the unsteered run
STEER = {"rkl": (56.1, 79.9), "grpo": (60.1, 81.7)}   # RKL: no-image vector, gated; GRPO: no-image, always
GRPO_ANS = (63.8, 68.7)

# ---------------------------------------------------------------- panels
PX = [(8, 300), (326, 272), (616, 256)]       # x, width
PY, PH = 8, H - 16
for (x, w) in PX:
    rect(x, PY, w, PH, fill=PANEL, stroke=RULE, sw=1, rx=8)
for (x, w), (x2, _) in zip(PX[:-1], PX[1:]):
    cy = PY + PH / 2
    line(x + w + 3, cy, x2 - 3, cy, color=MUTED, sw=1.6, arrow=True)


def header(x, w, letter, kicker, title, col):
    circle(x + 20, PY + 19, 9.5, col)
    text(x + 20, PY + 12.2, [(letter, {"b": True, "c": "#FFFFFF"})], size=12.5, anchor="middle")
    text(x + 36, PY + 13, [(kicker.upper(), {"b": True, "c": col})], size=10.5)
    return text(x + 12, PY + 34, [(title, B)], size=14, maxw=w - 24)


# ---- (a) the problem -------------------------------------------------------------
ax, aw = PX[0]
header(ax, aw, "a", "The problem", "Robustness fine-tuning teaches VLMs to answer the unanswerable", BAD)

im = Image.open(PHOTO).convert("RGB")
im.thumbnail((360, 480))
buf = io.BytesIO()
im.save(buf, "JPEG", quality=82)
uri = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
ph_w = 96
ph_h = round(ph_w * im.size[1] / im.size[0])
image(ax + 12, 76, ph_w, ph_h, uri)
rect(ax + 12, 76, ph_w, ph_h, stroke=RULE, sw=0.8)
text(ax + 12 + ph_w / 2, 76 + ph_h + 4, [("VizWiz photo", {"c": MUTED})], size=10.5, anchor="middle")

qx = ax + 12 + ph_w + 10
qh = text(qx, 78, [("Q: ", B), ("“What flavor of jellybeans are in this package?”", {"i": True})], size=13, maxw=ax + aw - 12 - qx)
text(qx, 82 + qh, [("Label: ", {"c": MUTED}), ("unanswerable", {"b": True, "c": MUTED})], size=11)
# headline numbers
sy = 82 + qh + 24
text(qx, sy, [("Corrupted-image accuracy", {"c": MUTED})], size=11)
text(qx, sy + 14, [("RKL ", {"b": True, "c": RKL}), (f"+{CORR_GAIN['rkl']:.1f}   ", {"b": True, "c": GOOD}),
                   ("GRPO ", {"b": True, "c": GRPO}), (f"+{CORR_GAIN['grpo']:.1f}", {"b": True, "c": GOOD})], size=13)
text(qx, sy + 36, [("but unanswerable accuracy", {"c": MUTED})], size=11)
text(qx, sy + 50, [(f"{UNANS['base']:.1f} → ", B), (f"{UNANS['rkl']:.1f}", {"b": True, "c": RKL}),
                   (" / ", B), (f"{UNANS['grpo']:.1f}", {"b": True, "c": GRPO})], size=13)

cards = [
    ("Base (Qwen3-VL-4B)", BASE, "ok",
     [("“The image is too blurry and the text is not legible …” ", {"i": True}), ("→ ", B),
      ("unanswerable", {"b": True, "c": GOOD})], None),
    ("+ RKL distillation", RKL, "bad",
     [("“The flavors visible are ", {"i": True}), ("Cherry, Lime, Orange, and Strawberry", {"i": True, "c": BAD}),
      (" … clearly visible.” ", {"i": True}), ("→ ", B), ("Cherry, Lime, Orange, Strawberry", {"b": True, "c": BAD})],
     "reasoning hallucinates"),
    ("+ GRPO", GRPO, "bad",
     [("“… the labels are illegible. ", {"i": True}), ("Therefore, the answer is unanswerable.", {"i": True, "c": GOOD}),
      ("” ", {"i": True}), ("→ ", B), ("strawberry", {"b": True, "c": BAD})],
     "answer ignores reasoning"),
]
cy = 76 + ph_h + 22
for name, col, ok, body, tag in cards:
    lines_h = len(wrap(body, 12, aw - 40)) * 12 * 1.25
    ch = 22 + lines_h
    rect(ax + 12, cy, aw - 24, ch, fill="#FFFFFF", stroke=RULE, sw=0.8, rx=5)
    rect(ax + 12, cy, 3.5, ch, fill=col)
    text(ax + 22, cy + 5, [(name, {"b": True, "c": col})], size=12)
    if tag:
        text(ax + aw - 31, cy + 6, [(tag, {"i": True, "c": BAD})], size=10.5, anchor="end")
    icon(ok, ax + aw - 27, cy + 5.5, 11)
    text(ax + 22, cy + 20, body, size=12, maxw=aw - 40)
    cy += ch + 8
text(ax + 12, cy + 2, [("Verbatim response excerpts, greedy decoding.", {"i": True, "c": MUTED})],
     size=10.5, maxw=aw - 24)

# ---- (b) diagnosis --------------------------------------------------------------
bx, bw = PX[1]
header(bx, bw, "b", "Diagnosis", "The model still knows; training breaks a different step", "#6B4FA0")

# b1: probe
y0 = 80
text(bx + 12, y0, [("1  Linear probe on hidden states", B)], size=12)
text(bx + 12, y0 + 15, [("Does the model encode “unanswerable”? (AUROC)", {"c": MUTED})], size=10.5)
chart_x, chart_y, chart_h = bx + 20, y0 + 36, 62
lo, hi = 0.5, 1.0
line(chart_x, chart_y + chart_h, chart_x + 124, chart_y + chart_h, color=RULE, sw=1)
for k, (m, col, lab) in enumerate((("base", BASE, "Base"), ("rkl", RKL, "RKL"), ("grpo", GRPO, "GRPO"))):
    v = PROBE[m]
    hgt = chart_h * (v - lo) / (hi - lo)
    x = chart_x + 8 + k * 40
    rect(x, chart_y + chart_h - hgt, 26, hgt, fill=col)
    text(x + 13, chart_y + chart_h - hgt - 14, [(f"{v:.2f}", B)], size=10.5, anchor="middle")
    text(x + 13, chart_y + chart_h + 3, [(lab, {"c": col, "b": True})], size=10.5, anchor="middle")
line(chart_x, chart_y + chart_h * (1 - (0.5 - lo) / (hi - lo)), chart_x + 124, chart_y + chart_h * (1 - (0.5 - lo) / (hi - lo)),
     color=MUTED, sw=0.8, dash=True)
text(chart_x + 134, chart_y + 4, [("Unchanged after", {}), ("", {})], size=11.5, maxw=110)
text(chart_x + 134, chart_y + 18, [("fine-tuning; the probe", {})], size=11.5)
text(chart_x + 134, chart_y + 32, [("flags ", {}), (f"{FLAGGED}%", B), (" of the", {})], size=11.5)
text(chart_x + 134, chart_y + 46, [("hallucinated answers.", {})], size=11.5)
text(chart_x + 134, chart_y + 66, [("Knows, but answers", {"b": True, "c": "#6B4FA0"})], size=11.5)
text(chart_x + 134, chart_y + 80, [("anyway.", {"b": True, "c": "#6B4FA0"})], size=11.5)

# b2: commitment test
y1 = 228
line(bx + 12, y1 - 10, bx + bw - 12, y1 - 10, color=RULE, sw=0.8)
text(bx + 12, y1, [("2  Answer-commitment test", B)], size=12)
text(bx + 12, y1 + 15, [("Give each model the base model’s “can’t tell”", {"c": MUTED})], size=10.5)
text(bx + 12, y1 + 28, [("reasoning: does its answer follow?", {"c": MUTED})], size=10.5)


def pipeline(y, model, col, broken):
    xs = [bx + 58, bx + 118, bx + 204]
    labels = ["image", "reasoning", "answer"]
    ws = [46, 72, 56]
    for k, (x, lab, wd) in enumerate(zip(xs, labels, ws)):
        bad = (k == broken)
        rect(x, y, wd, 22, fill="#FDECEC" if bad else "#FFFFFF", stroke=BAD if bad else RULE, sw=1.1 if bad else 0.8, rx=4)
        text(x + wd / 2, y + 5, [(lab, {"c": BAD if bad else INK, "b": bad})], size=11, anchor="middle")
        if k < 2:
            line(x + wd + 2, y + 11, xs[k + 1] - 3, y + 11, color=MUTED, sw=1, arrow=True)
    text(bx + 12, y + 5, [(model, {"b": True, "c": col})], size=11.5)
    icon("bad", xs[broken] + ws[broken] - 7, y - 6, 11)


pipeline(y1 + 50, "RKL", RKL, 1)
pipeline(y1 + 84, "GRPO", GRPO, 2)
# bars: refuse rate given base reasoning
yb = y1 + 122
text(bx + 12, yb, [("Refuses when given “can’t tell” reasoning", {"c": MUTED})], size=10.5)
for k, (m, col, lab) in enumerate((("rkl", RKL, "RKL"), ("grpo", GRPO, "GRPO"))):
    yy = yb + 17 + k * 20
    text(bx + 12, yy + 1, [(lab, {"b": True, "c": col})], size=11)
    full = bw - 110
    rect(bx + 52, yy, full, 13, fill="#EEF0F3")
    rect(bx + 52, yy, full * COMMIT[m] / 100, 13, fill=col)
    text(bx + 56 + full, yy + 1, [(f"{COMMIT[m]:.1f}%", B)], size=11)
text(bx + 12, yb + 62, [("RKL: ", {"b": True, "c": RKL}), ("the reasoning goes wrong.  ", {}),
                        ("GRPO: ", {"b": True, "c": GRPO}), ("the answer step ignores the reasoning.", {})],
     size=11, maxw=bw - 24)

# ---- (c) the fix ----------------------------------------------------------------
cx, cw = PX[2]
header(cx, cw, "c", "Our fix", "Training-free contrastive steering restores abstention", FIX)

# schematic: decoder stack with injection at layer 18
y0 = 82
sx, sw_ = cx + 14, 44
n = 7
for k in range(n):
    yy = y0 + k * 13
    hl = (k == 3)
    rect(sx, yy, sw_, 10, fill=FIX if hl else "#E6E9ED", stroke="none", rx=2)
text(sx + sw_ / 2, y0 + n * 13 + 3, [("decoder layers", {"c": MUTED})], size=10.5, anchor="middle")
text(sx + sw_ + 4, y0 + 3 * 13 - 9, [("ℓ = 18", {"b": True, "c": FIX})], size=10.5)
# vector box
vx, vy = cx + 104, y0 - 2
bw_ = cw - 116
desc = [("averaged over refusal-vs-answer pairs", {"c": MUTED})]
dh = len(wrap(desc, 10.5, bw_ - 14)) * 10.5 * 1.25
rect(vx, vy, bw_, 24 + dh, fill="#FFFFFF", stroke=FIX, sw=1.1, rx=5)
text(vx + 7, vy + 5, [("v = h(", {}), ("refusal", {"b": True, "c": GOOD}), (") − h(", {}), ("answer", {"b": True, "c": BAD}),
                      (")", {})], size=11)
text(vx + 7, vy + 20, desc, size=10.5, maxw=bw_ - 14)
line(vx - 2, y0 + 3 * 13 + 5, sx + sw_ + 36, y0 + 3 * 13 + 5, color=FIX, sw=1.4, arrow=True)
text(vx, vy + 32 + dh, [("h", {"i": True}), ("ℓ", {}), (" ← h", {"i": True}), ("ℓ", {}), (" + λ·v", {})], size=11.5)
text(vx, vy + 47 + dh, [("optional gate: entropy > τ", {"c": MUTED})], size=10.5)

# results: grouped bars, unanswerable accuracy
ry = 202
line(cx + 12, ry - 8, cx + cw - 12, ry - 8, color=RULE, sw=0.8)
text(cx + 12, ry, [("Unanswerable VizWiz accuracy (%)", B)], size=12)
gx, gy, gh = cx + 30, ry + 22, 86
line(gx, gy + gh, gx + 200, gy + gh, color=RULE, sw=1)
scale = gh / 100
for k, (m, col, lab) in enumerate((("rkl", RKL, "RKL"), ("grpo", GRPO, "GRPO"))):
    before, after = STEER[m]
    x = gx + 14 + k * 100
    for j, (v, fill) in enumerate(((before, col), (after, FIX))):
        hgt = v * scale
        xx = x + j * 32
        rect(xx, gy + gh - hgt, 26, hgt, fill=fill, stroke="none")
        text(xx + 13, gy + gh - hgt - 14, [(f"{v:.1f}", B)], size=10.5, anchor="middle")
    text(x + 29, gy + gh + 3, [(lab, {"b": True, "c": col})], size=11, anchor="middle")
by = gy + gh - UNANS["base"] * scale
line(gx, by, gx + 196, by, color=BASE, sw=1, dash=True)
text(gx + 199, by - 6, [("base", {"c": BASE})], size=10)
# legend
lx, ly = cx + 12, gy + gh + 22
rect(lx, ly + 2, 9, 9, fill=RKL)
rect(lx + 11, ly + 2, 9, 9, fill=GRPO)
text(lx + 24, ly, [("fine-tuned", {"c": MUTED})], size=10.5)
rect(lx + 92, ly + 2, 9, 9, fill=FIX)
text(lx + 105, ly, [("+ steering", {"c": FIX, "b": True})], size=10.5)
text(cx + 12, ly + 17, [("GRPO answerable ", {}), (f"{GRPO_ANS[0]:.1f} → {GRPO_ANS[1]:.1f}", B),
                        ("; corrupted accuracy within ±0.5", {})], size=10.5, maxw=cw - 24)
# steered response
sy = ly + 50
rect(cx + 12, sy, cw - 24, H - 16 - sy, fill="#FFFFFF", stroke=RULE, sw=0.8, rx=5)
rect(cx + 12, sy, 3.5, H - 16 - sy, fill=FIX)
text(cx + 22, sy + 5, [("GRPO + steering", {"b": True, "c": FIX})], size=12)
icon("ok", cx + cw - 27, sy + 5.5, 11)
text(cx + 22, sy + 20, [("“The image is too blurry to determine the flavor …” ", {"i": True}), ("→ ", B),
                        ("unanswerable", {"b": True, "c": GOOD})], size=11.5, maxw=cw - 40)

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
(OUT / "teaser.svg").write_text(doc)

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
(OUT / "teaser.drawio").write_text("".join(xml))
print("wrote", OUT / "teaser.svg", OUT / "teaser.drawio", len(cells), "cells")
