"""Teaser figure v3: method-agnostic, illustrated (vector icons, robot VLM, speech bubbles, stickers).

    python make_teaser_v3.py <photo.jpg> <out_dir>      -> teaser_v3.{svg,png,pdf}

Problem -> diagnosis -> fix, with no method names and no probe/steering numbers.
The only numbers are the panel-(a) stickers: full fine-tuned GRPO vs. the base Qwen3-VL-4B,
accuracy averaged over TextVQA / GQA / VQAv2 (clean, corrupted) and VizWiz-val unanswerable.
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
INK, MUTED, LINE = "#1F2430", "#6A7381", "#D9DEE5"
RED, RED_BG, RED_TINT = "#D9483B", "#FDE6E2", "#FFF6F4"
PUR, PUR_BG, PUR_TINT = "#7155C9", "#ECE6FB", "#F8F6FE"
GRN, GRN_BG, GRN_TINT = "#169A72", "#DDF3EA", "#F3FBF7"
BLUE, GOLD = "#3C7DD9", "#F2B33D"
FONT_DIR = "/usr/share/fonts/truetype/liberation/"
FAMILY = "Helvetica, Arial, Liberation Sans, sans-serif"
W, H = 880, 330
svg = []
_fonts = {}


def font(size, b=False, i=False):
    name = "LiberationSans-" + {(0, 0): "Regular", (1, 0): "Bold", (0, 1): "Italic", (1, 1): "BoldItalic"}[(b, i)]
    if (name, size) not in _fonts:
        _fonts[(name, size)] = ImageFont.truetype(FONT_DIR + name + ".ttf", size=round(size * 4))
    return _fonts[(name, size)]


def width(t, size, b=False, i=False):
    return font(size, b, i).getlength(t) / 4


def text(x, y, t, size=12, c=INK, b=False, i=False, anchor="start", maxw=None, lh=1.22):
    """Text with its top at y; wraps to maxw. Returns the height."""
    words, lines = t.split(" "), []
    if maxw:
        cur = ""
        for w_ in words:
            nxt = (cur + " " + w_).strip()
            if cur and width(nxt, size, b, i) > maxw:
                lines.append(cur)
                cur = w_
            else:
                cur = nxt
        lines.append(cur)
    else:
        lines = [t]
    attrs = f'font-family="{FAMILY}" font-size="{size}" fill="{c}" text-anchor="{anchor}"'
    attrs += ' font-weight="bold"' if b else ""
    attrs += ' font-style="italic"' if i else ""
    for k, ln in enumerate(lines):
        svg.append(f'<text x="{x}" y="{y + size * 0.8 + k * size * lh}" {attrs}>{html.escape(ln)}</text>')
    return size * (1 + lh * (len(lines) - 1))


def rect(x, y, w, h, fill="none", stroke="none", sw=1, rx=0, extra=""):
    svg.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" '
               f'stroke-width="{sw}" {extra}/>')


def group(body, x, y, s=1.0, rot=0, vb=24):
    """Place a glyph drawn in a vb x vb box at (x, y), size s px, rotated about its centre."""
    k = s / vb
    svg.append(f'<g transform="translate({x},{y}) rotate({rot},{s / 2},{s / 2}) scale({k})">{body}</g>')


# ---------------------------------------------------------------- glyphs (24 x 24)
def g_image(col, corrupt=False):
    b = (f'<rect x="2.5" y="4" width="19" height="16" rx="3" fill="#FFFFFF" stroke="{col}" stroke-width="1.7"/>'
         f'<circle cx="8" cy="9.2" r="2" fill="{GOLD}"/>'
         f'<path d="M3.6 18.5 L9.5 12.5 L13 16 L16 13 L20.4 17.5 V18.8 H3.6 Z" fill="{col}" opacity="0.85"/>')
    if corrupt:
        b += (f'<path d="M4.5 7 H10 M12.5 10.5 H19.5 M5 14 H9.5" stroke="#FFFFFF" stroke-width="1.3" opacity="0.9"/>'
              f'<path d="M15.5 4 L13.8 8.8 L16.4 10.4 L14.4 15" fill="none" stroke="{RED}" stroke-width="1.6" '
              f'stroke-linejoin="round" stroke-linecap="round"/>')
    return b


def g_q(col, fill="#FFFFFF"):
    return (f'<circle cx="12" cy="12" r="9.6" fill="{fill}" stroke="{col}" stroke-width="1.8"/>'
            f'<path d="M9.2 9.3 A2.9 2.9 0 1 1 13.4 11.9 C12.5 12.4 12 13 12 14.2" fill="none" stroke="{col}" '
            f'stroke-width="2" stroke-linecap="round"/><circle cx="12" cy="17.2" r="1.25" fill="{col}"/>')


def g_check(bg):
    return (f'<circle cx="12" cy="12" r="11" fill="{bg}" stroke="#FFFFFF" stroke-width="2"/>'
            f'<path d="M7 12.4 L10.4 15.8 L17 8.8" fill="none" stroke="#FFFFFF" stroke-width="2.8" '
            f'stroke-linecap="round" stroke-linejoin="round"/>')


def g_cross(bg):
    return (f'<circle cx="12" cy="12" r="11" fill="{bg}" stroke="#FFFFFF" stroke-width="2"/>'
            f'<path d="M8 8 L16 16 M16 8 L8 16" stroke="#FFFFFF" stroke-width="2.8" stroke-linecap="round"/>')


def g_shield(col):
    return (f'<path d="M12 2 L20 5.2 V11 C20 16 16.6 19.6 12 21.6 C7.4 19.6 4 16 4 11 V5.2 Z" fill="{col}"/>'
            f'<path d="M8.3 11.6 L11 14.3 L15.9 9.1" fill="none" stroke="#FFFFFF" stroke-width="2.2" '
            f'stroke-linecap="round" stroke-linejoin="round"/>')


def g_arrow(col, up=True):
    d = "M12 4 L19.5 12.5 H14.8 V20 H9.2 V12.5 H4.5 Z" if up else "M12 20 L19.5 11.5 H14.8 V4 H9.2 V11.5 H4.5 Z"
    return f'<path d="{d}" fill="{col}" stroke-linejoin="round"/>'


def g_lens(col):
    return (f'<circle cx="10" cy="10" r="6.8" fill="#FFFFFF" fill-opacity="0.55" stroke="{col}" stroke-width="2.2"/>'
            f'<path d="M15.2 15.2 L21 21" stroke="{col}" stroke-width="3" stroke-linecap="round"/>'
            f'<path d="M6.6 8.4 A4 4 0 0 1 9.4 6.2" fill="none" stroke="#FFFFFF" stroke-width="1.5" stroke-linecap="round"/>')


def g_spark(col):
    return f'<path d="M12 2 L14 10 L22 12 L14 14 L12 22 L10 14 L2 12 L10 10 Z" fill="{col}"/>'


def g_dumbbell(col):
    return (f'<rect x="2" y="8" width="3.4" height="8" rx="1.2" fill="{col}"/><rect x="5.6" y="5.5" width="3.4" height="13" rx="1.2" fill="{col}"/>'
            f'<rect x="9" y="10.8" width="6" height="2.4" fill="{col}"/>'
            f'<rect x="15" y="5.5" width="3.4" height="13" rx="1.2" fill="{col}"/><rect x="18.6" y="8" width="3.4" height="8" rx="1.2" fill="{col}"/>')


def robot(accent, face="eyes", vb=48):
    """A friendly VLM robot (48 x 48). face: eyes | happy | think."""
    b = (f'<line x1="24" y1="9" x2="24" y2="3.5" stroke="{INK}" stroke-width="2"/>'
         f'<circle cx="24" cy="3.5" r="2.8" fill="{accent}"/>'
         f'<rect x="2.5" y="19" width="4.5" height="11" rx="2.2" fill="{accent}"/>'
         f'<rect x="41" y="19" width="4.5" height="11" rx="2.2" fill="{accent}"/>'
         f'<rect x="6.5" y="9" width="35" height="29" rx="9" fill="#FFFFFF" stroke="{INK}" stroke-width="2"/>'
         f'<rect x="11.5" y="14" width="25" height="18" rx="6" fill="#2A3140"/>'
         f'<rect x="15" y="39" width="18" height="7" rx="3" fill="{accent}"/>')
    if face == "eyes":
        b += (f'<circle cx="18.5" cy="22" r="2.6" fill="#7FE3FF"/><circle cx="29.5" cy="22" r="2.6" fill="#7FE3FF"/>'
              f'<path d="M20 27.5 H28" stroke="#7FE3FF" stroke-width="1.8" stroke-linecap="round"/>')
    elif face == "happy":
        b += (f'<path d="M15.8 22.8 Q18.5 19.6 21.2 22.8 M26.8 22.8 Q29.5 19.6 32.2 22.8" fill="none" stroke="#8BF0C6" '
              f'stroke-width="2" stroke-linecap="round"/><path d="M19.5 26.3 Q24 30 28.5 26.3" fill="none" '
              f'stroke="#8BF0C6" stroke-width="2" stroke-linecap="round"/>')
    else:   # "think": a glowing question mark on the screen
        b += (f'<circle cx="24" cy="23" r="7.4" fill="{accent}" opacity="0.35"/>'
              f'<path d="M21.2 20.3 A2.9 2.9 0 1 1 25.4 22.9 C24.5 23.4 24 24 24 25.2" fill="none" stroke="#FFFFFF" '
              f'stroke-width="2" stroke-linecap="round"/><circle cx="24" cy="28.2" r="1.2" fill="#FFFFFF"/>')
    return b


def bubble(x, y, w, h, tail, fill="#FFFFFF", stroke=LINE, sw=1.4, r=10):
    """Speech bubble with a tail pointing to (tx, ty) from its left or bottom edge."""
    tx, ty = tail
    if tx < x:          # tail on the left edge
        my = min(max(ty, y + r + 6), y + h - r - 6)
        d = (f"M{x + r},{y} H{x + w - r} Q{x + w},{y} {x + w},{y + r} V{y + h - r} Q{x + w},{y + h} {x + w - r},{y + h} "
             f"H{x + r} Q{x},{y + h} {x},{y + h - r} V{my + 5} L{tx},{ty} L{x},{my - 5} V{y + r} Q{x},{y} {x + r},{y} Z")
    else:               # tail on the bottom edge
        mx = min(max(tx, x + r + 6), x + w - r - 6)
        d = (f"M{x + r},{y} H{x + w - r} Q{x + w},{y} {x + w},{y + r} V{y + h - r} Q{x + w},{y + h} {x + w - r},{y + h} "
             f"H{mx + 5} L{tx},{ty} L{mx - 5},{y + h} H{x + r} Q{x},{y + h} {x},{y + h - r} V{y + r} Q{x},{y} {x + r},{y} Z")
    svg.append(f'<path d="{d}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}" stroke-linejoin="round"/>')


def thought(x, y, w, h, trail, fill="#FFFFFF", stroke=LINE):
    """Cloud-ish thought bubble (rounded pill) with a trail of dots towards `trail`."""
    tx, ty = trail
    for k, r in enumerate((2.2, 3.4)):
        f = (k + 1) / 3
        svg.append(f'<circle cx="{tx + (x + 14 - tx) * f}" cy="{ty + (y + h - tx * 0 - ty) * f}" r="{r}" '
                   f'fill="{fill}" stroke="{stroke}" stroke-width="1.3"/>')
    rect(x, y, w, h, fill=fill, stroke=stroke, sw=1.4, rx=h / 2)


def sticker(cx, cy, w, h, rot, bg, body):
    """A die-cut sticker: white border, soft drop shadow, slight tilt. body(x0, y0) draws its content."""
    x0, y0 = cx - w / 2, cy - h / 2
    svg.append(f'<g transform="rotate({rot},{cx},{cy})">')
    rect(x0 + 1.5, y0 + 2.5, w, h, fill="#1F2430", rx=12, extra='opacity="0.10"')
    rect(x0, y0, w, h, fill=bg, stroke="#FFFFFF", sw=3, rx=12)
    body(x0, y0)
    svg.append('</g>')


def pill(x, y, num, label, col, bg):
    tw = width(label, 10.5, True)
    rect(x, y, tw + 34, 20, fill=bg, rx=10)
    svg.append(f'<circle cx="{x + 10}" cy="{y + 10}" r="7.5" fill="{col}"/>')
    text(x + 10, y + 4.8, num, size=10, c="#FFFFFF", b=True, anchor="middle")
    text(x + 23, y + 5, label, size=10.5, c=col, b=True)


def photo(x, y, w, uri, h):
    cid = f"clip{len(svg)}"
    svg.append(f'<clipPath id="{cid}"><rect x="{x}" y="{y}" width="{w}" height="{h}" rx="9"/></clipPath>')
    rect(x + 1.5, y + 2.5, w, h, fill="#1F2430", rx=9, extra='opacity="0.10"')
    svg.append(f'<image x="{x}" y="{y}" width="{w}" height="{h}" href="{uri}" clip-path="url(#{cid})" '
               f'preserveAspectRatio="xMidYMid slice"/>')
    rect(x, y, w, h, stroke="#FFFFFF", sw=3, rx=9)


# ---------------------------------------------------------------- NUMBERS
# full fine-tuned GRPO vs base, outputs/eval/*: mean of TextVQA/GQA/VQAv2; VizWiz-val unanswerable (n=1,385)
D_CLEAN, D_CORR, D_UNANS = 79.01 - 72.09, 71.84 - 64.26, 60.3 - 76.0

im = Image.open(PHOTO).convert("RGB")
im.thumbnail((360, 480))
buf = io.BytesIO()
im.save(buf, "JPEG", quality=85)
URI = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
ASP = im.size[1] / im.size[0]

# ---------------------------------------------------------------- panels
P = [(10, 284, RED_TINT), (310, 262, PUR_TINT), (588, 282, GRN_TINT)]
PY, PH = 10, H - 20
for x, w, tint in P:
    rect(x, PY, w, PH, fill=tint, stroke=LINE, sw=1, rx=16)
for (x, w, _), (x2, _, _) in zip(P[:-1], P[1:]):
    cx, cy = (x + w + x2) / 2, PY + PH / 2
    svg.append(f'<circle cx="{cx}" cy="{cy}" r="11" fill="#FFFFFF" stroke="{LINE}" stroke-width="1.2"/>')
    svg.append(f'<path d="M{cx - 3} {cy - 5} L{cx + 3} {cy} L{cx - 3} {cy + 5}" fill="none" stroke="{MUTED}" '
               f'stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/>')

# ---- (a) the problem -------------------------------------------------------------
x0, w0, _ = P[0]
pill(x0 + 14, 22, "1", "THE PROBLEM", RED, RED_BG)
text(x0 + 14, 50, "Robust training makes VLMs guess", size=15.5, b=True)
ph = round(70 * ASP)
photo(x0 + 16, 80, 70, URI, ph)
bubble(x0 + 100, 78, 172, 48, (x0 + 90, 104))
text(x0 + 110, 85, "“What flavor of jellybeans are in this package?”", size=11.5, i=True, maxw=154)
group(robot(RED), x0 + 100, 134, 46, vb=48)
group(g_dumbbell(INK), x0 + 132, 172, 17, rot=-20)
bubble(x0 + 156, 140, 104, 30, (x0 + 146, 160))
text(x0 + 208, 148, "Strawberry!", size=13, b=True, c=RED, anchor="middle")
group(g_cross(RED), x0 + 250, 130, 20)
text(x0 + 208, 175, "should abstain", size=10, c=MUTED, i=True, anchor="middle")


def stat(glyph_body, label, val, col, arrow_up):
    def body(sx, sy):
        group(glyph_body, sx + 8, sy + 8, 22)
        group(g_arrow(col, arrow_up), sx + 56, sy + 9, 18)
        text(sx + 40, sy + 34, label, size=10 if width(label, 10, True) < 68 else 8.4, c=MUTED, b=True, anchor="middle")
        text(sx + 40, sy + 48, f"{val:+.1f}".replace("-", "−"), size=16, c=col, b=True, anchor="middle")
    return body


sticker(x0 + 54, 244, 80, 72, -4, GRN_BG, stat(g_image(INK), "CLEAN", D_CLEAN, GRN, True))
sticker(x0 + 142, 240, 80, 72, 3, GRN_BG, stat(g_image(INK, True), "CORRUPTED", D_CORR, GRN, True))
sticker(x0 + 230, 246, 80, 72, -3, RED_BG, stat(g_q(INK), "UNANSWERABLE", D_UNANS, RED, False))
text(x0 + w0 / 2, 294, "accuracy vs. base model (points)", size=9.5, c=MUTED, anchor="middle")

# ---- (b) diagnosis ----------------------------------------------------------------
x0, w0, _ = P[1]
pill(x0 + 14, 22, "2", "DIAGNOSIS", PUR, PUR_BG)
text(x0 + 14, 50, "Inside, it still knows", size=15.5, b=True)
group(robot(PUR, "think"), x0 + 40, 104, 96, vb=48)
group(g_lens(PUR), x0 + 22, 104, 40, rot=-8)
# thought: "can't tell" (blurred photo + question)
thought(x0 + 150, 78, 98, 46, (x0 + 122, 112))
group(g_image(INK, True), x0 + 160, 89, 24)
text(x0 + 190, 87, "can’t", size=11, c=PUR, b=True)
text(x0 + 190, 101, "tell!", size=11, c=PUR, b=True)
group(g_q(PUR, PUR_BG), x0 + 222, 89, 22)
# but says
bubble(x0 + 150, 150, 98, 30, (x0 + 132, 166))
text(x0 + 199, 158, "Strawberry!", size=12.5, b=True, c=RED, anchor="middle")


def tag(glyph_body, line1, line2, col):
    def body(sx, sy):
        group(glyph_body, sx + 9, sy + 11, 24)
        text(sx + 40, sy + 11, line1, size=10, c=MUTED, b=True)
        text(sx + 40, sy + 24, line2, size=12.5, c=col, b=True)
    return body


sticker(x0 + 72, 250, 112, 46, -3, PUR_BG, tag(g_check(PUR), "INSIDE", "it knows", PUR))
sticker(x0 + 190, 246, 112, 46, 3, RED_BG, tag(g_cross(RED), "OUTSIDE", "it guesses", RED))
group(g_spark(GOLD), x0 + 136, 92, 14)
group(g_spark(GOLD), x0 + 30, 196, 10)

# ---- (c) the goal / fix -------------------------------------------------------------
x0, w0, _ = P[2]
pill(x0 + 14, 22, "3", "OUR FIX", GRN, GRN_BG)
text(x0 + 14, 50, "Abstain when unsure, stay robust", size=15.5, b=True)
photo(x0 + 20, 82, 46, URI, round(46 * ASP))
group(robot(GRN, "happy"), x0 + 78, 92, 70, vb=48)
bubble(x0 + 156, 96, 112, 34, (x0 + 146, 120), stroke=GRN, sw=1.6)
text(x0 + 212, 106, "Unanswerable.", size=12.5, b=True, c=GRN, anchor="middle")
group(g_check(GRN), x0 + 258, 86, 20)
text(x0 + 212, 136, "the image can’t tell", size=10, c=MUTED, i=True, anchor="middle")
group(g_spark(GOLD), x0 + 150, 84, 12)
group(g_spark(GOLD), x0 + 72, 160, 9)


def kept(glyph_body, label, status):
    def body(sx, sy):
        group(glyph_body, sx + 8, sy + 8, 22)
        group(g_shield(GRN), sx + 54, sy + 8, 20)
        text(sx + 40, sy + 34, label, size=10 if width(label, 10, True) < 68 else 8.4, c=MUTED, b=True, anchor="middle")
        text(sx + 40, sy + 48, status, size=13, c=GRN, b=True, anchor="middle")
    return body


sticker(x0 + 52, 244, 80, 72, -4, GRN_BG, kept(g_image(INK), "CLEAN", "gain kept"))
sticker(x0 + 140, 240, 80, 72, 3, GRN_BG, kept(g_image(INK, True), "CORRUPTED", "gain kept"))
sticker(x0 + 228, 246, 80, 72, -3, GRN_BG, kept(g_q(INK), "UNANSWERABLE", "abstains"))

# ---------------------------------------------------------------- write
doc = (f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="{W}" height="{H}" '
       f'viewBox="0 0 {W} {H}"><rect width="{W}" height="{H}" fill="#FFFFFF"/>' + "".join(svg) + "</svg>")
(OUT / "teaser_v3.svg").write_text(doc)
import cairosvg  # noqa: E402
cairosvg.svg2png(bytestring=doc.encode(), write_to=str(OUT / "teaser_v3.png"), scale=3)
cairosvg.svg2pdf(bytestring=doc.encode(), write_to=str(OUT / "teaser_v3.pdf"))
print("wrote", OUT / "teaser_v3.svg")
