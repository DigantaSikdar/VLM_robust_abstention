"""Convert a draw.io drawing into TikZ, so that edits made in draw.io reach the PDF. By default
the left half of ../drawio_files/teaser.drawio (everything except the plot, whose cells have
ids "plot-*"); with --input and --extract-images, any drawing using the shapes below.

Handles the shapes used there: (rounded) rectangles, the ViT trapezoid, ellipses, the photo,
text with simple HTML (i, b, sub, br, div, colors), the check/cross glyphs and edges (straight,
polyline, curved orthogonal).
draw.io px -> pt: divide by 2 (the drawing was laid out at 2 px per pt).

Run:  python3 drawio_left_to_tikz.py   ->  teaser_left.tex (\\input by fig1.tex)
      python3 drawio_left_to_tikz.py --skip 1,2,15 --out teaser_qual_left.tex   (cells by the
      number at the end of their id; coordinates stay those of the full drawing)
      python3 drawio_left_to_tikz.py --input ../drawio_files/method.drawio \\
          --out method_diagram.tex --extract-images method_img
"""
import argparse
import base64
import html.parser
import os
import re
import xml.etree.ElementTree as ET

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "drawio_files", "teaser.drawio")
ap = argparse.ArgumentParser()
ap.add_argument("--skip", default="", help="comma-separated id suffixes of cells to leave out")
ap.add_argument("--out", default="teaser_left.tex")
ap.add_argument("--input", default=SRC)
ap.add_argument("--css-line-box", action="store_true",
                help="place text like draw.io: every line gets its HTML line box (\\\\cssstrut)")
ap.add_argument("--extract-images", default="",
                help="save embedded images as <prefix>_N.<ext> instead of using the teaser photo")
args = ap.parse_args()
OUT = os.path.join(HERE, args.out)
SKIP = {x for x in args.skip.split(",") if x}
PHOTO = "vizwiz_00002295.jpg"


def style(c):
    return dict((kv.split("=", 1) + ["1"])[:2] for kv in (c.get("style") or "").split(";") if kv)


cells = [c for c in ET.parse(args.input).getroot().iter("mxCell")
         if c.find("mxGeometry") is not None and not c.get("id").startswith("plot-")]
by_id = {c.get("id"): c for c in cells}


def box(c):
    g = c.find("mxGeometry")
    return tuple(float(g.get(k, 0)) for k in ("x", "y", "width", "height"))


X0 = min(box(c)[0] for c in cells if not c.get("edge"))
Y0 = min(box(c)[1] for c in cells if not c.get("edge"))
cells = [c for c in cells if c.get("id").rsplit("-", 1)[-1] not in SKIP]  # origin of the full drawing
def fx(x): return round((x - X0) / 2, 3)
def fy(y): return round(-(y - Y0) / 2, 3)
def fl(v): return round(v / 2, 3)


colors = {}
def col(hexa):
    name = colors.setdefault(hexa.upper(), f"dio{len(colors)}")
    return name


MATH = {"\u03c0": r"\pi", "\u03b8": r"\theta", "\u03b1": r"\alpha", "\u03c4": r"\tau",
        "\u03bb": r"\lambda", "\u03bc": r"\mu", "\u03c3": r"\sigma", "\u03c1": r"\rho",
        "\u03a3": r"\Sigma", "\u2113": r"\ell", "\u2191": r"\uparrow", "\u2193": r"\downarrow",
        "\u2032": "'", "\u2212": "-", "\u2016": r"\|", "\u2298": r"\oslash", "\u22a4": r"\top",
        "\u22c6": r"\star", "\u2112": r"\mathcal{L}", "\U0001d4a5": r"\mathcal{J}",
        "\U0001d49e": r"\mathcal{C}", "\U0001d4ab": r"\mathcal{P}", "\U0001d4ae": r"\mathcal{S}",
        "\u2227": r"\wedge", "\U0001d7d9": r"\mathds{1}", "\ue001": r"\tilde{x}", "\ue002": r"\bar{h}"}
CHARS = {"&": r"\&", "%": r"\%", "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}", "\xa0": "~",
         "\u2776": r"\ding{182}", "\u2777": r"\ding{183}", "\u2778": r"\ding{184}",
         **{k: rf"\ensuremath{{{v}}}" for k, v in MATH.items()}}


class Html(html.parser.HTMLParser):
    """The small HTML subset of draw.io labels -> LaTeX."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.out, self.close = [], []

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == "br":
            self.out.append(r"\\ ")
            return
        if tag == "div" and self.out and not self.out[-1].endswith(r"\\ "):
            self.out.append(r"\\ ")
        pre, css = "", a.get("style", "")
        if tag in ("i", "em"):
            pre += r"\itshape "
        if tag in ("b", "strong"):
            pre += r"\bfseries "
        if re.search(r"font-weight:\s*(400|normal)", css):
            pre += r"\mdseries "
        if re.search(r"font-weight:\s*(700|bold)", css):
            pre += r"\bfseries "
        m = re.search(r"(?:^|[;\s])color:\s*rgb\((\d+),\s*(\d+),\s*(\d+)\)", css)
        if m:
            pre += rf"\color[RGB]{{{m[1]},{m[2]},{m[3]}}}"
        if "color" in a and a["color"].startswith("#"):
            pre += rf"\color[HTML]{{{a['color'][1:].upper()}}}"
        if tag in ("sub", "sup"):
            self.out.append(r"\text" + {"sub": "subscript", "sup": "superscript"}[tag] + "{{" + pre)
            self.close.append("}}")
        else:
            self.out.append("{" + pre)
            self.close.append("}")

    def handle_endtag(self, tag):
        if tag != "br" and self.close:
            self.out.append(self.close.pop())

    def handle_data(self, d):
        d = d.replace("x\u0303", "\ue001").replace("h\u0304", "\ue002")
        self.out.append("".join(CHARS.get(ch, ch) for ch in d))


def tex(v):
    h = Html()
    h.feed(v)
    return "".join(h.out) + "".join(reversed(h.close))


def line_opts(st, default_w=1.0):
    """Stroke options, or None when the shape has no stroke."""
    sc = st.get("strokeColor", "#000000")
    if sc == "none":
        return None
    w = fl(float(st.get("strokeWidth", default_w)))
    o = [f"draw={col(sc)}", f"line width={w}pt"]
    if st.get("dashed") == "1":
        on, off = (float(x) for x in st.get("dashPattern", "3 3").split()[:2])
        o.append(f"dash pattern=on {round(on * w, 3)}pt off {round(off * w, 3)}pt")
    return o


def fill_opts(st):
    f = st.get("fillColor", "none")
    return [] if f == "none" else [f"fill={col(f)}"]


def text_node(st, x, y, w, h, value, is_text):
    size = fl(float(st.get("fontSize", 12)))
    fs = int(st.get("fontStyle", 0))
    font = rf"\fs{{{size}}}" + (r"\bfseries" if fs & 1 else "") + (r"\itshape" if fs & 2 else "")
    color = col(st.get("fontColor", "#000000"))
    al = st.get("align", "center")
    pad_l = 0 if is_text else float(st.get("spacing", 2)) + float(st.get("spacingLeft", 0))
    pad_r = 0 if is_text else float(st.get("spacing", 2))
    va = st.get("verticalAlign", "middle")
    ty = {"top": y, "middle": y + h / 2, "bottom": y + h}[va]
    anchor = {"top": "north", "middle": "center", "bottom": "south"}[va]
    inner = w - pad_l - pad_r
    tx = x + pad_l + inner / 2
    label = tex(value)
    if args.css_line_box:
        label = r"\cssstrut " + label.replace(r"\\ ", r"\\ \cssstrut ")
        if al == "center":          # TeX breaks a line that only just fills the box; draw.io does not
            inner += 8
    flush = {"left": "flush left", "right": "flush right", "center": "flush center"}[al]
    return (rf"\node[anchor={anchor}, text width={fl(inner)}pt, align={flush}, font={font}, "
            rf"text={color}] at ({fx(tx)},{fy(ty)}) {{{label}}};")


def terminal(edge_st, other, prefix):
    """Point where an edge meets a connected shape (exit/entry constraint in unit coordinates)."""
    st, (x, y, w, h) = style(other), box(other)
    ux, uy = float(edge_st[prefix + "X"]), float(edge_st[prefix + "Y"])
    if st.get("direction") == "south":       # constraints rotate with the shape
        ux, uy = 0.5 - (uy - 0.5), 0.5 + (ux - 0.5)
    return x + ux * w, y + uy * h


body, n_images = [], 0
for c in cells:
    st, value = style(c), c.get("value") or ""
    if c.get("edge"):
        g = c.find("mxGeometry")
        P = {p.get("as"): (float(p.get("x")), float(p.get("y"))) for p in g.findall("mxPoint")}
        mids = [(float(p.get("x")), float(p.get("y"))) for a in g.findall("Array")
                for p in a.findall("mxPoint")]
        src = by_id.get(c.get("source")); tgt = by_id.get(c.get("target"))
        b = terminal(st, tgt, "entry") if tgt is not None and "entryX" in st else P["targetPoint"]
        if src is not None and "exitX" in st:
            a = terminal(st, src, "exit")
        elif src is not None:                # floating exit: leave the right side level with b
            sx, sy, sw, sh = box(src)
            a = (sx + sw, min(max(b[1], sy), sy + sh))
        else:
            a = P["sourcePoint"]
        o = line_opts(st, 1.0)
        if st.get("endArrow", "classic") != "none":
            size = float(st.get("endSize", 6))
            o.append(f"-{{Triangle[length={round(size * 0.75, 2)}pt,width={round(size * 0.7, 2)}pt]}}")
        o = ", ".join(o).replace("draw=", "color=", 1)
        if st.get("curved") == "1" and not mids:
            m = (a[0] + b[0]) / 2
            path = (f"({fx(a[0])},{fy(a[1])}) .. controls ({fx(m)},{fy(a[1])}) and "
                    f"({fx(m)},{fy(b[1])}) .. ({fx(b[0])},{fy(b[1])})")
        else:
            path = " -- ".join(f"({fx(p[0])},{fy(p[1])})" for p in [a] + mids + [b])
        body.append(rf"\draw[{o}] {path};")
        continue

    x, y, w, h = box(c)
    if st.get("shape") == "image":
        photo = PHOTO
        if args.extract_images:
            kind, data = re.match(r"data:image/(\w+)[,;](?:base64,)?(.*)", st["image"], re.S).groups()
            n_images += 1
            photo = f"{args.extract_images}_{n_images}.{'jpg' if kind == 'jpeg' else kind}"
            with open(os.path.join(HERE, photo), "wb") as fh:
                fh.write(base64.b64decode(data + "=" * (-len(data) % 4)))
        body.append(rf"\node[anchor=north west] at ({fx(x)},{fy(y)}) "
                    rf"{{\includegraphics[width={fl(w)}pt,height={fl(h)}pt]{{{photo}}}}};")
        body.append(rf"\draw[{col(st.get('imageBorder', '#000000'))}, line width=0.5pt] "
                    rf"({fx(x)},{fy(y)}) rectangle ({fx(x + w)},{fy(y + h)});")
        continue
    if "text" in st:
        body.append(text_node(st, x, y, w, h, value, True))
        continue
    if "ellipse" in st and value in ("✓", "✕"):
        glyph = chr(92) + ("ok" if value == chr(0x2713) else "no")
        if abs(w - 10.8) > 0.05:
            glyph = rf"\scalebox{{{round(w / 10.8, 3)}}}{{{glyph}}}"
        body.append(rf"\node at ({fx(x + w / 2)},{fy(y + h / 2)}) {{{glyph}}};")
        continue

    opts = (line_opts(st) or []) + fill_opts(st)
    opts = ", ".join(opts + ["line join=miter"])
    if "ellipse" in st:
        path = f"({fx(x + w / 2)},{fy(y + h / 2)}) ellipse ({fl(w / 2)}pt and {fl(h / 2)}pt)"
    elif st.get("shape") == "trapezoid":        # direction=south: wide side left, narrow right
        d = h * float(st.get("size", 0.2))
        path = (f"({fx(x)},{fy(y)}) -- ({fx(x + w)},{fy(y + d)}) -- ({fx(x + w)},{fy(y + h - d)})"
                f" -- ({fx(x)},{fy(y + h)}) -- cycle")
    else:
        r = fl(min(w, h) * float(st.get("arcSize", 15)) / 100) if st.get("rounded") == "1" else 0
        path = f"[rounded corners={r}pt] ({fx(x)},{fy(y)}) rectangle ({fx(x + w)},{fy(y + h)})"
    body.append(rf"\path[{opts}] {path};")
    if value:
        body.append(text_node(st, x, y, w, h, value, False))

head = ["% Generated by drawio_left_to_tikz.py from ../drawio_files/teaser.drawio. Do not edit."]
head += [rf"\definecolor{{{n}}}{{HTML}}{{{h[1:]}}}" for h, n in colors.items()]
open(OUT, "w", encoding="utf-8").write("\n".join(head + body) + "\n")
ys = [box(c)[1] + box(c)[3] for c in cells if not c.get("edge")]
print(OUT, len(body), "items; height", (max(ys) - Y0) / 2, "pt; x0, y0 =", X0, Y0)
