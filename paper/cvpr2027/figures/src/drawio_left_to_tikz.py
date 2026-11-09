"""Convert the left half of ../drawio_files/teaser.drawio (everything except the plot, whose
cells have ids "plot-*") into TikZ, so that edits made in draw.io reach teaser.pdf.

Handles the shapes used there: (rounded) rectangles, the ViT trapezoid, ellipses, the photo,
text, the check/cross glyphs and edges (straight, polyline, curved orthogonal).
draw.io px -> pt: divide by 2 (the drawing was laid out at 2 px per pt).

Run:  python3 drawio_left_to_tikz.py   ->  teaser_left.tex (\\input by fig1.tex)
      python3 drawio_left_to_tikz.py --skip 1,2,15 --out teaser_qual_left.tex   (cells by the
      number at the end of their id; coordinates stay those of the full drawing)
"""
import argparse
import os
import xml.etree.ElementTree as ET

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "drawio_files", "teaser.drawio")
ap = argparse.ArgumentParser()
ap.add_argument("--skip", default="", help="comma-separated id suffixes of cells to leave out")
ap.add_argument("--out", default="teaser_left.tex")
args = ap.parse_args()
OUT = os.path.join(HERE, args.out)
SKIP = {x for x in args.skip.split(",") if x}
PHOTO = "vizwiz_00002295.jpg"


def style(c):
    return dict((kv.split("=", 1) + ["1"])[:2] for kv in (c.get("style") or "").split(";") if kv)


cells = [c for c in ET.parse(SRC).getroot().iter("mxCell")
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


def tex(v):
    for a, b in (("&", r"\&"), ("%", r"\%"), ("#", r"\#"), ("_", r"\_")):
        v = v.replace(a, b)
    return v


def line_opts(st, default_w=1.0):
    """Stroke options, or None when the shape has no stroke."""
    sc = st.get("strokeColor", "#000000")
    if sc == "none":
        return None
    w = fl(float(st.get("strokeWidth", default_w)))
    o = [f"draw={col(sc)}", f"line width={w}pt"]
    if st.get("dashed") == "1":
        o.append(f"dash pattern=on {round(3 * w, 3)}pt off {round(3 * w, 3)}pt")
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
    flush = {"left": "flush left", "right": "flush right", "center": "flush center"}[al]
    return (rf"\node[anchor={anchor}, text width={fl(inner)}pt, align={flush}, font={font}, "
            rf"text={color}] at ({fx(tx)},{fy(ty)}) {{{tex(value)}}};")


def terminal(edge_st, other, prefix):
    """Point where an edge meets a connected shape (exit/entry constraint in unit coordinates)."""
    st, (x, y, w, h) = style(other), box(other)
    ux, uy = float(edge_st[prefix + "X"]), float(edge_st[prefix + "Y"])
    if st.get("direction") == "south":       # constraints rotate with the shape
        ux, uy = 0.5 - (uy - 0.5), 0.5 + (ux - 0.5)
    return x + ux * w, y + uy * h


body = []
for c in cells:
    st, value = style(c), c.get("value") or ""
    if c.get("edge"):
        g = c.find("mxGeometry")
        P = {p.get("as"): (float(p.get("x")), float(p.get("y"))) for p in g.findall("mxPoint")}
        mids = [(float(p.get("x")), float(p.get("y"))) for a in g.findall("Array")
                for p in a.findall("mxPoint")]
        src = by_id.get(c.get("source")); tgt = by_id.get(c.get("target"))
        a = terminal(st, src, "exit") if src is not None and "exitX" in st else P["sourcePoint"]
        b = terminal(st, tgt, "entry") if tgt is not None and "entryX" in st else P["targetPoint"]
        o = line_opts(st, 1.0)
        if st.get("endArrow", "classic") != "none":
            o.append("-{Triangle[length=3pt,width=2.8pt]}")
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
        body.append(rf"\node[anchor=north west] at ({fx(x)},{fy(y)}) "
                    rf"{{\includegraphics[width={fl(w)}pt,height={fl(h)}pt]{{{PHOTO}}}}};")
        body.append(rf"\draw[{col(st.get('imageBorder', '#000000'))}, line width=0.5pt] "
                    rf"({fx(x)},{fy(y)}) rectangle ({fx(x + w)},{fy(y + h)});")
        continue
    if "text" in st:
        body.append(text_node(st, x, y, w, h, value, True))
        continue
    if "ellipse" in st and value in ("✓", "✕"):
        body.append(rf"\node at ({fx(x + w / 2)},{fy(y + h / 2)}) "
                    rf"{{{chr(92) + ('ok' if value == chr(0x2713) else 'no')}}};")
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
