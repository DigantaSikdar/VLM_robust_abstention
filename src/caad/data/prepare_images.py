"""
caad/data/prepare_images.py
===========================
Render corrupted copies of clean images and write a CAAD image manifest
(schema: data/dataset.py).

    # tiny synthetic set for the smoke test (no real data needed)
    python -m caad.data.prepare_images synthetic --n 8 \
        --out-dir data/smoke_img --manifest data/smoke_img/manifest.jsonl --hw 224

    # real data: source JSONL rows {"image_id"?, "image", "question", "answer"?}
    python -m caad.data.prepare_images real \
        --source data/raw/images.jsonl --out-dir data/img_corrupted \
        --manifest data/img_train_manifest.jsonl --styles fog,night,jpeg --severity 3

Both the clean and the corrupted image are written as PNG from the SAME decoded
array. That guarantees they decode to identical (H, W) under qwen-vl-utils
(source JPEGs with EXIF rotation would otherwise be able to disagree between
loaders), which the cross-input alignment invariant requires. Pass
--keep-clean-source to point ``clean_image`` at the original file instead.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .corruption import CORRUPTION_STYLES, SEVERITIES, derive_seed
from .dataset import render_corrupted


def _load_rgb(path) -> np.ndarray:
    from PIL import Image
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"))


def _save_rgb(arr, path):
    from PIL import Image
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path)


def _parse_list(s, all_values, cast=str):
    if s == "all":
        return list(all_values)
    vals = [cast(v) for v in s.split(",") if v.strip()]
    bad = [v for v in vals if v not in all_values]
    if bad:
        raise SystemExit(f"unknown values {bad}; choose from {list(all_values)}")
    return vals


def _jobs(image_id, styles, severities, sample_one, global_seed):
    """(style, severity) pairs to render for one image. ``sample_one`` picks a
    single pair, deterministically from the image identity."""
    pairs = [(st, sv) for st in styles for sv in severities]
    if not sample_one:
        return pairs
    rng = np.random.default_rng(derive_seed(image_id, "pick", 0, global_seed))
    return [pairs[rng.integers(len(pairs))]]


def build_manifest(items, *, out_dir, manifest, styles, severities, global_seed,
                   sample_one=False, keep_clean_source=False):
    """items: iterable of dicts with image_id, image (path or ndarray), question,
    optional answer. Writes PNGs under out_dir and the manifest JSONL."""
    out_dir = Path(out_dir)
    manifest = Path(manifest)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with manifest.open("w") as fh:
        for it in items:
            iid = str(it["image_id"])
            src = it["image"]
            img = src if isinstance(src, np.ndarray) else _load_rgb(src)
            if keep_clean_source and not isinstance(src, np.ndarray):
                clean_path = str(src)
            else:
                clean_path = str(out_dir / "clean" / f"{iid}.png")
                _save_rgb(img, Path(clean_path))
            for style, sev in _jobs(iid, styles, severities, sample_one, global_seed):
                corr, spec = render_corrupted(
                    img, image_id=iid, style=style, severity=sev, global_seed=global_seed)
                corr_path = out_dir / "corrupted" / f"{iid}__{style}__s{sev}.png"
                _save_rgb(corr, corr_path)
                row = {"image_id": iid, "question": it["question"],
                       "clean_image": clean_path, "corrupted_image": str(corr_path),
                       "corruption": {"style": style, "severity": sev, "seed": spec.seed}}
                if it.get("answer") is not None:
                    row["answer"] = it["answer"]
                fh.write(json.dumps(row) + "\n")
                n += 1
    print(f"wrote {n} rows -> {manifest}")
    return n


# --------------------------------------------------------------------------- #
_COLORS = {"red": (220, 40, 40), "green": (40, 180, 60),
           "blue": (40, 80, 220), "yellow": (230, 210, 40)}


def synthetic_items(n, hw, seed=0):
    """Random-background images with one large colored rectangle and an MCQ
    about its color — enough to exercise the pipeline, not to learn anything."""
    rng = np.random.default_rng(seed)
    names = list(_COLORS)
    for i in range(n):
        img = rng.integers(0, 256, (hw, hw, 3), dtype=np.uint8)
        k = int(rng.integers(len(names)))
        y0, x0 = rng.integers(0, hw // 2, size=2)
        img[y0:y0 + hw // 2, x0:x0 + hw // 2] = _COLORS[names[k]]
        opts = " ".join(f"{chr(65 + j)}) {c}" for j, c in enumerate(names))
        yield {"image_id": f"syn{i:04d}", "image": img,
               "question": f"What color is the large rectangle? {opts}",
               "answer": chr(65 + k)}


def real_items(source, image_root=None):
    root = Path(image_root) if image_root else None
    for i, line in enumerate(Path(source).read_text().splitlines()):
        if not line.strip():
            continue
        r = json.loads(line)
        p = Path(r["image"])
        if root and not p.is_absolute():
            p = root / p
        yield {"image_id": r.get("image_id", r.get("id", f"img{i:07d}")),
               "image": str(p), "question": r["question"], "answer": r.get("answer")}


def main():
    ap = argparse.ArgumentParser(description="render corrupted images + CAAD manifest")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("synthetic", "real"):
        sp = sub.add_parser(name)
        sp.add_argument("--out-dir", required=True)
        sp.add_argument("--manifest", required=True)
        sp.add_argument("--styles", default="all",
                        help=f"comma list or 'all' of {CORRUPTION_STYLES}")
        sp.add_argument("--severity", default="3", help="comma list of 1..5, or 'all'")
        sp.add_argument("--sample-one", action="store_true",
                        help="one random (style, severity) per image instead of every combination")
        sp.add_argument("--seed", type=int, default=0)
        if name == "synthetic":
            sp.add_argument("--n", type=int, default=8)
            sp.add_argument("--hw", type=int, default=224)
        else:
            sp.add_argument("--source", required=True)
            sp.add_argument("--image-root", default=None)
            sp.add_argument("--keep-clean-source", action="store_true")
    a = ap.parse_args()

    styles = _parse_list(a.styles, CORRUPTION_STYLES)
    sevs = _parse_list(a.severity, SEVERITIES, cast=int)
    if a.cmd == "synthetic":
        items, keep = synthetic_items(a.n, a.hw, a.seed), False
    else:
        items, keep = real_items(a.source, a.image_root), a.keep_clean_source
    build_manifest(items, out_dir=a.out_dir, manifest=a.manifest, styles=styles,
                   severities=sevs, global_seed=a.seed, sample_one=a.sample_one,
                   keep_clean_source=keep)


if __name__ == "__main__":
    main()
