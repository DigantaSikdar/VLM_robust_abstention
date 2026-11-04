"""
caad/data/prepare_vqa_c.py
==========================
Build the TextVQA-C / GQA-C evaluation set (Usama et al., 2025) under a
balanced single-draw protocol: every question is evaluated once on its clean
image and once on ONE corrupted copy, whose (ImageNet-C corruption, severity)
is drawn per question so that each of the 19 x 5 = 95 cells receives the same
number of questions (+-1) within each dataset.

The assignment is fixed by ``--seed`` and written into the manifests, so every
model / checkpoint is scored on exactly the same corrupted images.

    # 1) fixed assignment + manifests + clean images (seconds)
    python -m caad.data.prepare_vqa_c assign \
        --raw /gscratch/dsikdar/caad_data/raw --out /gscratch/dsikdar/caad_data/vqa_c

    # 2) render corrupted PNGs (CPU-heavy; resumable, shardable for SLURM)
    python -m caad.data.prepare_vqa_c render --out /gscratch/dsikdar/caad_data/vqa_c \
        --workers 90 [--shard 0 --num-shards 1]

Layout under --out:
    {textvqa,gqa}/manifest.jsonl     one row per question (schema below)
    {textvqa,gqa}/clean/<image_id>.<ext>      original bytes, untouched
    {textvqa,gqa}/corrupted/<qid>__<corruption>__s<sev>.png   lossless
    assignment.json                  seed, versions, per-cell counts

Manifest row:
    {"dataset": "gqa", "question_id": "201307251", "image_id": "n161313",
     "question": "Is it overcast?", "answer": "no",          # gqa
     "answers": ["2010", ...],                                # textvqa (10)
     "clean_image": ".../clean/n161313.jpg",
     "corrupted_image": ".../corrupted/201307251__frost__s4.png",
     "corruption": {"name": "frost", "category": "weather", "severity": 4,
                    "seed": 123456789, "key": "gqa:201307251",
                    "seen_in_training": false, "caad_counterpart": null}}
"""

from __future__ import annotations

import argparse
import glob
import io
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from .corruption import derive_seed
from .imagenet_c import CATEGORIES, IMAGENET_C, SEVERITIES, corrupt_image

DATASETS = ("textvqa", "gqa")
CATEGORY_OF = {n: c for c, ns in CATEGORIES.items() for n in ns}
# ImageNet-C corruptions with a same-kind counterpart in CAAD's training engine
# (caad.data.corruption). Everything else is unseen during training.
CAAD_COUNTERPART = {
    "gaussian_noise": "gaussian_noise", "gaussian_blur": "gaussian_blur",
    "motion_blur": "motion_blur", "snow": "snow", "fog": "fog",
    "brightness": "overexposure", "jpeg_compression": "jpeg", "pixelate": "pixelate",
}


# --------------------------------------------------------------------------- #
# Raw data (lmms-lab parquet mirrors of the official splits)
# --------------------------------------------------------------------------- #
def _read(pattern, columns=None):
    import pyarrow.parquet as pq
    rows = []
    for f in sorted(glob.glob(pattern)):
        rows += pq.read_table(f, columns=columns).to_pylist()
    if not rows:
        raise SystemExit(f"no parquet files match {pattern}")
    return rows


def load_textvqa(raw):
    rows = _read(f"{raw}/textvqa/data/validation-*.parquet",
                 ["question_id", "question", "image_id", "image", "answers"])
    images = {r["image_id"]: r["image"]["bytes"] for r in rows}
    qs = [{"question_id": str(r["question_id"]), "image_id": r["image_id"],
           "question": r["question"], "answers": list(r["answers"])} for r in rows]
    return qs, images


def load_gqa(raw):
    rows = _read(f"{raw}/GQA/testdev_balanced_instructions/*.parquet",
                 ["id", "imageId", "question", "answer", "fullAnswer"])
    images = {r["id"]: r["image"]["bytes"]
              for r in _read(f"{raw}/GQA/testdev_balanced_images/*.parquet")}
    qs = [{"question_id": str(r["id"]), "image_id": r["imageId"], "question": r["question"],
           "answer": r["answer"], "full_answer": r["fullAnswer"]} for r in rows]
    return qs, images


def load_vqav2(raw, n=5000, seed=0,
               train_manifest="/gscratch/dsikdar/caad_data/gqa_train_30k/manifest.jsonl",
               vg_meta="/gscratch/dsikdar/caad_data/raw/VG/image_data.json"):
    """A random subset of VQAv2 val questions whose COCO images are NOT among the
    CAAD training images (GQA train images are Visual Genome images, ~half of
    which are COCO images; Visual Genome's image_data.json gives the mapping)."""
    import pyarrow.parquet as pq
    vg2coco = {str(x["image_id"]): x["coco_id"] for x in json.load(open(vg_meta)) if x.get("coco_id")}
    train_coco = {vg2coco[json.loads(l)["image_id"]] for l in open(train_manifest)
                  if json.loads(l)["image_id"] in vg2coco}
    files = sorted(glob.glob(f"{raw}/VQAv2/data/validation-*.parquet"))
    meta = []
    for fi, f in enumerate(files):
        t = pq.read_table(f, columns=["question_id", "image_id", "question", "answers", "answer_type"]).to_pylist()
        meta += [(fi, i, r) for i, r in enumerate(t) if int(r["image_id"]) not in train_coco]
    rng = np.random.default_rng(derive_seed("vqav2", "subset", 0, seed))
    pick = sorted(rng.choice(len(meta), size=n, replace=False))
    chosen = [meta[i] for i in pick]
    by_file = defaultdict(list)
    for fi, i, r in chosen:
        by_file[fi].append(i)
    images = {}
    for fi, idx in by_file.items():
        t = pq.read_table(files[fi], columns=["image_id", "image"]).take(idx).to_pylist()
        images.update({str(r["image_id"]): r["image"]["bytes"] for r in t})
    qs = [{"question_id": str(r["question_id"]), "image_id": str(r["image_id"]), "question": r["question"],
           "answers": [a["answer"] for a in r["answers"]], "category": r["answer_type"]} for _, _, r in chosen]
    print(f"vqav2: {len(meta)} eligible questions (excluded images shared with CAAD training: "
          f"{len(train_coco)} COCO ids), sampled {len(qs)}")
    return qs, images


LOADERS = {"textvqa": load_textvqa, "gqa": load_gqa, "vqav2": load_vqav2}


# --------------------------------------------------------------------------- #
# Balanced assignment
# --------------------------------------------------------------------------- #
CELLS = [(n, s) for n in IMAGENET_C for s in SEVERITIES]


def balanced_assignment(question_ids, seed, dataset):
    """question_id -> (corruption, severity). Each of the 95 cells gets
    floor(n/95) or ceil(n/95) questions; which cells get the extra one, and
    which question lands in which cell, are random but fixed by ``seed``."""
    ids = sorted(question_ids, key=lambda q: (len(q), q))      # stable, numeric-aware
    rng = np.random.default_rng(derive_seed(dataset, "assignment", 0, seed))
    order = [CELLS[i] for i in rng.permutation(len(CELLS))]
    tiled = (order * math.ceil(len(ids) / len(CELLS)))[:len(ids)]
    perm = rng.permutation(len(ids))
    return {ids[i]: tiled[j] for j, i in enumerate(perm)}


def _ext(img_bytes):
    from PIL import Image
    fmt = (Image.open(io.BytesIO(img_bytes)).format or "png").lower()
    return {"jpeg": "jpg", "mpo": "jpg"}.get(fmt, fmt)   # MPO = multi-picture JPEG


def cmd_assign(a):
    import imagecorruptions
    out = Path(a.out)
    summary = {"protocol": "balanced single-draw (19 ImageNet-C corruptions x 5 severities)",
               "seed": a.seed, "imagecorruptions": imagecorruptions.__version__ if hasattr(
                   imagecorruptions, "__version__") else "1.1.2",
               "numpy": np.__version__, "datasets": {}}
    for ds in a.datasets.split(","):
        qs, images = LOADERS[ds](a.raw)
        d = out / ds
        (d / "clean").mkdir(parents=True, exist_ok=True)
        (d / "corrupted").mkdir(parents=True, exist_ok=True)
        clean_path = {}
        for iid, b in images.items():
            p = d / "clean" / f"{iid}.{_ext(b)}"
            if not p.exists():
                p.write_bytes(b)                        # original bytes, no re-encode
            clean_path[iid] = str(p)
        assign = balanced_assignment([q["question_id"] for q in qs], a.seed, ds)
        with (d / "manifest.jsonl").open("w") as fh:
            for q in qs:
                name, sev = assign[q["question_id"]]
                key = f"{ds}:{q['question_id']}"
                row = {"dataset": ds, **q, "clean_image": clean_path[q["image_id"]],
                       "corrupted_image": str(d / "corrupted" /
                                              f"{q['question_id']}__{name}__s{sev}.png"),
                       "corruption": {
                           "name": name, "category": CATEGORY_OF[name], "severity": sev,
                           "seed": derive_seed(key, name, sev, a.seed) % (2 ** 32),
                           "key": key, "global_seed": a.seed,
                           "seen_in_training": name in CAAD_COUNTERPART,
                           "caad_counterpart": CAAD_COUNTERPART.get(name)}}
                fh.write(json.dumps(row) + "\n")
        cells = Counter(assign.values())
        summary["datasets"][ds] = {
            "questions": len(qs), "images": len(images),
            "per_cell_min": min(cells.values()), "per_cell_max": max(cells.values()),
            "per_category": dict(Counter(CATEGORY_OF[n] for n, _ in assign.values())),
            "per_severity": {str(k): v for k, v in sorted(Counter(s for _, s in assign.values()).items())},
            "seen_unseen": dict(Counter("seen" if n in CAAD_COUNTERPART else "unseen"
                                        for n, _ in assign.values())),
            "cells": {f"{n}@{s}": cells[(n, s)] for n, s in CELLS},
        }
        print(f"{ds}: {len(qs)} questions, {len(images)} images, "
              f"{min(cells.values())}-{max(cells.values())} per cell")
    prev = out / "assignment.json"
    if prev.exists():                                # keep other datasets' entries when adding one
        old = json.loads(prev.read_text())
        old["datasets"].update(summary["datasets"]); summary = old
    (out / "assignment.json").write_text(json.dumps(summary, indent=2))
    print("wrote", out / "assignment.json")


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
def _render_one(row):
    from PIL import Image
    dst = Path(row["corrupted_image"])
    if dst.exists() and dst.stat().st_size > 0:
        return "skip"
    with Image.open(row["clean_image"]) as im:
        img = np.asarray(im.convert("RGB"))
    c = row["corruption"]
    out = corrupt_image(img, c["name"], c["severity"], image_id=c["key"],
                        global_seed=c["global_seed"])
    tmp = dst.with_name(dst.name + ".tmp")
    Image.fromarray(out).save(tmp, format="PNG")
    os.replace(tmp, dst)                                # atomic: never a half-written PNG
    return "done"


def cmd_render(a):
    import warnings
    from multiprocessing import Pool
    warnings.filterwarnings("ignore")
    rows = []
    for ds in a.datasets.split(","):
        rows += [json.loads(l) for l in (Path(a.out) / ds / "manifest.jsonl").read_text().splitlines()]
    rows = [r for i, r in enumerate(rows) if i % a.num_shards == a.shard]
    # slowest corruptions first so the tail of the pool isn't one glass_blur
    rows.sort(key=lambda r: r["corruption"]["name"] not in ("glass_blur", "saturate", "brightness"))
    print(f"shard {a.shard}/{a.num_shards}: {len(rows)} images, {a.workers} workers", flush=True)
    stats = Counter()
    with Pool(a.workers) as pool:
        for i, s in enumerate(pool.imap_unordered(_render_one, rows, chunksize=1), 1):
            stats[s] += 1
            if i % 500 == 0 or i == len(rows):
                print(f"  {i}/{len(rows)} {dict(stats)}", flush=True)


def main():
    ap = argparse.ArgumentParser(description="build TextVQA-C / GQA-C (balanced single-draw)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("assign")
    s.add_argument("--raw", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--datasets", default="textvqa,gqa")
    s = sub.add_parser("render")
    s.add_argument("--out", required=True)
    s.add_argument("--workers", type=int, default=os.cpu_count())
    s.add_argument("--shard", type=int, default=0)
    s.add_argument("--num-shards", type=int, default=1)
    s.add_argument("--datasets", default="textvqa,gqa")
    a = ap.parse_args()
    {"assign": cmd_assign, "render": cmd_render}[a.cmd](a)


if __name__ == "__main__":
    main()
