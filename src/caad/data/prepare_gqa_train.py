"""
caad/data/prepare_gqa_train.py
==============================
CAAD training set from GQA train_balanced: N questions drawn uniformly at
random, each paired with ONE corrupted copy of its image from CAAD's own
corruption engine (caad.data.corruption, 16 types) at a fixed severity.

Corruption types are balanced (each type gets N/16 +-1 questions); which
question gets which type is random but fixed by --seed. The engine is
deliberately NOT ImageNet-C, so TextVQA-C / GQA-C keep held-out corruptions.
GQA train and testdev share no images.

    python -m caad.data.prepare_gqa_train assign \
        --raw /gscratch/dsikdar/caad_data/raw --out /gscratch/dsikdar/caad_data/gqa_train_30k \
        --n 30000 --severity 1 --seed 0
    python -m caad.data.prepare_gqa_train render --out /gscratch/dsikdar/caad_data/gqa_train_30k \
        --workers 26 [--shard i --num-shards N]

Layout under --out:
    manifest.jsonl          one row per question (schema below), training input
    clean/<image_id>.png    decoded once, lossless
    corrupted/<qid>__<style>__s<sev>.png
    assignment.json         seed, counts per style, distinct images

Row (read by caad.data.dataset.ManifestDataset):
    {"image_id": "2354786", "question_id": "02930152", "question": "...",
     "answer": "yes", "full_answer": "...",
     "clean_image": ".../clean/2354786.png",
     "corrupted_image": ".../corrupted/02930152__fog__s1.png",
     "corruption": {"style": "fog", "severity": 1, "seed": 123, "key": "gqa_train:02930152"}}
"""

from __future__ import annotations

import argparse
import glob
import io
import json
import math
import os
from collections import Counter
from pathlib import Path

import numpy as np

from .corruption import (CORRUPTION_STYLES, CorruptionSpec, apply_corruption, derive_seed,
                         make_spec)


def cmd_assign(a):
    import pyarrow.parquet as pq
    raw = Path(a.raw) / "GQA"
    files = sorted(glob.glob(str(raw / "train_balanced_instructions" / "*.parquet")))
    qs = []
    for f in files:
        qs += pq.read_table(f, columns=["id", "imageId", "question", "answer", "fullAnswer"]).to_pylist()
    qs.sort(key=lambda r: r["id"])                           # stable base order
    rng = np.random.default_rng(derive_seed("gqa_train", "sample", 0, a.seed))
    pick = [qs[i] for i in sorted(rng.choice(len(qs), size=a.n, replace=False))]

    styles = [CORRUPTION_STYLES[i] for i in rng.permutation(len(CORRUPTION_STYLES))]
    tiled = (styles * math.ceil(a.n / len(styles)))[:a.n]
    order = rng.permutation(a.n)

    out = Path(a.out)
    (out / "clean").mkdir(parents=True, exist_ok=True)
    (out / "corrupted").mkdir(parents=True, exist_ok=True)
    with (out / "manifest.jsonl").open("w") as fh:
        for j, i in enumerate(order):
            q = pick[i]
            style = tiled[j]
            key = f"gqa_train:{q['id']}"
            spec = make_spec(key, style, a.severity, a.seed)
            row = {"image_id": q["imageId"], "question_id": q["id"], "question": q["question"],
                   "answer": q["answer"], "full_answer": q["fullAnswer"],
                   "clean_image": str(out / "clean" / f"{q['imageId']}.png"),
                   "corrupted_image": str(out / "corrupted" / f"{q['id']}__{style}__s{a.severity}.png"),
                   "corruption": {"style": style, "severity": a.severity, "seed": spec.seed,
                                  "key": key, "engine": "caad.data.corruption"}}
            fh.write(json.dumps(row) + "\n")
    counts = Counter(tiled)
    summary = {"source": "GQA train_balanced", "pool_questions": len(qs), "n": a.n,
               "seed": a.seed, "severity": a.severity, "engine": "caad.data.corruption",
               "distinct_images": len({q["imageId"] for q in pick}),
               "per_style": dict(sorted(counts.items()))}
    (out / "assignment.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "per_style"}),
          f"per style {min(counts.values())}-{max(counts.values())}")


def _image_bytes(raw, needed):
    """image_id -> bytes, reading only the shards' rows we need."""
    import pyarrow.parquet as pq
    found = {}
    for f in sorted(glob.glob(str(Path(raw) / "GQA" / "train_balanced_images" / "*.parquet"))):
        t = pq.read_table(f, columns=["id"])
        ids = t.column("id").to_pylist()
        idx = [i for i, x in enumerate(ids) if x in needed and x not in found]
        if idx:
            imgs = pq.read_table(f, columns=["id", "image"]).take(idx).to_pylist()
            found.update({r["id"]: r["image"]["bytes"] for r in imgs})
    missing = needed - set(found)
    if missing:
        raise SystemExit(f"{len(missing)} images not found in train_balanced_images")
    return found


def _render_one(job):
    """job: (row, clean_png_bytes_or_None). Writes clean (if absent) + corrupted."""
    from PIL import Image
    row, src = job
    clean, dst = Path(row["clean_image"]), Path(row["corrupted_image"])
    if dst.exists() and dst.stat().st_size > 0 and clean.exists():
        return "skip"
    img = np.asarray(Image.open(io.BytesIO(src)).convert("RGB"))
    if not clean.exists():
        tmp = clean.with_name(f"{clean.name}.{os.uname().nodename}.{os.getpid()}.tmp")   # shards may race
        Image.fromarray(img).save(tmp, format="PNG")
        os.replace(tmp, clean)
    c = row["corruption"]                                     # the recorded seed is authoritative
    spec = CorruptionSpec(image_id=c["key"], style=c["style"], severity=c["severity"], seed=c["seed"])
    out = apply_corruption(img, spec)
    assert out.shape == img.shape
    tmp = dst.with_name(dst.name + ".tmp")
    Image.fromarray(out).save(tmp, format="PNG")
    os.replace(tmp, dst)
    return "done"


def cmd_render(a):
    from multiprocessing import Pool
    out = Path(a.out)
    rows = [json.loads(l) for l in (out / "manifest.jsonl").read_text().splitlines()]
    rows = [r for i, r in enumerate(rows) if i % a.num_shards == a.shard]
    todo = [r for r in rows if not (Path(r["corrupted_image"]).exists() and Path(r["clean_image"]).exists())]
    print(f"shard {a.shard}/{a.num_shards}: {len(rows)} rows, {len(todo)} to render", flush=True)
    if not todo:
        return
    # group by image so a clean PNG is written by exactly one task per image
    by_img = {}
    for r in todo:
        by_img.setdefault(r["image_id"], []).append(r)
    src = _image_bytes(a.raw, set(by_img))
    jobs = []
    for iid, rs in by_img.items():
        jobs.append((rs[0], src[iid]))                        # first row also writes the clean image
    stats = Counter()
    with Pool(a.workers) as pool:
        for s in pool.imap_unordered(_render_one, jobs, chunksize=4):
            stats[s] += 1
        rest = [(r, src[iid]) for iid, rs in by_img.items() for r in rs[1:]]
        for i, s in enumerate(pool.imap_unordered(_render_one, rest, chunksize=4), 1):
            stats[s] += 1
    print(f"shard {a.shard}: {dict(stats)}", flush=True)


def main():
    ap = argparse.ArgumentParser(description="CAAD GQA training set (random corruption, fixed severity)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("assign")
    s.add_argument("--raw", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--n", type=int, default=30000)
    s.add_argument("--severity", type=int, default=1, choices=[1, 2, 3, 4, 5])
    s.add_argument("--seed", type=int, default=0)
    s = sub.add_parser("render")
    s.add_argument("--raw", default="/gscratch/dsikdar/caad_data/raw")
    s.add_argument("--out", required=True)
    s.add_argument("--workers", type=int, default=os.cpu_count())
    s.add_argument("--shard", type=int, default=0)
    s.add_argument("--num-shards", type=int, default=1)
    a = ap.parse_args()
    {"assign": cmd_assign, "render": cmd_render}[a.cmd](a)


if __name__ == "__main__":
    main()
