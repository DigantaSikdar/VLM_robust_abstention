"""
caad/data/prepare_steer_calib.py
================================
Calibration pools for activation steering, built ONLY from training splits that
no model saw and that share no question or image with any evaluation set:

  vizwiz   VizWiz train (Multimodal-Fatima/VizWiz_train)    real photos, category from answer_type
  gqa      GQA train-balanced, minus every question AND image used for distillation
           or the cold start                                  corrupted with ImageNet-C
  vqav2    VQAv2 train (COCO train2014; eval uses val2014), minus COCO images that
           appear among our GQA training images (Visual Genome includes COCO)
                                                               corrupted with ImageNet-C
  textvqa  TextVQA train (eval uses val)                       clean only (no-image pairs)

The layout mirrors caad_data/vqa_c (<root>/<dataset>/manifest.jsonl, same row
schema), so caad.eval.vllm_eval can generate responses with --root pointed here.
Corruptions: one ImageNet-C type and severity per question, balanced over the 95
type x severity cells, seeded per question.

    python -m caad.data.prepare_steer_calib --out /gscratch/dsikdar/caad_data/steer_calib
"""

from __future__ import annotations

import argparse
import glob
import io
import json
import os
import random
from collections import Counter
from pathlib import Path

import numpy as np

from .corruption import derive_seed
from .imagenet_c import CATEGORIES, corrupt_image
from .prepare_vqa_c import CAAD_COUNTERPART

RAW = "/gscratch/dsikdar/caad_data/raw"
TRAIN = f"{RAW}/train_splits"
SEEN_MANIFESTS = ["/gscratch/dsikdar/caad_data/gqa_train_30k/manifest.jsonl",
                  "/gscratch/dsikdar/caad_data/gqa_train_30k_s4/manifest.jsonl",
                  "/gscratch/dsikdar/caad_data/coldstart_gqa30k/manifest.jsonl"]
CELLS = [(name, cat, sev) for cat, names in CATEGORIES.items() for name in names for sev in (1, 2, 3, 4, 5)]


def save_img(img_bytes, path, maxside=None):
    from PIL import Image
    im = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    path.parent.mkdir(parents=True, exist_ok=True)
    im.save(path, quality=95) if path.suffix == ".jpg" else im.save(path)
    return np.asarray(im)


def corruption_for(key, rng_cells, i):
    name, cat, sev = rng_cells[i % len(rng_cells)]
    seed = derive_seed(key, name, sev, 0) % (2 ** 32)
    return {"name": name, "category": cat, "severity": sev, "seed": seed, "key": key, "global_seed": 0,
            "seen_in_training": name in CAAD_COUNTERPART, "caad_counterpart": CAAD_COUNTERPART.get(name)}


def write_corrupted(arr, corr, path):
    from PIL import Image
    out = corrupt_image(arr, corr["name"], corr["severity"], image_id=corr["key"])
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(out).save(path)


def seen_gqa():
    q, img = set(), set()
    for m in SEEN_MANIFESTS:
        if Path(m).exists():
            for l in open(m):
                r = json.loads(l)
                q.add(r["question_id"])
                img.add(str(r["image_id"]))
    return q, img


def build_vizwiz(out, n_unans, n_ans, rng):
    import pyarrow.parquet as pq
    rows = []
    for f in sorted(glob.glob(f"{TRAIN}/Multimodal-Fatima__VizWiz_train/data/*.parquet")):
        rows += pq.read_table(f, columns=["id", "image", "question", "answers", "answer_type"]).to_pylist()
    un = [r for r in rows if r["answer_type"] == "unanswerable"]
    an = [r for r in rows if r["answer_type"] != "unanswerable"]
    pick = rng.sample(un, min(n_unans, len(un))) + rng.sample(an, min(n_ans, len(an)))
    d = Path(out) / "vizwiz"
    d.mkdir(parents=True, exist_ok=True)
    with (d / "manifest.jsonl").open("w") as fh:
        for r in pick:
            qid = f"VizWiz_train_{r['id']}"
            p = d / "images" / f"{qid}.jpg"
            save_img(r["image"]["bytes"], p)
            fh.write(json.dumps({"dataset": "vizwiz", "question_id": qid, "image_id": qid, "question": r["question"],
                                 "answers": list(r["answers"]), "category": r["answer_type"],
                                 "image": str(p)}) + "\n")
    return {"pool": len(rows), "unanswerable": sum(r["answer_type"] == "unanswerable" for r in pick),
            "answerable": sum(r["answer_type"] != "unanswerable" for r in pick)}


def build_gqa(out, n, rng):
    import pyarrow.parquet as pq
    from .prepare_gqa_train import _image_bytes
    seen_q, seen_img = seen_gqa()
    qs = []
    for f in sorted(glob.glob(f"{RAW}/GQA/train_balanced_instructions/*.parquet")):
        qs += pq.read_table(f, columns=["id", "imageId", "question", "answer", "fullAnswer"]).to_pylist()
    pool = [q for q in qs if q["id"] not in seen_q and q["imageId"] not in seen_img]
    pool.sort(key=lambda r: r["id"])
    by_img = {}
    for q in rng.sample(pool, len(pool)):                 # one question per image -> n distinct images
        by_img.setdefault(q["imageId"], q)
        if len(by_img) >= n:
            break
    pick = list(by_img.values())
    imgs = _image_bytes(RAW, {q["imageId"] for q in pick})
    cells = rng.sample(CELLS, len(CELLS))
    d = Path(out) / "gqa"
    d.mkdir(parents=True, exist_ok=True)
    with (d / "manifest.jsonl").open("w") as fh:
        for i, q in enumerate(pick):
            clean = d / "clean" / f"{q['imageId']}.png"
            arr = save_img(imgs[q["imageId"]], clean)
            corr = corruption_for(f"gqa_train:{q['id']}", cells, i)
            cpath = d / "corrupted" / f"{q['id']}__{corr['name']}__s{corr['severity']}.png"
            write_corrupted(arr, corr, cpath)
            fh.write(json.dumps({"dataset": "gqa", "question_id": q["id"], "image_id": q["imageId"],
                                 "question": q["question"], "answer": q["answer"], "full_answer": q["fullAnswer"],
                                 "clean_image": str(clean), "corrupted_image": str(cpath), "corruption": corr}) + "\n")
    return {"pool": len(pool), "picked": len(pick), "excluded_questions": len(seen_q), "excluded_images": len(seen_img)}


def coco_ids_in_our_gqa_training():
    """COCO ids of the Visual Genome images used for distillation / cold start."""
    _, seen_img = seen_gqa()
    vg = json.load(open(f"{RAW}/VG/image_data.json"))
    return {str(r["coco_id"]) for r in vg if r.get("coco_id") and str(r["image_id"]) in seen_img}


def build_vqav2(out, n, rng):
    import pyarrow.parquet as pq
    bad = coco_ids_in_our_gqa_training()
    rows = []
    for f in sorted(glob.glob(f"{TRAIN}/lmms-lab__VQAv2/data/*.parquet")):
        rows += pq.read_table(f, columns=["question_id", "image_id", "question", "answers", "answer_type",
                                          "image"]).to_pylist()
    pool = [r for r in rows if str(r["image_id"]) not in bad]
    by_img = {}
    for r in rng.sample(pool, len(pool)):
        by_img.setdefault(r["image_id"], r)
        if len(by_img) >= n:
            break
    pick = list(by_img.values())
    cells = rng.sample(CELLS, len(CELLS))
    d = Path(out) / "vqav2"
    d.mkdir(parents=True, exist_ok=True)
    with (d / "manifest.jsonl").open("w") as fh:
        for i, r in enumerate(pick):
            clean = d / "clean" / f"{r['image_id']}.jpg"
            arr = save_img(r["image"]["bytes"], clean)
            corr = corruption_for(f"vqav2_train:{r['question_id']}", cells, i)
            cpath = d / "corrupted" / f"{r['question_id']}__{corr['name']}__s{corr['severity']}.png"
            write_corrupted(arr, corr, cpath)
            fh.write(json.dumps({"dataset": "vqav2", "question_id": str(r["question_id"]), "image_id": str(r["image_id"]),
                                 "question": r["question"], "answers": [a["answer"] for a in r["answers"]],
                                 "category": r["answer_type"], "clean_image": str(clean),
                                 "corrupted_image": str(cpath), "corruption": corr}) + "\n")
    return {"rows": len(rows), "excluded_coco_overlap": len(rows) - len(pool), "picked": len(pick)}


def build_textvqa(out, n, rng):
    import pyarrow.parquet as pq
    rows = []
    for f in sorted(glob.glob(f"{TRAIN}/lmms-lab__textvqa/data/*.parquet")):
        rows += pq.read_table(f, columns=["question_id", "image_id", "question", "answers", "image"]).to_pylist()
    by_img = {}
    for r in rng.sample(rows, len(rows)):
        by_img.setdefault(r["image_id"], r)
        if len(by_img) >= n:
            break
    d = Path(out) / "textvqa"
    d.mkdir(parents=True, exist_ok=True)
    with (d / "manifest.jsonl").open("w") as fh:
        for r in by_img.values():
            clean = d / "clean" / f"{r['image_id']}.jpg"
            save_img(r["image"]["bytes"], clean)
            fh.write(json.dumps({"dataset": "textvqa", "question_id": str(r["question_id"]), "image_id": r["image_id"],
                                 "question": r["question"], "answers": list(r["answers"]),
                                 "clean_image": str(clean), "corrupted_image": str(clean),   # clean only
                                 "corruption": None}) + "\n")
    return {"picked": len(by_img)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/gscratch/dsikdar/caad_data/steer_calib")
    ap.add_argument("--vizwiz-unans", type=int, default=1200)
    ap.add_argument("--vizwiz-ans", type=int, default=800)
    ap.add_argument("--n", type=int, default=800, help="questions (distinct images) per other source")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only", default="vizwiz,gqa,vqav2,textvqa")
    a = ap.parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    stats = {}
    for ds in a.only.split(","):
        rng = random.Random(f"{a.seed}:{ds}")
        stats[ds] = {"vizwiz": lambda: build_vizwiz(a.out, a.vizwiz_unans, a.vizwiz_ans, rng),
                     "gqa": lambda: build_gqa(a.out, a.n, rng), "vqav2": lambda: build_vqav2(a.out, a.n, rng),
                     "textvqa": lambda: build_textvqa(a.out, a.n, rng)}[ds]()
        print(ds, stats[ds], flush=True)
    (Path(a.out) / f"build_stats_{a.only.replace(',', '_')}.json").write_text(json.dumps(stats, indent=2, default=str))


if __name__ == "__main__":
    main()
