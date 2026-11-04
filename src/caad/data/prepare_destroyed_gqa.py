"""
caad/data/prepare_destroyed_gqa.py
==================================
"Unanswerable" training questions made from GQA: a GQA train question whose image
is destroyed by an extreme corruption (severity far beyond the 1-5 scale; the
engine scales every effect linearly with severity/5), labelled "unanswerable".
Images come from GQA train and share no question or image with our training
sets or the steering calibration pool.

  candidates  render every style at the given severities on a few images
              -> <out>/candidates/{images,manifest.jsonl}, a contact sheet, and
              how much of the image survives (correlation with the original)
  build       N questions (one per image), destroyed with the chosen style/severity
              cells (cycled), plus the training mixes:
                <out>/unanswerable/manifest.jsonl
                <mix_out>/manifest.jsonl = 30k severity-4 GQA + these, every question
                carrying the VizWiz note; on the destroyed items clean_image ==
                corrupted_image, so a distillation teacher sees the destroyed image too.

    python -m caad.data.prepare_destroyed_gqa candidates --out /gscratch/dsikdar/caad_data/destroyed_gqa
    python -m caad.data.prepare_destroyed_gqa build --out ... --cells gaussian_noise:100 night:50 --n 3334
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np

from .corruption import CORRUPTION_STYLES, apply_corruption, make_spec
from .prepare_steer_calib import RAW, save_img, seen_gqa

BASE_MANIFEST = "/gscratch/dsikdar/caad_data/gqa_train_30k_s4/manifest.jsonl"
CALIB_GQA = "/gscratch/dsikdar/caad_data/steer_calib/gqa/manifest.jsonl"


def fresh_pool(seed=0):
    """GQA train questions on images used nowhere else; one question per image."""
    import glob
    import pyarrow.parquet as pq
    seen_q, seen_img = seen_gqa()
    if Path(CALIB_GQA).exists():
        for r in map(json.loads, open(CALIB_GQA)):
            seen_q.add(r["question_id"]); seen_img.add(str(r["image_id"]))
    qs = []
    for f in sorted(glob.glob(f"{RAW}/GQA/train_balanced_instructions/*.parquet")):
        qs += pq.read_table(f, columns=["id", "imageId", "question", "answer"]).to_pylist()
    pool = sorted((q for q in qs if q["id"] not in seen_q and str(q["imageId"]) not in seen_img),
                  key=lambda r: r["id"])
    by_img = {}
    for q in random.Random(seed).sample(pool, len(pool)):
        by_img.setdefault(str(q["imageId"]), q)
    return list(by_img.values())


def destroy(arr, image_key, style, severity):
    return apply_corruption(arr, make_spec(image_key, style, severity, 0))


def survival(a, b):
    """Pearson correlation of 64x64 grayscale thumbnails (1 = intact, ~0 = destroyed)."""
    import cv2
    g = lambda x: cv2.resize(cv2.cvtColor(x, cv2.COLOR_RGB2GRAY), (64, 64), interpolation=cv2.INTER_AREA).ravel()
    x, y = g(a).astype(np.float64), g(b).astype(np.float64)
    if x.std() < 1e-6 or y.std() < 1e-6:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def cmd_candidates(a):
    from PIL import Image
    from .prepare_gqa_train import _image_bytes
    pool = fresh_pool()[: a.n_images]
    imgs = _image_bytes(RAW, {str(q["imageId"]) for q in pool})
    d = Path(a.out) / "candidates"
    rows, stats = [], {}
    for q in pool:
        arr = save_img(imgs[str(q["imageId"])], d / "clean" / f"{q['imageId']}.png")
        for style in CORRUPTION_STYLES:
            for sev in a.severities:
                key = f"{style}:{sev}"
                try:
                    out = destroy(arr, f"gqa_train:{q['id']}", style, sev)
                except Exception as err:                       # e.g. JPEG quality < 0 at sev >> 5
                    stats.setdefault(key, {"error": f"{type(err).__name__}: {str(err)[:80]}"})
                    continue
                p = d / "images" / f"{q['id']}__{style}__s{sev}.png"
                p.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(out).save(p)
                stats.setdefault(key, {"corr": []})["corr"].append(survival(arr, out))
                rows.append({"question_id": q["id"], "image_id": str(q["imageId"]), "question": q["question"],
                             "answer": q["answer"], "style": style, "severity": sev, "image": str(p)})
    with (d / "manifest.jsonl").open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    summ = {k: ({"mean_corr": float(np.mean(v["corr"])), "max_corr": float(np.max(v["corr"]))} if "corr" in v else v)
            for k, v in stats.items()}
    (d / "survival.json").write_text(json.dumps(summ, indent=2))
    # contact sheet: one image, every style x severity
    q0 = pool[0]
    tiles = [np.asarray(Image.open(d / "clean" / f"{q0['imageId']}.png").convert("RGB").resize((160, 120)))]
    labels = ["clean"]
    for style in CORRUPTION_STYLES:
        for sev in a.severities:
            p = d / "images" / f"{q0['id']}__{style}__s{sev}.png"
            tiles.append(np.asarray(Image.open(p).convert("RGB").resize((160, 120))) if p.exists()
                         else np.zeros((120, 160, 3), np.uint8))
            labels.append(f"{style}:{sev}")
    cols = 1 + len(a.severities)
    import cv2
    grid = []
    for i in range(0, len(tiles), cols):
        row = [cv2.putText(t.copy(), lab[:22], (2, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 0, 0), 1)
               for t, lab in zip(tiles[i:i + cols], labels[i:i + cols])]
        row += [np.zeros((120, 160, 3), np.uint8)] * (cols - len(row))
        grid.append(np.concatenate(row, 1))
    Image.fromarray(np.concatenate(grid, 0)).save(d / "contact_sheet.png")
    for k in sorted(summ):
        print(k, summ[k])


def cmd_build(a):
    from PIL import Image
    from .prepare_gqa_train import _image_bytes
    from ..eval.tasks.vqa_c import VIZWIZ_NOTE
    cells = [(c.split(":")[0], int(c.split(":")[1])) for c in a.cells]
    pool = fresh_pool()[a.skip: a.skip + a.n]                  # skip the candidate images
    imgs = _image_bytes(RAW, {str(q["imageId"]) for q in pool})
    d = Path(a.out) / "unanswerable"
    rows = []
    for i, q in enumerate(pool):
        style, sev = cells[i % len(cells)]
        arr = np.asarray(Image.open(__import__("io").BytesIO(imgs[str(q["imageId"])])).convert("RGB"))
        out = destroy(arr, f"gqa_train:{q['id']}", style, sev)
        p = d / "images" / f"{q['id']}__{style}__s{sev}.png"
        p.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(out).save(p)
        rows.append({"image_id": str(q["imageId"]), "question_id": q["id"], "question": q["question"],
                     "answer": "unanswerable", "original_answer": q["answer"],
                     "clean_image": str(p), "corrupted_image": str(p),     # teacher sees the destroyed image too
                     "corruption": {"style": f"destroyed_{style}", "severity": sev, "key": f"gqa_train:{q['id']}"},
                     "source": "gqa_destroyed_unanswerable"})
    with (d / "manifest.jsonl").open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    base = [json.loads(l) for l in open(BASE_MANIFEST)]
    note = lambda r: {**r, "question": r["question"].rstrip() + VIZWIZ_NOTE}
    mix = [note(r) for r in base + rows]
    random.Random(1).shuffle(mix)
    mo = Path(a.mix_out)
    mo.mkdir(parents=True, exist_ok=True)
    with (mo / "manifest.jsonl").open("w") as fh:
        for r in mix:
            fh.write(json.dumps(r) + "\n")
    print(f"{len(rows)} destroyed + {len(base)} GQA = {len(mix)} ({100 * len(rows) / len(mix):.1f}% unanswerable)"
          f" -> {mo / 'manifest.jsonl'}; cells {cells}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("candidates")
    c.add_argument("--out", required=True)
    c.add_argument("--n-images", type=int, default=24)
    c.add_argument("--severities", type=int, nargs="+", default=[50, 100])
    b = sub.add_parser("build")
    b.add_argument("--out", required=True)
    b.add_argument("--mix-out", required=True)
    b.add_argument("--cells", nargs="+", required=True, help="style:severity, cycled over the questions")
    b.add_argument("--n", type=int, default=3334)
    b.add_argument("--skip", type=int, default=24, help="skip the images used for the candidate check")
    a = ap.parse_args()
    {"candidates": cmd_candidates, "build": cmd_build}[a.cmd](a)


if __name__ == "__main__":
    main()
