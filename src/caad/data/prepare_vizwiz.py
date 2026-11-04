"""
caad/data/prepare_vizwiz.py
===========================
VizWiz-VQA validation (4,319 questions) for real-world robustness evaluation:
photos taken by blind / low-vision people, often blurry, dark, overexposed or
badly framed. There is no clean counterpart, so each question is evaluated on
its real photo only (view "real").

    python -m caad.data.prepare_vizwiz --raw /gscratch/dsikdar/caad_data/raw/VizWiz \
        --out /gscratch/dsikdar/caad_data/vqa_c/vizwiz
"""
from __future__ import annotations

import argparse
import glob
import io
import json
from pathlib import Path


def main():
    import pyarrow.parquet as pq
    from PIL import Image
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out); (out / "images").mkdir(parents=True, exist_ok=True)
    n = 0
    with (out / "manifest.jsonl").open("w") as fh:
        for f in sorted(glob.glob(f"{a.raw}/data/val-*.parquet")):
            for r in pq.read_table(f).to_pylist():
                b = r["image"]["bytes"]
                ext = {"jpeg": "jpg", "mpo": "jpg"}.get((Image.open(io.BytesIO(b)).format or "png").lower(), "png")
                p = out / "images" / f"{r['question_id']}.{ext}"
                if not p.exists():
                    p.write_bytes(b)                       # original bytes, untouched
                fh.write(json.dumps({"dataset": "vizwiz", "question_id": r["question_id"],
                                     "image_id": r["question_id"], "question": r["question"],
                                     "answers": list(r["answers"]), "category": r["category"],
                                     "image": str(p)}) + "\n")
                n += 1
    print(f"wrote {n} rows -> {out / 'manifest.jsonl'}")


if __name__ == "__main__":
    main()
