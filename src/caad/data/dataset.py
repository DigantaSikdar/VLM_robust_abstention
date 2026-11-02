"""
caad/data/dataset.py
====================
Manifest-backed dataset for CAAD training.

A manifest is a JSONL file, one sample per row (written by
caad.data.prepare_images):

    {"image_id": "...", "question": "...", "answer": "B",
     "clean_image": "data/clean/abc.png",
     "corrupted_image": "data/corrupted/abc__fog__s3.png",
     "corruption": {"style": "fog", "severity": 3, "seed": 12345}}

The trainer only reads ``question`` / ``clean_image`` / ``corrupted_image`` (it
hands the paths straight to qwen-vl-utils). The ``corruption`` block records the
exact CorruptionSpec so a corrupted image can be re-rendered bit-for-bit from the
clean source — see ``render_corrupted``.

Corruptions are rendered to disk ahead of training (not on the fly) because the
cross-input alignment invariant requires the corrupted image to decode to the
same resolution as the clean one; pre-rendering makes that checkable once,
offline.
"""

from __future__ import annotations

import json
from pathlib import Path

from torch.utils.data import Dataset

from .corruption import CorruptionSpec, apply_corruption, make_spec

MEDIA_KEYS = ("clean_image", "corrupted_image")


class ManifestDataset(Dataset):
    """Yields raw manifest rows; collation/tokenization happens in the trainer.
    Relative image paths are resolved against ``image_root``."""

    def __init__(self, manifest_path, *, image_root=None):
        self.rows = [json.loads(l) for l in Path(manifest_path).read_text().splitlines() if l.strip()]
        self.image_root = Path(image_root) if image_root else None

    def _abspath(self, p):
        p = Path(p)
        return str(self.image_root / p) if self.image_root and not p.is_absolute() else str(p)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = dict(self.rows[i])
        for k in MEDIA_KEYS:
            r[k] = self._abspath(r[k])
        return r


def render_corrupted(image, *, image_id, style, severity, global_seed):
    """
    Apply a deterministic corruption to a clean (H, W, 3) uint8 image and return
    (corrupted_image, spec). Persist the spec in the manifest so the image can be
    reproduced from the clean source on any machine.
    """
    spec = make_spec(image_id, style, severity, global_seed)
    return apply_corruption(image, spec), spec


def spec_from_manifest_row(row) -> CorruptionSpec:
    """Rebuild the CorruptionSpec recorded in a manifest row."""
    c = row["corruption"]
    return CorruptionSpec(
        image_id=row["image_id"], style=c["style"], severity=int(c["severity"]),
        seed=int(c["seed"]), extra=c.get("extra", {}),
    )
