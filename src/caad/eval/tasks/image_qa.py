"""
caad/eval/tasks/image_qa.py
===========================
Image benchmarks over a CAAD image manifest (rows from
caad.data.prepare_images): ``image_mcq`` scores a single option letter (A-E),
``image_qa`` scores a normalized exact-match open answer.

Set use_corrupted: true in the eval config to score on the corrupted image (the
robustness axis CAAD targets) instead of the clean one.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from .registry import register


def _rows(cfg):
    rows = [json.loads(l) for l in Path(cfg["manifest"]).read_text().splitlines() if l.strip()]
    key = "corrupted_image" if cfg.get("use_corrupted") else "clean_image"
    return [r for r in rows if r.get("answer") is not None], key


def _example(r, key, gold):
    return {"id": r["image_id"], "question": r["question"], "image": r[key], "gold": gold}


@register("image_mcq")
class ImageMCQ:
    @staticmethod
    def load(cfg):
        rows, key = _rows(cfg)
        return [_example(r, key, str(r["answer"]).strip().upper()[:1]) for r in rows]

    @staticmethod
    def score(example, completion) -> bool:
        pred = extract_letter(completion)
        return bool(pred) and pred == example["gold"]


@register("image_qa")
class ImageQA:
    @staticmethod
    def load(cfg):
        rows, key = _rows(cfg)
        return [_example(r, key, _norm(str(r["answer"]))) for r in rows]

    @staticmethod
    def score(example, completion) -> bool:
        return _norm(completion) == example["gold"]


def extract_letter(text: str) -> str:
    """Best-effort MCQ letter extraction from a free-form completion."""
    # 1) explicit "answer is X" / "answer: X" / "the answer (X)"
    m = re.search(r"answer\s*(?:is|:)?\s*\(?\s*([A-E])\b", text, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    # 2) a boxed answer \boxed{X}
    m = re.search(r"\\boxed\{\s*([A-E])\s*\}", text, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    # 3) last standalone capital letter A-E (final answers tend to come last)
    cands = re.findall(r"\b([A-E])\b", text)
    return cands[-1].upper() if cands else ""


def _norm(s: str) -> str:
    """Lowercase, strip punctuation/whitespace — shared so load() and score()
    normalize identically (mismatched normalization silently tanks accuracy)."""
    return re.sub(r"[^a-z0-9 ]+", "", s.lower()).strip()
