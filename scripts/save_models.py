#!/usr/bin/env python3
"""
Freeze the cold-start model pair as standalone, loadable checkpoints:
  * the 4B student after SFT: base Qwen3-VL-4B-Instruct with the SFT LoRA merged in
    (full weights + processor), plus a copy of the LoRA adapter itself
  * the 8B teacher: a plain copy of the Hugging Face snapshot used in every run
    (so it no longer depends on the HF cache)
Both load with AutoModelForImageTextToText.from_pretrained(<dir>) and vLLM.

    python scripts/save_models.py --adapter /gscratch/dsikdar/outputs/sft_coldstart_8b/checkpoints/final/student \
        --out /gscratch/dsikdar/models
"""
import argparse
import json
import shutil
from pathlib import Path

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--student-base", default="Qwen/Qwen3-VL-4B-Instruct")
    ap.add_argument("--teacher", default="Qwen/Qwen3-VL-8B-Instruct")
    ap.add_argument("--out", default="/gscratch/dsikdar/models")
    ap.add_argument("--recipe", default="configs/sft/coldstart_8b_qwen3vl4b.yaml")
    ap.add_argument("--student-name", default="qwen3vl4b_sft_coldstart_8b")
    ap.add_argument("--skip-teacher", action="store_true", help="the teacher copy already exists")
    a = ap.parse_args()
    from huggingface_hub import snapshot_download
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText, AutoProcessor

    out = Path(a.out)
    s_dir = out / a.student_name
    base = AutoModelForImageTextToText.from_pretrained(a.student_base, dtype=torch.bfloat16)
    merged = PeftModel.from_pretrained(base, a.adapter).merge_and_unload()
    merged.save_pretrained(s_dir, safe_serialization=True)
    AutoProcessor.from_pretrained(a.student_base).save_pretrained(s_dir)
    shutil.copytree(a.adapter, s_dir / "lora_adapter", dirs_exist_ok=True)
    (s_dir / "provenance.json").write_text(json.dumps(
        {"base": a.student_base, "adapter": str(Path(a.adapter).resolve()),
         "recipe": a.recipe, "merged_dtype": "bfloat16"}, indent=2))
    print("saved student ->", s_dir, flush=True)

    t_dir = out / Path(a.teacher).name
    if a.skip_teacher:
        t_dir = None
    if t_dir is not None:
        snap = snapshot_download(a.teacher, local_files_only=True)
        shutil.copytree(snap, t_dir, symlinks=False, dirs_exist_ok=True)   # resolve cache symlinks
        (t_dir / "provenance.json").write_text(json.dumps({"source": a.teacher, "snapshot": snap}, indent=2))
        print("saved teacher ->", t_dir, flush=True)

    for d in (s_dir, t_dir or out / Path(a.teacher).name):                                         # quick load check
        n = sum(p.numel() for p in AutoModelForImageTextToText.from_pretrained(d, dtype=torch.bfloat16).parameters())
        print(f"load ok: {d} ({n / 1e9:.2f}B params)", flush=True)


if __name__ == "__main__":
    main()
