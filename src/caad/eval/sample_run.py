"""
caad/eval/sample_run.py
=======================
Small check run on TextVQA-C / GQA-C with a Hugging Face VLM: picks a few
questions per dataset (spread across corruption types), answers each on its
clean AND its assigned corrupted image, and records the full completion, the
reasoning, the extracted answer, which extraction rule fired, and the score.
Use it to eyeball answer extraction before a full run.

    python -m caad.eval.sample_run --root /gscratch/dsikdar/caad_data/vqa_c \
        --model Qwen/Qwen3-VL-4B-Instruct --n 12 --out outputs/check/qwen3vl4b.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

from ..data.prepare_vqa_c import _render_one
from .tasks.registry import get_task
from .tasks.vqa_c import format_check, normalize_answer, split_reasoning

FIGURE3 = {"textvqa": "what is the year on the calender?", "gqa": "Is it overcast?"}


def pick(examples, n, seed, dataset):
    """Figure-3 question first, then one question per corruption type
    (round-robin over categories), then further random questions until ``n``.
    Deterministic in ``seed``."""
    rng = random.Random(seed)
    first = [e for e in examples if e["question"] == FIGURE3[dataset]][:1]
    by_name = {}
    for e in rng.sample(examples, len(examples)):
        by_name.setdefault(e["corruption"]["name"], e)
    cats = {}
    for name, e in by_name.items():
        cats.setdefault(e["corruption"]["category"], []).append(e)
    rr = []
    while any(cats.values()):
        for c in sorted(cats):
            if cats[c]:
                rr.append(cats[c].pop(0))
    chosen = first + [e for e in rr if e["id"] not in {f["id"] for f in first}]
    seen = {e["id"] for e in chosen}
    chosen += [e for e in rng.sample(examples, len(examples)) if e["id"] not in seen]
    return chosen[:n]


class HFSampler:
    def __init__(self, model_name, max_new_tokens=768):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor
        self.torch = torch
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_name, dtype=torch.bfloat16, device_map="cuda").eval()
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.max_new_tokens = max_new_tokens

    def __call__(self, image_path, prompt, system=None):
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image_path}, {"type": "text", "text": prompt}]}]
        if system:
            messages.insert(0, {"role": "system", "content": [{"type": "text", "text": system}]})
        inputs = self.processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt").to(self.model.device)
        with self.torch.no_grad():
            out = self.model.generate(**inputs, do_sample=False,
                                      max_new_tokens=self.max_new_tokens)
        gen = out[0, inputs["input_ids"].shape[1]:]
        return self.processor.decode(gen, skip_special_tokens=True), int(gen.shape[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--model", default="Qwen/Qwen3-VL-4B-Instruct")
    ap.add_argument("--n", type=int, default=12, help="questions per dataset")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-new-tokens", type=int, default=768)
    ap.add_argument("--prompt-format", default="inline", choices=["inline", "think_answer"])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    manifest_rows = {}
    for ds in ("textvqa", "gqa"):
        for l in (Path(a.root) / ds / "manifest.jsonl").read_text().splitlines():
            r = json.loads(l)
            manifest_rows[(ds, r["question_id"])] = r

    sampler = HFSampler(a.model, a.max_new_tokens)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as fh:
        for ds in ("textvqa", "gqa"):
            task = get_task(ds)
            cfg = {"root": a.root, "prompt_format": a.prompt_format}
            clean = task.load({**cfg, "use_corrupted": False})
            corr = {e["id"]: e for e in task.load({**cfg, "use_corrupted": True})}
            for ex in pick(clean, a.n, a.seed, ds):
                _render_one(manifest_rows[(ds, ex["id"])])     # ensure corrupted PNG exists
                for view_ex in (ex, corr[ex["id"]]):
                    t0 = time.time()
                    completion, n_tok = sampler(view_ex["image"], view_ex["prompt"],
                                                view_ex["system"])
                    reasoning, answer, method = split_reasoning(completion)
                    rec = {"dataset": ds, "question_id": ex["id"], "image_id": ex["image_id"],
                           "view": view_ex["view"], "image": view_ex["image"],
                           "question": ex["question"], "prompt": view_ex["prompt"],
                           "system": view_ex["system"], "prompt_format": a.prompt_format,
                           "format": format_check(completion),
                           "gold": ex["gold"], "answers": ex.get("answers"),
                           "full_answer": ex.get("full_answer"),
                           "corruption": ex["corruption"],
                           "completion": completion, "reasoning": reasoning,
                           "extracted": answer, "normalized": normalize_answer(answer),
                           "extraction": method, "correct": task.score(view_ex, completion),
                           "gen_tokens": n_tok, "truncated": n_tok >= a.max_new_tokens,
                           "seconds": round(time.time() - t0, 2),
                           "model": a.model}
                    if ds == "textvqa":
                        rec["soft"] = task.soft_score(view_ex, completion)
                    fh.write(json.dumps(rec) + "\n")
                    fh.flush()
                    print(f"[{ds} {view_ex['view']:9s} {ex['corruption']['name']}@"
                          f"{ex['corruption']['severity']}] {ex['question'][:50]!r} -> "
                          f"{answer!r} ({method}) gold={ex['gold']!r} "
                          f"{'OK' if rec['correct'] else 'X'}", flush=True)
    print("wrote", out)


if __name__ == "__main__":
    main()
