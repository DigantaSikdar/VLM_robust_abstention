"""
caad/data/coldstart.py
======================
Off-policy cold start before on-policy distillation (Li et al. 2026, Sec. 5.1):
the frozen teacher answers fresh GQA train questions on the CLEAN image, and the
student is fine-tuned on those answers, also on the CLEAN image, so the cold
start only closes the teacher/student style gap and teaches no robustness.

Prompts are disjoint from the distillation set: no shared question AND no shared
image with --exclude, so the later on-policy stage sees unseen images.

    python -m caad.data.coldstart assign --exclude .../gqa_train_30k_s4/manifest.jsonl \
        --out /gscratch/dsikdar/caad_data/coldstart_gqa30k --n 30000
    python -m caad.data.coldstart render --out ...          # clean PNGs (caad_env)
    python -m caad.data.coldstart generate --out ... --model Qwen/Qwen3-VL-8B-Instruct   # vllm_env
    python -m caad.data.coldstart filter --out ...

Rows of sft.jsonl (read by caad.sft): the manifest row plus "completion".
Filtering follows the paper (drop truncated and degenerate outputs) plus our
answer format (an <answer> tag with a non-empty answer). Correctness against the
gold answer is NOT used, so the cold start stays label-free.
"""

from __future__ import annotations

import argparse
import glob
import io
import json
import os
import re
import time
from collections import Counter
from pathlib import Path

import numpy as np

from .corruption import derive_seed


def cmd_assign(a):
    import pyarrow.parquet as pq
    ex_q, ex_img = set(), set()
    for m in a.exclude or []:
        for l in open(m):
            r = json.loads(l)
            ex_q.add(r["question_id"])
            ex_img.add(r["image_id"])
    files = sorted(glob.glob(str(Path(a.raw) / "GQA" / "train_balanced_instructions" / "*.parquet")))
    qs = []
    for f in files:
        qs += pq.read_table(f, columns=["id", "imageId", "question", "answer", "fullAnswer"]).to_pylist()
    pool = sorted((q for q in qs if q["id"] not in ex_q and q["imageId"] not in ex_img),
                  key=lambda r: r["id"])
    rng = np.random.default_rng(derive_seed("gqa_coldstart", "sample", 0, a.seed))
    pick = [pool[i] for i in sorted(rng.choice(len(pool), size=a.n, replace=False))]
    out = Path(a.out)
    (out / "clean").mkdir(parents=True, exist_ok=True)
    with (out / "manifest.jsonl").open("w") as fh:
        for q in pick:
            img = str(out / "clean" / f"{q['imageId']}.png")
            fh.write(json.dumps({"image_id": q["imageId"], "question_id": q["id"], "question": q["question"],
                                 "answer": q["answer"], "full_answer": q["fullAnswer"],
                                 "clean_image": img, "corrupted_image": img, "corruption": None}) + "\n")
    summary = {"source": "GQA train_balanced", "pool_after_exclusion": len(pool), "n": a.n, "seed": a.seed,
               "excluded_questions": len(ex_q), "excluded_images": len(ex_img),
               "distinct_images": len({q["imageId"] for q in pick})}
    (out / "assignment.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary))


def cmd_render(a):
    from PIL import Image
    from .prepare_gqa_train import _image_bytes
    out = Path(a.out)
    rows = [json.loads(l) for l in open(out / "manifest.jsonl")]
    need = {r["image_id"] for r in rows if not Path(r["clean_image"]).exists()}
    print(f"{len(need)} images to decode", flush=True)
    if not need:
        return
    for iid, b in _image_bytes(a.raw, need).items():
        dst = out / "clean" / f"{iid}.png"
        tmp = dst.with_name(dst.name + ".tmp")
        Image.open(io.BytesIO(b)).convert("RGB").save(tmp, format="PNG")
        os.replace(tmp, dst)
    print("rendered", len(need), flush=True)


def cmd_generate(a):
    from PIL import Image
    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams
    from ..eval.tasks.vqa_c import PROMPT_FORMATS

    out = Path(a.out)
    rows = [json.loads(l) for l in open(out / "manifest.jsonl")]
    res = out / f"rollouts_{Path(a.model).name}{_suffix(a.prompt_format)}.jsonl"
    done = {json.loads(l)["question_id"] for l in open(res)} if res.exists() else set()
    todo = [r for r in rows if r["question_id"] not in done]
    print(f"[start] {len(rows)} prompts, {len(done)} done, {len(todo)} to go", flush=True)
    system, user_tpl = PROMPT_FORMATS[a.prompt_format]
    proc = AutoProcessor.from_pretrained(a.model)
    llm = LLM(model=a.model, dtype="bfloat16", max_model_len=a.max_model_len,
              limit_mm_per_prompt={"image": 1}, gpu_memory_utilization=0.9, seed=a.seed,
              tensor_parallel_size=a.tp)
    # the paper's offline-rollout sampling (temperature 0.7, top-p 0.95)
    sp = SamplingParams(temperature=a.temperature, top_p=a.top_p, max_tokens=a.max_tokens, seed=a.seed)

    def prompt(r):
        msgs = [{"role": "user", "content": [{"type": "image"},
                                             {"type": "text", "text": user_tpl.format(question=r["question"])}]}]
        if system:
            msgs.insert(0, {"role": "system", "content": [{"type": "text", "text": system}]})
        return proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    t0 = time.time()
    with res.open("a") as fh:
        for i in range(0, len(todo), a.chunk):
            chunk = todo[i:i + a.chunk]
            inputs = []
            for r in chunk:
                with Image.open(r["clean_image"]) as im:
                    inputs.append({"prompt": prompt(r), "multi_modal_data": {"image": im.convert("RGB")}})
            for r, o in zip(chunk, llm.generate(inputs, sp, use_tqdm=False)):
                g = o.outputs[0]
                fh.write(json.dumps({"question_id": r["question_id"], "completion": g.text,
                                     "finish_reason": g.finish_reason, "gen_tokens": len(g.token_ids)}) + "\n")
            fh.flush()
            n = i + len(chunk)
            print(f"[progress] {n}/{len(todo)} ({n / (time.time() - t0):.1f}/s)", flush=True)


def _suffix(prompt_format):
    """File-name suffix: none for the original inline prompt, else the format name."""
    return "" if prompt_format == "inline" else f"_{prompt_format}"


def degenerate(text, n=8, max_repeat=4):
    """True if some word n-gram repeats more than max_repeat times (looping output)."""
    w = text.split()
    if len(w) < n * max_repeat:
        return False
    c = Counter(tuple(w[i:i + n]) for i in range(len(w) - n + 1))
    return max(c.values()) > max_repeat


def cmd_filter(a):
    from ..eval.tasks.vqa_c import GQA, split_reasoning
    out = Path(a.out)
    rows = {json.loads(l)["question_id"]: json.loads(l) for l in open(out / "manifest.jsonl")}
    sfx = _suffix(a.prompt_format)
    res = out / f"rollouts_{Path(a.model).name}{sfx}.jsonl"
    why, kept, correct, words = Counter(), [], 0, []
    for l in open(res):
        g = json.loads(l)
        r = rows[g["question_id"]]
        _, ans, method = split_reasoning(g["completion"])
        if g["finish_reason"] != "stop":
            why["truncated"] += 1
        elif method != "tag" or not ans.strip():
            why["no_answer_tag"] += 1
        elif degenerate(g["completion"]):
            why["degenerate"] += 1
        else:
            kept.append({**r, "completion": g["completion"]})
            correct += GQA.score({"gold": r["answer"]}, g["completion"])   # reported only, not filtered on
            words.append(len(ans.split()))
            continue
    with (out / f"sft{sfx}.jsonl").open("w") as fh:
        for r in kept:
            fh.write(json.dumps(r) + "\n")
    stats = {"rollouts": sum(why.values()) + len(kept), "kept": len(kept), "dropped": dict(why),
             "kept_gqa_exact_match": correct / max(len(kept), 1),
             "kept_mean_answer_words": float(np.mean(words)) if words else 0.0}
    (out / f"filter_stats{sfx}.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


def main():
    ap = argparse.ArgumentParser(description="Teacher-rollout cold-start data (clean images)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("assign")
    s.add_argument("--raw", default="/gscratch/dsikdar/caad_data/raw")
    s.add_argument("--out", required=True)
    s.add_argument("--exclude", nargs="*", help="manifests whose questions AND images are excluded")
    s.add_argument("--n", type=int, default=30000)
    s.add_argument("--seed", type=int, default=0)
    s = sub.add_parser("render")
    s.add_argument("--raw", default="/gscratch/dsikdar/caad_data/raw")
    s.add_argument("--out", required=True)
    s = sub.add_parser("generate")
    s.add_argument("--out", required=True)
    s.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    s.add_argument("--prompt-format", default="inline")
    s.add_argument("--temperature", type=float, default=0.7)
    s.add_argument("--top-p", type=float, default=0.95)
    s.add_argument("--max-tokens", type=int, default=768)
    s.add_argument("--max-model-len", type=int, default=4096)   # GQA images are <=640px: ~1k-token prompts
    s.add_argument("--tp", type=int, default=1, help="tensor-parallel GPUs")
    s.add_argument("--chunk", type=int, default=1000)
    s.add_argument("--seed", type=int, default=0)
    s = sub.add_parser("filter")
    s.add_argument("--out", required=True)
    s.add_argument("--prompt-format", default="inline")
    s.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    a = ap.parse_args()
    {"assign": cmd_assign, "render": cmd_render, "generate": cmd_generate, "filter": cmd_filter}[a.cmd](a)


if __name__ == "__main__":
    main()
