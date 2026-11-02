"""
caad/eval/vllm_eval.py
======================
Full TextVQA-C / GQA-C evaluation with vLLM (offline, one GPU): every
question on its clean image and on its assigned corrupted image
(2 x 17,578 = 35,156 generations), greedy decoding, the inline
"reason step by step ... <answer></answer>" prompt.

Resumable: results are appended per chunk to <out>/results.jsonl and finished
(dataset, question, view) triples are skipped on restart. After every chunk
the log reports progress, ETA, the <answer>-tag extraction rate and running
accuracies; at the end <out>/summary.{json,md} hold the full breakdown.

    python -m caad.eval.vllm_eval run --root /gscratch/dsikdar/caad_data/vqa_c \
        --model Qwen/Qwen3-VL-4B-Instruct --out outputs/eval/Qwen3-VL-4B-Instruct
    # a trained LoRA adapter on top of the base model:
    python -m caad.eval.vllm_eval run ... --lora <checkpoint>/student --out outputs/eval/<run>-student
    python -m caad.eval.vllm_eval summarize --out outputs/eval/Qwen3-VL-4B-Instruct
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from collections import Counter, defaultdict
from pathlib import Path

from .tasks.registry import get_task
from .tasks.vqa_c import format_check, normalize_answer, split_reasoning

DATASETS = ("textvqa", "gqa")                       # default (paired clean/corrupted)
ALL_DATASETS = ("textvqa", "gqa", "vqav2", "vizwiz")
SINGLE_VIEW = {"vizwiz"}                              # real degraded photos, no clean copy
MULTI_ANSWER = {"textvqa", "vqav2", "vizwiz"}         # 10 annotator answers -> VQA soft score too
VIEWS = ("clean", "corrupted", "real", "noimage")   # noimage: question only, image withheld


def _requests(root, prompt_format, limit=None, datasets=DATASETS, no_image=False):
    """All examples: clean/corrupted interleaved per question, or the single
    real view for single-view datasets. With no_image, one text-only request
    per question (view "noimage"): same prompt, image withheld."""
    if no_image:
        one = {}
        for e in _requests(root, prompt_format, limit, datasets):
            if e["view"] in ("clean", "real"):
                one[(e["dataset"], e["id"])] = {**e, "view": "noimage"}
        return list(one.values())
    reqs = []
    for ds in datasets:
        task = get_task(ds)
        cfg = {"root": root, "prompt_format": prompt_format}
        if ds in SINGLE_VIEW:
            ex = task.load(cfg)
            reqs += ex[:limit] if limit else ex
            continue
        clean = task.load({**cfg, "use_corrupted": False})
        corr = {e["id"]: e for e in task.load({**cfg, "use_corrupted": True})}
        for e in clean[:limit] if limit else clean:
            reqs += [e, corr[e["id"]]]
    return reqs


def _datasets_in(records):
    seen = []
    for r in records:
        if r["dataset"] not in seen:
            seen.append(r["dataset"])
    return [d for d in ALL_DATASETS if d in seen]


def _key(ds, qid, view):
    return f"{ds}|{qid}|{view}"


def _md5(path):
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


# --------------------------------------------------------------------------- #
# Summary
# --------------------------------------------------------------------------- #
def summarize(records):
    """Accuracy + extraction stats, overall and broken down for the corrupted view."""
    out = {}
    for ds in _datasets_in(records):
        rs = [r for r in records if r["dataset"] == ds]
        if not rs:
            continue
        d = {}
        for v in VIEWS:
            vr = [r for r in rs if r["view"] == v]
            if not vr:
                continue
            n = len(vr)
            d[v] = {"n": n, "accuracy": sum(r["correct"] for r in vr) / n,
                    "extraction_rate": sum(r["format"]["answer"] for r in vr) / n,
                    "extraction_methods": dict(Counter(r["extraction"] for r in vr)),
                    "truncated": sum(r["truncated"] for r in vr)}
            if ds in MULTI_ANSWER:
                d[v]["vqa_soft_accuracy"] = sum(r["soft"] for r in vr) / n
        # paired comparison on questions answered in both views
        by_q = defaultdict(dict)
        for r in rs:
            by_q[r["question_id"]][r["view"]] = r["correct"]
        pairs = [x for x in by_q.values() if len(x) == 2]
        if pairs:
            d["paired"] = {"n": len(pairs),
                           "clean_right_corrupted_wrong": sum(x["clean"] and not x["corrupted"] for x in pairs),
                           "clean_wrong_corrupted_right": sum(not x["clean"] and x["corrupted"] for x in pairs)}
            if "clean" in d and "corrupted" in d:
                d["drop_points"] = 100 * (d["clean"]["accuracy"] - d["corrupted"]["accuracy"])
        if ds in SINGLE_VIEW:                       # breakdown by the dataset's own category
            g = defaultdict(list)
            for r in rs:
                g[r.get("category") or "?"].append(r)
            d["by_question_category"] = {k: {"n": len(v), "accuracy": sum(x["correct"] for x in v) / len(v),
                                              "vqa_soft_accuracy": sum(x["soft"] for x in v) / len(v)}
                                         for k, v in sorted(g.items())}
        cr = [r for r in rs if r["view"] == "corrupted"]
        for name, keyf in ([] if not cr else [("by_category", lambda r: r["corruption"]["category"]),
                           ("by_severity", lambda r: str(r["corruption"]["severity"])),
                           ("by_seen", lambda r: "seen" if r["corruption"]["seen_in_training"] else "unseen"),
                           ("by_corruption", lambda r: r["corruption"]["name"])]):
            g = defaultdict(list)
            for r in cr:
                g[keyf(r)].append(r)
            d[name] = {k: {"n": len(v), "accuracy": sum(x["correct"] for x in v) / len(v),
                           "extraction_rate": sum(x["format"]["answer"] for x in v) / len(v)}
                       for k, v in sorted(g.items())}
        out[ds] = d
    allr = records
    out["overall"] = {"n": len(allr),
                      "extraction_rate": sum(r["format"]["answer"] for r in allr) / max(len(allr), 1),
                      "truncated": sum(r["truncated"] for r in allr),
                      "errors": sum(bool(r.get("error")) for r in allr)}
    return out


def _md(summary, meta):
    L = [f"# Robustness eval — {meta.get('model', '?')}" + (f" + {meta['lora']}" if meta.get("lora") else ""), "",
         f"- outputs: {summary['overall']['n']:,} · `<answer>` extraction rate: "
         f"**{100 * summary['overall']['extraction_rate']:.2f}%** · hit token limit: "
         f"{summary['overall']['truncated']} · generation errors: {summary['overall'].get('errors', 0)}", ""]
    L += ["| Dataset | View | N | Accuracy | VQA soft | `<answer>` extracted | Hit limit |",
          "|---|---|---|---|---|---|---|"]
    for ds in [d for d in ALL_DATASETS if d in summary]:
        for v in VIEWS:
            s = summary.get(ds, {}).get(v)
            if s:
                soft = f"{100 * s['vqa_soft_accuracy']:.2f}" if "vqa_soft_accuracy" in s else "—"
                L.append(f"| {ds} | {v} | {s['n']:,} | {100 * s['accuracy']:.2f} | {soft} | "
                         f"{100 * s['extraction_rate']:.2f}% | {s['truncated']} |")
    for ds in [d for d in ALL_DATASETS if d in summary]:
        d = summary.get(ds)
        if d and "by_question_category" in d:
            L += ["", f"## {ds}: by question category", "| Category | N | Accuracy | VQA soft |", "|---|---|---|---|"]
            for k, s in d["by_question_category"].items():
                L.append(f"| {k} | {s['n']:,} | {100 * s['accuracy']:.2f} | {100 * s['vqa_soft_accuracy']:.2f} |")
        if not d or "by_category" not in d:
            continue
        L += ["", f"## {ds}: corrupted view breakdown"]
        if "drop_points" in d:
            L.append(f"Clean → corrupted drop: **{d['drop_points']:.2f} points**. "
                     f"Paired over {d['paired']['n']:,} questions: right→wrong "
                     f"{d['paired']['clean_right_corrupted_wrong']:,}, wrong→right "
                     f"{d['paired']['clean_wrong_corrupted_right']:,}.")
        for name, title in [("by_category", "Category"), ("by_severity", "Severity"),
                            ("by_seen", "Seen in CAAD training"), ("by_corruption", "Corruption")]:
            L += ["", f"| {title} | N | Accuracy | `<answer>` extracted |", "|---|---|---|---|"]
            for k, s in d[name].items():
                L.append(f"| {k} | {s['n']:,} | {100 * s['accuracy']:.2f} | {100 * s['extraction_rate']:.2f}% |")
    return "\n".join(L) + "\n"


def _write_summary(out, records, meta):
    s = summarize(records)
    (out / "summary.json").write_text(json.dumps({"meta": meta, **s}, indent=2))
    (out / "summary.md").write_text(_md(s, meta))
    return s


def _progress_line(records, total, t0, n_new_total):
    s = summarize(records)
    el = time.time() - t0
    rate = n_new_total / el if el > 0 else 0
    eta = (total - len(records)) / rate / 60 if rate else float("nan")
    parts = [f"{len(records):,}/{total:,} ({100 * len(records) / total:.1f}%)",
             f"{rate:.1f} ans/s", f"ETA {eta:.0f} min",
             f"<answer> extracted {100 * s['overall']['extraction_rate']:.2f}%"]
    for ds in [d for d in ALL_DATASETS if d in s]:
        for v in VIEWS:
            x = s.get(ds, {}).get(v)
            if x:
                parts.append(f"{ds}/{v} {100 * x['accuracy']:.1f}%")
    return " | ".join(parts)


# --------------------------------------------------------------------------- #
# Run
# --------------------------------------------------------------------------- #
def cmd_run(a):
    from PIL import Image
    from transformers import AutoProcessor
    import vllm
    from vllm import LLM, SamplingParams

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    res_path = out / "results.jsonl"
    datasets = tuple(a.datasets.split(","))
    reqs = _requests(a.root, a.prompt_format, a.limit, datasets, no_image=a.no_image)
    records = [json.loads(l) for l in res_path.read_text().splitlines() if l.strip()] \
        if res_path.exists() else []
    done = {_key(r["dataset"], r["question_id"], r["view"]) for r in records}
    todo = [e for e in reqs if _key(e["dataset"], e["id"], e["view"]) not in done]
    lora_cfg = json.loads((Path(a.lora) / "adapter_config.json").read_text()) if a.lora else None
    meta = {"model": a.model, "lora": str(Path(a.lora).resolve()) if a.lora else None,
            "lora_rank": lora_cfg["r"] if lora_cfg else None,
            "vllm": vllm.__version__, "prompt_format": a.prompt_format,
            "prompt_example": reqs[0]["prompt"], "system": reqs[0].get("system"),
            "sampling": {"temperature": 0.0, "max_tokens": a.max_tokens, "seed": 0},
            "max_model_len": a.max_model_len, "root": a.root, "no_image": a.no_image,
            "manifests_md5": {ds: _md5(Path(a.root) / ds / "manifest.jsonl") for ds in datasets},
            "total_requests": len(reqs)}
    (out / "run_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[start] {len(reqs):,} requests, {len(records):,} already done, {len(todo):,} to go",
          flush=True)
    if not todo:
        _write_summary(out, records, meta)
        print("[done] nothing to do; summary rewritten", flush=True)
        return

    processor = AutoProcessor.from_pretrained(a.model)
    lora_kw, lora_req = {}, None
    if a.lora:
        from vllm.lora.request import LoRARequest
        lora_kw = {"enable_lora": True, "max_lora_rank": lora_cfg["r"], "max_loras": 1}
        lora_req = LoRARequest("adapter", 1, str(Path(a.lora).resolve()))
        print(f"[lora] {a.lora} (rank {lora_cfg['r']})", flush=True)
    llm = LLM(model=a.model, dtype="bfloat16", max_model_len=a.max_model_len,
              limit_mm_per_prompt={"image": 1}, gpu_memory_utilization=a.gpu_mem,
              seed=0, max_num_seqs=a.max_num_seqs, tensor_parallel_size=a.tp, **lora_kw)
    sp = SamplingParams(temperature=0.0, max_tokens=a.max_tokens, seed=0)
    prompt_cache = {}

    def prompt_for(e):
        img = e["view"] != "noimage"
        k = (e.get("system"), e["prompt"], img)
        if k not in prompt_cache:
            msgs = [{"role": "user", "content": ([{"type": "image"}] if img else [])
                     + [{"type": "text", "text": e["prompt"]}]}]
            if e.get("system"):
                msgs.insert(0, {"role": "system", "content": [{"type": "text", "text": e["system"]}]})
            prompt_cache[k] = processor.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True)
        return prompt_cache[k]

    t0, n_new = time.time(), 0
    with res_path.open("a") as fh:
        for i in range(0, len(todo), a.chunk):
            chunk = todo[i:i + a.chunk]
            inputs = []
            for e in chunk:
                if e["view"] == "noimage":
                    inputs.append({"prompt": prompt_for(e)})
                    continue
                with Image.open(e["image"]) as im:
                    img = im.convert("RGB")
                inputs.append({"prompt": prompt_for(e), "multi_modal_data": {"image": img}})
            try:
                outs = llm.generate(inputs, sp, use_tqdm=False, lora_request=lora_req)
            except Exception as err:                      # one bad request must not kill the run
                print(f"[warn] chunk failed ({type(err).__name__}); retrying request by request",
                      flush=True)
                outs = []
                for inp in inputs:
                    try:
                        outs.append(llm.generate([inp], sp, use_tqdm=False,
                                                 lora_request=lora_req)[0])
                    except Exception as err1:
                        outs.append(err1)
            for e, o in zip(chunk, outs):
                task = get_task(e["dataset"])
                if isinstance(o, Exception):
                    rec = {"dataset": e["dataset"], "question_id": e["id"], "image_id": e["image_id"],
                           "view": e["view"], "image": e["image"], "question": e["question"],
                           "gold": e["gold"], "corruption": e["corruption"], "category": e.get("category"), "completion": "",
                           "extracted": "", "normalized": "", "extraction": "error",
                           "format": {"think": False, "answer": False, "strict": False},
                           "correct": False, "gen_tokens": 0, "truncated": False,
                           "error": f"{type(o).__name__}: {str(o)[:300]}"}
                    if e["dataset"] in MULTI_ANSWER:
                        rec["soft"] = 0.0
                    print(f"[error] {e['dataset']} {e['id']} {e['view']}: {rec['error'][:200]}", flush=True)
                    fh.write(json.dumps(rec) + "\n")
                    records.append(rec)
                    continue
                g = o.outputs[0]
                text = g.text
                reasoning, ans, method = split_reasoning(text)
                rec = {"dataset": e["dataset"], "question_id": e["id"], "image_id": e["image_id"],
                       "view": e["view"], "image": e["image"], "question": e["question"],
                       "gold": e["gold"], "corruption": e["corruption"], "category": e.get("category"),
                       "completion": text, "extracted": ans, "normalized": normalize_answer(ans),
                       "extraction": method, "format": format_check(text),
                       "correct": task.score(e, text),
                       "gen_tokens": len(g.token_ids),
                       "truncated": g.finish_reason == "length"}
                if e["dataset"] in MULTI_ANSWER:
                    rec["soft"] = task.soft_score(e, text)
                fh.write(json.dumps(rec) + "\n")
                records.append(rec)
            fh.flush()
            os.fsync(fh.fileno())
            n_new += len(chunk)
            print("[progress] " + _progress_line(records, len(reqs), t0, n_new), flush=True)
            if (i // a.chunk) % 5 == 4:
                _write_summary(out, records, meta)         # partial summary every 5 chunks
    s = _write_summary(out, records, meta)
    print("[done] " + _progress_line(records, len(reqs), t0, n_new), flush=True)
    print((out / "summary.md").read_text(), flush=True)


def cmd_summarize(a):
    out = Path(a.out)
    records = [json.loads(l) for l in (out / "results.jsonl").read_text().splitlines() if l.strip()]
    meta = json.loads((out / "run_meta.json").read_text()) if (out / "run_meta.json").exists() else {}
    _write_summary(out, records, meta)
    print((out / "summary.md").read_text())


def main():
    ap = argparse.ArgumentParser(description="TextVQA-C / GQA-C evaluation with vLLM")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--root", required=True)
    r.add_argument("--model", default="Qwen/Qwen3-VL-4B-Instruct")
    r.add_argument("--out", required=True)
    r.add_argument("--prompt-format", default="inline", choices=["inline", "think_answer", "inline_brief", "internvl_r1", "internvl_r1_short", "internvl_vqa"])
    r.add_argument("--lora", default=None, help="PEFT adapter dir to apply on top of --model")
    r.add_argument("--datasets", default=",".join(DATASETS),
                   help=f"comma list from {ALL_DATASETS} (default: textvqa,gqa)")
    r.add_argument("--max-tokens", type=int, default=768)
    # Qwen3-VL caps an image at 16,384 visual tokens; + question + 768 answer tokens
    r.add_argument("--max-model-len", type=int, default=18432)
    r.add_argument("--max-num-seqs", type=int, default=128)
    r.add_argument("--gpu-mem", type=float, default=0.90)
    r.add_argument("--no-image", action="store_true",
                   help="text-only: ask each question once with the image withheld (view 'noimage')")
    r.add_argument("--tp", type=int, default=1, help="tensor-parallel GPUs (e.g. 2 for an 8B model on A30s)")
    r.add_argument("--chunk", type=int, default=1024)
    r.add_argument("--limit", type=int, default=None, help="questions per dataset (smoke test)")
    s = sub.add_parser("summarize")
    s.add_argument("--out", required=True)
    a = ap.parse_args()
    {"run": cmd_run, "summarize": cmd_summarize}[a.cmd](a)


if __name__ == "__main__":
    main()
