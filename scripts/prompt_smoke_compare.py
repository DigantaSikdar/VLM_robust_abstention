#!/usr/bin/env python3
"""Tag (extraction) rate and accuracy per prompt format on the prompt smoke-test
subset; the current inline prompt is read from the full base evaluations,
restricted to the same questions (greedy decoding, so identical requests give
identical outputs)."""
import json
from pathlib import Path

DS = ["textvqa", "gqa", "vqav2"]
FULL = {"4b": ["outputs/eval/Qwen3-VL-4B-Instruct", "outputs/eval/base-extra"],
        "8b": ["outputs/eval/Qwen3-VL-8B-Instruct", "outputs/eval/base8b-extra"]}


def rows(paths):
    out = {}
    for p in paths:
        f = Path(p) / "results.jsonl"
        if f.exists():
            for r in map(json.loads, open(f)):
                out[(r["dataset"], r["question_id"], r["view"])] = r
    return out


def stats(R, keys):
    rs = [R[k] for k in keys if k in R]
    if not rs:
        return None
    tag = 100 * sum(r["extraction"] == "tag" for r in rs) / len(rs)
    acc = {f"{d}/{v}": 100 * sum(r["correct"] for r in rs if (r["dataset"], r["view"]) == (d, v)) /
           max(1, sum((r["dataset"], r["view"]) == (d, v) for r in rs))
           for d in DS for v in ("clean", "corrupted")}
    toks = sorted(r["gen_tokens"] for r in rs)
    return len(rs), tag, acc, toks[len(toks) // 2], 100 * sum(r["correct"] for r in rs) / len(rs)


def main():
    cols = [f"{d}/{v}" for d in DS for v in ("clean", "corrupted")]
    print(f"{'model':5s} {'prompt':14s} {'n':>5s} {'tag%':>6s} {'overall':>8s} " +
          " ".join(f"{c[:7] + '/' + c.split('/')[1][:4]:>12s}" for c in cols) + "  median_tokens")
    for m in ("4b", "8b"):
        new = {f: rows([f"outputs/prompt_smoke/{m}_{f}"]) for f in ("inline_brief", "think_answer")}
        keys = sorted(set().union(*[set(r) for r in new.values()]))
        for name, R in [("inline (now)", rows(FULL[m])), ("option1 brief", new["inline_brief"]),
                        ("option2 system", new["think_answer"])]:
            s = stats(R, keys)
            if s is None:
                print(f"{m:5s} {name:14s}  (not run yet)")
                continue
            n, tag, acc, med, ov = s
            print(f"{m:5s} {name:14s} {n:5d} {tag:6.1f} {ov:8.2f} " +
                  " ".join(f"{acc[c]:12.1f}" for c in cols) + f"  {med}")


if __name__ == "__main__":
    main()
