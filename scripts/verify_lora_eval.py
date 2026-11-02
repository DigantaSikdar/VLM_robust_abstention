#!/usr/bin/env python3
"""
Check that the vLLM evaluation really applied a LoRA adapter: teacher-force the
completions vLLM generated (greedy, with the adapter) through an independent
Hugging Face + PEFT model, with and without the adapter, and measure how often
HF's argmax matches vLLM's token. If vLLM applied the adapter fully, agreement
with the adapter should be ~100% and higher than without it.

    python scripts/verify_lora_eval.py <eval_dir> <adapter_dir> [n_examples]
"""
import json
import sys

import torch

sys.path.insert(0, "src")
from caad.eval.tasks.vqa_c import PROMPT_FORMATS  # noqa: E402
from caad.trainers.caad import build_inputs  # noqa: E402


@torch.inference_mode()
def main(eval_dir, adapter, n=24):
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText, AutoProcessor
    name = "Qwen/Qwen3-VL-4B-Instruct"
    proc = AutoProcessor.from_pretrained(name)
    base = AutoModelForImageTextToText.from_pretrained(name, dtype=torch.bfloat16, device_map="cuda").eval()
    model = PeftModel.from_pretrained(base, adapter).eval()
    system, tpl = PROMPT_FORMATS["inline"]
    rows = [json.loads(l) for l in open(f"{eval_dir}/results.jsonl")]
    rows = [r for r in rows if r["dataset"] == "gqa" and r["view"] == "corrupted" and not r["truncated"]][:n // 2] + \
           [r for r in rows if r["dataset"] == "textvqa" and r["view"] == "corrupted" and not r["truncated"]][:n // 2]
    tot = {"with_adapter": [0, 0], "without_adapter": [0, 0]}
    for r in rows:
        batch, cmask, plen = build_inputs(proc, system, tpl.format(question=r["question"]), r["image"],
                                          r["completion"], "cuda")
        tgt = batch["input_ids"][0, 1:]
        m = cmask[0, 1:].bool()
        m[-1] = False                                    # skip the appended end-of-turn token
        for key, ctx in (("with_adapter", None), ("without_adapter", model.disable_adapter())):
            if ctx is None:
                logits = model(**batch).logits[0, :-1]
            else:
                with ctx:
                    logits = model(**batch).logits[0, :-1]
            agree = (logits.argmax(-1) == tgt)[m]
            tot[key][0] += int(agree.sum()); tot[key][1] += int(m.sum())
    for k, (a, b) in tot.items():
        print(f"{k:16s}: HF argmax matches vLLM's greedy token on {a}/{b} = {100 * a / b:.2f}% of tokens")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 24)
