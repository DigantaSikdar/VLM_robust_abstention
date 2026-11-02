"""Does the trainer train on the tokens the student actually sampled?

The trainer receives the rollout as TEXT and re-tokenizes it (build_inputs: tok(completion + end)).
Sample rollouts with vLLM exactly as in training (think_answer system prompt, corrupted GQA-train
image, temperature 1), and compare vLLM's sampled token ids with that re-tokenization.

    python scripts/retokenize_check.py <model> <out.json> [n_questions]
"""
import json
import sys
from collections import Counter

from PIL import Image
from transformers import AutoProcessor
from vllm import LLM, SamplingParams

sys.path.insert(0, "src")
from caad.eval.tasks.vqa_c import PROMPT_FORMATS  # noqa: E402

MODEL, OUT = sys.argv[1], sys.argv[2]
N = int(sys.argv[3]) if len(sys.argv) > 3 else 300
MANIFEST = "/gscratch/dsikdar/caad_data/gqa_train_30k_s4/manifest.jsonl"
system, tpl = PROMPT_FORMATS["think_answer"]
rows = [json.loads(l) for _, l in zip(range(N), open(MANIFEST))]

proc = AutoProcessor.from_pretrained(MODEL)
tok = proc.tokenizer
vocab = tok.get_vocab()
end = next((t for t in ("<|im_end|>", "<end_of_turn>") if t in vocab), tok.eos_token or "")
end_ids = tok(end, add_special_tokens=False).input_ids


def prompt(q):
    msgs = [{"role": "system", "content": [{"type": "text", "text": system}]},
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": tpl.format(question=q)}]}]
    return proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


llm = LLM(model=MODEL, dtype="bfloat16", max_model_len=8192, limit_mm_per_prompt={"image": 1},
          gpu_memory_utilization=0.85, seed=0, max_num_seqs=64)
sp = SamplingParams(n=2, temperature=1.0, top_p=1.0, max_tokens=512, seed=0)
inputs = [{"prompt": prompt(r["question"]),
           "multi_modal_data": {"image": Image.open(r["corrupted_image"]).convert("RGB")}} for r in rows]
outs = llm.generate(inputs, sp)

n = n_diff = n_ans = n_ans_lost = 0
first_diff, examples = Counter(), []
for r, o in zip(rows, outs):
    for g in o.outputs:
        sampled = list(g.token_ids)
        if g.finish_reason == "stop" and sampled[-len(end_ids):] != end_ids:
            sampled += end_ids                       # vLLM may drop the stop token; trainer appends it
        retok = tok(g.text + end, add_special_tokens=False).input_ids
        n += 1
        diff = sampled != retok
        n_diff += diff
        # is there a token boundary right after "<answer>" in each sequence (i.e. does the trainer
        # ever see the position where the student picks the first answer token)?
        def boundary(ids):
            s = ""
            for t in ids:
                s += tok.decode([t])
                if s.endswith("<answer>"):
                    return True
                if "<answer>" in s:
                    return False
            return None
        b_s, b_r = boundary(sampled), boundary(retok)
        if b_s is not None:
            n_ans += 1
            n_ans_lost += bool(b_s and not b_r)
        if diff:
            k = next(i for i, (a, b) in enumerate(zip(sampled + [-1], retok + [-1])) if a != b)
            ctx = tok.decode(sampled[max(0, k - 3):k])
            first_diff[(ctx[-12:], tok.convert_ids_to_tokens(sampled[k]) if k < len(sampled) else "<end>",
                        tok.convert_ids_to_tokens(retok[k]) if k < len(retok) else "<end>")] += 1
            if len(examples) < 15:
                i = g.text.find("<answer>")
                examples.append({"answer_text": g.text[i:i + 40] if i >= 0 else g.text[-40:],
                                 "sampled": tok.convert_ids_to_tokens(sampled[k:k + 4]),
                                 "retokenized": tok.convert_ids_to_tokens(retok[k:k + 4])})

rep = {"model": MODEL, "completions": n, "token_sequence_differs": n_diff / n,
       "with_answer_tag": n_ans,
       "answer_boundary_sampled_but_merged_in_retok": n_ans_lost / max(n_ans, 1),
       "first_diff_top": [[list(k), v] for k, v in first_diff.most_common(12)], "examples": examples}
print(json.dumps({k: v for k, v in rep.items() if k != "examples"}, indent=1))
json.dump(rep, open(OUT, "w"), indent=1)
import os  # noqa: E402
os._exit(0)                                  # vLLM can hang on interpreter shutdown
