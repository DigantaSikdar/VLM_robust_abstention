"""What do the teacher and the base student predict at the answer tokens of the distilled model's output?

For GQA eval questions, take the distilled student's completion, re-score it with each scorer model
on the CLEAN image, and print the top-5 next tokens at the first answer token (right after
"<answer>") and the probability each scorer gives to the token the student actually emitted.

    python scripts/answer_token_probe.py <student results.jsonl> <out.json> <scorer model> [<scorer model> ...]
"""
import json
import sys
from collections import Counter

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

sys.path.insert(0, "src")
from caad.eval.tasks.vqa_c import PROMPT_FORMATS, extract_answer  # noqa: E402
from caad.trainers.caad import build_inputs  # noqa: E402

RES, OUT, SCORERS = sys.argv[1], sys.argv[2], sys.argv[3:]
system, tpl = PROMPT_FORMATS["think_answer"]
rows = [r for r in map(json.loads, open(RES)) if r["dataset"] == "gqa" and r["view"] == "clean" and "<answer>" in r["completion"]]
proc = AutoProcessor.from_pretrained(SCORERS[0])
tok = proc.tokenizer
report = {}
for name in SCORERS:
    model = AutoModelForImageTextToText.from_pretrained(name, dtype=torch.bfloat16, attn_implementation="sdpa",
                                                        device_map="cuda").eval()
    stats, agree, top1 = [], 0, Counter()
    for r in rows:
        comp = r["completion"]
        b, mask, _ = build_inputs(proc, system, tpl.format(question=r["question"]), r["image"], comp, "cuda")
        ids = b["input_ids"][0]
        pos = mask[0, 1:].bool().nonzero().squeeze(-1)           # logits row i predicts token i+1
        comp_ids = ids[1:][pos].tolist()
        # the token after "<answer" is ">" or a merged ">c", ">w", ... (BPE joins ">" with the next letter)
        txt, k_ans = "", None
        for k, t in enumerate(comp_ids):
            txt += tok.decode([t])
            if txt.endswith("<answer"):
                k_ans = k + 1
                break
        if k_ans is None or k_ans + 1 >= len(comp_ids):
            continue
        with torch.inference_mode():
            lp = model(**b).logits[0, :-1][pos[k_ans:k_ans + 2]].float().log_softmax(-1)
        e0, e1 = comp_ids[k_ans], comp_ids[k_ans + 1]
        gt = lambda i: [(tok.decode([j]), round(float(v.exp()), 3)) for v, j in zip(*lp[i].topk(5))]
        p_gt = float(lp[0, tok.convert_tokens_to_ids(">")].exp())
        stats.append({"q": r["question"], "gold": r["gold"], "answer": extract_answer(r["completion"]),
                      "emitted": [tok.decode([e0]), tok.decode([e1])],
                      "p_emitted": [float(lp[0, e0].exp()), float(lp[1, e1].exp())],
                      "p_plain_gt": p_gt, "top5_tok1": gt(0), "top5_tok2": gt(1)})
        agree += lp[0].argmax().item() == e0
        top1[tok.decode([lp[0].argmax().item()])] += 1
    report[name] = {"n": len(stats), "top1_agree_with_student": agree / max(len(stats), 1),
                    "mean_p_emitted_tok1": sum(s["p_emitted"][0] for s in stats) / max(len(stats), 1),
                    "mean_p_plain_gt": sum(s["p_plain_gt"] for s in stats) / max(len(stats), 1),
                    "top1_first_tokens": top1.most_common(10), "items": stats}
    print(name, {k: v for k, v in report[name].items() if k != "items"}, flush=True)
    del model
    torch.cuda.empty_cache()
json.dump(report, open(OUT, "w"), indent=1)
