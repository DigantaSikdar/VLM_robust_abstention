"""Given the distilled student's reasoning (cut right after "<answer>"), which answer word does each model pick?

For every GQA clean eval item: next-token distribution after "...</think>\\n\\n<answer>" on the clean image,
P(first token of the student's emitted word) vs P(first token of the gold word), per scorer model.

    python scripts/answer_word_probe.py <student results.jsonl> <out.json> <scorer> [<scorer> ...]
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
COLLAPSED = {"cabinet", "couch", "woman"}
system, tpl = PROMPT_FORMATS["think_answer"]
rows = [r for r in map(json.loads, open(RES)) if r["dataset"] == "gqa" and r["view"] == "clean" and "<answer>" in r["completion"]]
proc = AutoProcessor.from_pretrained(SCORERS[0])
tok = proc.tokenizer
first = lambda w: tok.encode(w, add_special_tokens=False)[0]
GT = tok.convert_tokens_to_ids(">")
report = {}
for name in SCORERS:
    model = AutoModelForImageTextToText.from_pretrained(name, dtype=torch.bfloat16, attn_implementation="sdpa",
                                                        device_map="cuda").eval()
    items, top1 = [], Counter()
    for r in rows:
        prefix = r["completion"][:r["completion"].index("<answer>") + len("<answer>")]
        b, mask, _ = build_inputs(proc, system, tpl.format(question=r["question"]), r["image"], prefix, "cuda")
        last = int((b["input_ids"][0] == GT).nonzero().max())  # the ">" closing "<answer>" (an end-of-turn token follows)
        with torch.inference_mode():
            lp = model(**b).logits[0, last].float().log_softmax(-1)
        said, gold = extract_answer(r["completion"]).strip().lower(), str(r["gold"]).strip().lower()
        t = tok.decode([lp.argmax().item()])
        top1[t] += 1
        items.append({"q": r["question"], "gold": gold, "said": said, "collapsed": said in COLLAPSED,
                      "p_said": float(lp[first(said)].exp()) if said else None, "p_gold": float(lp[first(gold)].exp()),
                      "top5": [(tok.decode([j]), round(float(v.exp()), 3)) for v, j in zip(*lp.topk(5))]})
    col = [i for i in items if i["collapsed"]]
    avg = lambda xs, k: sum(x[k] for x in xs) / max(len(xs), 1)
    report[name] = {"n": len(items), "n_collapsed": len(col),
                    "collapsed: mean P(said first tok)": avg(col, "p_said"), "collapsed: mean P(gold first tok)": avg(col, "p_gold"),
                    "collapsed: top1 tokens": Counter(i["top5"][0][0] for i in col).most_common(6),
                    "all: top1 tokens": top1.most_common(8), "items": items}
    print(name, {k: v for k, v in report[name].items() if k != "items"}, flush=True)
    del model
    torch.cuda.empty_cache()
json.dump(report, open(OUT, "w"), indent=1)
