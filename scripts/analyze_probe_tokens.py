#!/usr/bin/env python3
"""
Which tokens does a forward-KL entropy gate actually train on?

Reads outputs of caad.eval.entropy_probe (tokens.npz + rollouts.jsonl) and,
for several teacher-entropy gates, reports how the gated tokens and their
forward-KL mass (the per-token gradient weight) split across token
categories, the top tokens by forward KL, and examples where the clean-image
teacher and the corrupted-image student disagree on a confident token.

    python scripts/analyze_probe_tokens.py outputs/probe/s4_base_tokens
"""
import json
import re
import string
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

FUNCTION = set("""a an the this that these those there here it its it's they them their
he she his her we our you your i me my is are was were be been being am do does did
has have had of in on at to for from by with without into onto over under about as
than then and or but nor so if because while which who whom whose what when where why
how not no yes also very just only can could would should will may might must shall
up down out off any some all each both either neither such same other another more most
much many few""".split())
PUNCT = set(string.punctuation) | {"“", "”", "’", "‘", "—", "–", "…"}


def category(tok_str, in_answer):
    t = tok_str.strip()
    if tok_str in ("<|im_end|>",) or t in ("<answer>", "</answer>", "answer", "<", ">", "</") or re.fullmatch(r"<?/?answer>?", t or "x"):
        return "format"
    if t == "":
        return "format"                               # newline / whitespace
    if all(c in PUNCT for c in t):
        return "punct"
    if re.fullmatch(r"\d+([.,]\d+)?", t):
        return "number"
    if t.lower() in FUNCTION:
        return "function"
    return "content"


def main(path):
    from transformers import AutoTokenizer
    root = Path(path)
    D = dict(np.load(root / "tokens.npz", allow_pickle=True))
    meta = [json.loads(l) for l in open(root / "rollouts.jsonl")]
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-VL-4B-Instruct")
    dec = lambda i: tok.decode([int(i)])

    n = len(D["H_t"])
    strs = np.array([dec(i) for i in D["tok"]], dtype=object)
    # inside the <answer> ... </answer> span of its rollout
    in_ans = np.zeros(n, bool)
    for r in np.unique(D["rollout"]):
        idx = np.where(D["rollout"] == r)[0]
        text, inside = "", False
        for j in idx:
            text += strs[j]
            if inside:
                in_ans[j] = True
            if text.endswith("<answer>"):
                inside = True
            if text.endswith("</answer>"):
                inside = False
                in_ans[j] = False
    cats = np.array([category(s, a) for s, a in zip(strs, in_ans)], dtype=object)
    cats = np.where((cats == "content") | (cats == "number"), np.where(in_ans, cats + "@answer", cats), cats)

    Ht, fkl = D["H_t"], D["fkl"]
    gates = {"all tokens": np.ones(n, bool), "H_t < 0.1": Ht < 0.1, "H_t < 0.2": Ht < 0.2,
             "0.01 <= H_t < 0.2": (Ht >= 0.01) & (Ht < 0.2), "H_t > 0.8 (EOPD)": Ht > 0.8}
    order = ["content", "content@answer", "number", "number@answer", "function", "punct", "format"]
    out = {}
    for name, g in gates.items():
        tot_tok, tot_fkl = g.sum(), fkl[g].sum()
        rows = {}
        for c in order:
            m = g & (cats == c)
            rows[c] = {"tok_share": float(m.sum() / tot_tok), "fkl_share": float(fkl[m].sum() / tot_fkl),
                       "fkl_mean": float(fkl[m].mean()) if m.any() else 0.0,
                       "disagree": float(1 - D["agree"][m].mean()) if m.any() else 0.0}
        top = defaultdict(lambda: [0.0, 0])
        for s_, f_ in zip(strs[g], fkl[g]):
            k = s_.strip().lower() or repr(s_)
            top[k][0] += f_; top[k][1] += 1
        top_list = sorted(top.items(), key=lambda kv: -kv[1][0])[:20]
        out[name] = {"tokens": int(tot_tok), "tok_frac": float(g.mean()), "fkl_mean": float(fkl[g].mean()),
                     "by_category": rows,
                     "top_tokens_by_fkl": [(k, round(v[0] / tot_fkl, 4), v[1], round(v[0] / v[1], 4)) for k, v in top_list]}
    # confident-token disagreements, with context
    conf_dis = np.where((Ht < 0.2) & (D["agree"] == 0))[0]
    conf_dis = conf_dis[np.argsort(-fkl[conf_dis])]
    ex = []
    for j in conf_dis[:40]:
        r = int(D["rollout"][j]); idx = np.where(D["rollout"] == r)[0]; p = int(np.where(idx == j)[0][0])
        ctx = "".join(strs[idx[max(0, p - 12):p]])
        ex.append({"style": meta[r]["style"], "question": meta[r]["question"], "gold": meta[r]["answer"],
                   "context": ctx[-90:], "rollout_token": strs[j], "teacher_top": dec(D["t_arg"][j]),
                   "student_top": dec(D["s_arg"][j]), "H_t": round(float(Ht[j]), 4),
                   "H_s": round(float(D["H_s"][j]), 4), "fkl": round(float(fkl[j]), 3), "category": cats[j]})
    out["confident_disagreements"] = {"count": int(len(conf_dis)),
                                      "frac_of_confident_tokens": float(len(conf_dis) / max((Ht < 0.2).sum(), 1)),
                                      "category_counts": dict(Counter(cats[conf_dis].tolist())),
                                      "examples": ex}
    (root / "token_analysis.json").write_text(json.dumps(out, indent=2, ensure_ascii=False, default=float))
    for name in gates:
        o = out[name]
        print(f"\n== {name}: {o['tokens']} tokens ({100 * o['tok_frac']:.1f}%), mean FKL {o['fkl_mean']:.4f}")
        print(f"   {'category':16s} {'tok%':>6} {'FKL%':>6} {'FKL mean':>9} {'top1 disagree':>13}")
        for c in order:
            x = o["by_category"][c]
            print(f"   {c:16s} {100 * x['tok_share']:6.1f} {100 * x['fkl_share']:6.1f} {x['fkl_mean']:9.4f} {100 * x['disagree']:12.1f}%")
        print("   top tokens by FKL share: " + ", ".join(f"{k!s}({100 * s:.1f}%)" for k, s, c_, m in o["top_tokens_by_fkl"][:15]))
    cd = out["confident_disagreements"]
    print(f"\n== confident (H_t<0.2) tokens where teacher and student pick different tokens: {cd['count']} "
          f"({100 * cd['frac_of_confident_tokens']:.1f}%), by category {cd['category_counts']}")
    for e in cd["examples"][:15]:
        print(f"   [{e['style']}] Q: {e['question'][:60]!r} gold={e['gold']!r}\n      …{e['context']!r} → teacher {e['teacher_top']!r} vs student {e['student_top']!r} "
              f"(H_t {e['H_t']}, FKL {e['fkl']}, {e['category']})")


if __name__ == "__main__":
    main(sys.argv[1])
