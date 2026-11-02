"""
caad/eval/entropy_probe.py
==========================
Token-level probe of the CAAD teacher/student pair, no training:

  student = model on the CORRUPTED image, generates the rollout (temp 1.0)
  teacher = same model on the CLEAN image, scores the student's tokens

Per completion token it records teacher / student entropy, top-k forward KL
(teacher || student, teacher renormalized over its top-k, as in the loss),
JSD_0.5, exact reverse KL, teacher top-1 probability and whether the two
argmaxes agree. Writes tokens.npz + a summary (entropy distribution, behaviour
per teacher-entropy band, and candidate low-entropy thresholds).

    python -m caad.eval.entropy_probe --manifest .../gqa_train_30k_s4/manifest.jsonl \
        --n 400 --out outputs/probe/s4_base
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from ..data.vision import build_messages, process_image
from ..eval.tasks.vqa_c import PROMPT_FORMATS
from ..trainers.caad import build_inputs

BANDS = [0, 1e-4, 1e-3, 1e-2, 0.05, 0.1, 0.2, 0.4, 0.8, 1.5, 100]


@torch.no_grad()
def token_stats(s_logits, t_logits, targets, topk):
    ls = s_logits.float().log_softmax(-1)
    lt = t_logits.float().log_softmax(-1)
    ps, pt = ls.exp(), lt.exp()
    H_s = -(ps * ls).sum(-1)
    H_t = -(pt * lt).sum(-1)
    top, idx = lt.topk(topk, -1)
    lpt = top.log_softmax(-1)
    fkl = (lpt.exp() * (lpt - ls.gather(-1, idx))).sum(-1)
    log_m = torch.logsumexp(torch.stack([ls + math.log(0.5), lt + math.log(0.5)]), 0)
    jsd = 0.5 * (ps * (ls - log_m)).sum(-1) + 0.5 * (pt * (lt - log_m)).sum(-1)
    rkl = (ps * (ls - lt)).sum(-1)
    return dict(H_t=H_t, H_s=H_s, fkl=fkl, jsd=jsd, rkl=rkl,
                t_top1=pt.max(-1).values, agree=(ps.argmax(-1) == pt.argmax(-1)).float(),
                tok=targets, t_arg=pt.argmax(-1), s_arg=ps.argmax(-1),
                s_logp_tok=ls.gather(-1, targets[..., None]).squeeze(-1),
                t_logp_tok=lt.gather(-1, targets[..., None]).squeeze(-1))


@torch.inference_mode()            # forward passes only: keep no autograd graph
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--model", default="Qwen/Qwen3-VL-4B-Instruct")
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--topk", type=int, default=16)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    from transformers import AutoModelForImageTextToText, AutoProcessor
    model = AutoModelForImageTextToText.from_pretrained(
        a.model, dtype=torch.bfloat16, attn_implementation="sdpa", device_map="cuda").eval()
    proc = AutoProcessor.from_pretrained(a.model)
    system, user_tpl = PROMPT_FORMATS["inline"]

    rows = [json.loads(l) for l in open(a.manifest)]
    by_style = defaultdict(list)
    for r in rows:
        by_style[r["corruption"]["style"]].append(r)
    rng = random.Random(a.seed)
    per = math.ceil(a.n / len(by_style))
    picks = [r for st in sorted(by_style) for r in rng.sample(by_style[st], per)][:a.n]

    torch.manual_seed(a.seed)
    rec = defaultdict(list)
    meta = []
    t0 = time.time()
    for i, r in enumerate(picks):
        user = user_tpl.format(question=r["question"])
        msgs = build_messages(r["corrupted_image"], user, system)
        text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inp = process_image(proc, text, msgs).to("cuda")
        gen = model.generate(**inp, do_sample=True, temperature=1.0, top_p=1.0,
                             max_new_tokens=a.max_new_tokens)
        completion = proc.decode(gen[0, inp["input_ids"].shape[1]:], skip_special_tokens=True)
        cb, cmask, _ = build_inputs(proc, system, user, r["corrupted_image"], completion, "cuda")
        kb, _, _ = build_inputs(proc, system, user, r["clean_image"], completion, "cuda")
        kb["input_ids"], kb["attention_mask"] = cb["input_ids"], cb["attention_mask"]
        m = cmask[:, 1:].bool()[0]
        pos = m.nonzero().squeeze(-1)                       # completion positions only
        s_logits = model(**cb).logits[0, :-1][pos]
        t_logits = model(**kb).logits[0, :-1][pos]
        targets = cb["input_ids"][0, 1:][pos]
        n_tok = int(pos.numel())
        parts = defaultdict(list)
        for c in range(0, n_tok, 64):                       # chunk: full-vocab float32 is large
            st = token_stats(s_logits[c:c + 64], t_logits[c:c + 64], targets[c:c + 64], a.topk)
            for k, v in st.items():
                parts[k].append(v.cpu().numpy())
        del s_logits, t_logits, gen
        torch.cuda.empty_cache()
        for k, v in parts.items():
            rec[k].append(np.concatenate(v))
        rec["style"].append(np.array([r["corruption"]["style"]] * n_tok))
        rec["rollout"].append(np.full(n_tok, i))
        rec["pos"].append(np.arange(n_tok))
        meta.append({"rollout": i, "question": r["question"], "answer": r["answer"],
                     "style": r["corruption"]["style"], "completion": completion})
        if (i + 1) % 25 == 0:
            print(f"[probe] {i + 1}/{len(picks)} rollouts, {time.time() - t0:.0f}s", flush=True)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    D = {k: np.concatenate(v) for k, v in rec.items()}
    np.savez_compressed(out / "tokens.npz", **D)
    with (out / "rollouts.jsonl").open("w") as fh:
        for m_ in meta:
            fh.write(json.dumps(m_) + "\n")
    summary = summarize(D)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def summarize(D):
    Ht, Hs = D["H_t"], D["H_s"]
    n = len(Ht)
    qs = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    S = {"tokens": n, "rollouts": int(D["rollout"].max() + 1),
         "teacher_entropy_percentiles": {f"p{q}": float(np.percentile(Ht, q)) for q in qs},
         "student_entropy_percentiles": {f"p{q}": float(np.percentile(Hs, q)) for q in qs},
         "spearman_Ht_Hs": float(np.corrcoef(np.argsort(np.argsort(Ht)), np.argsort(np.argsort(Hs)))[0, 1]),
         "frac_H_t_gt_0.8": float((Ht > 0.8).mean()),
         "mean": {k: float(D[k].mean()) for k in ("fkl", "jsd", "rkl", "H_t", "H_s", "agree")}}
    bands = []
    tot_fkl = D["fkl"].sum()
    for lo, hi in zip(BANDS[:-1], BANDS[1:]):
        m = (Ht >= lo) & (Ht < hi)
        if not m.any():
            continue
        bands.append({"H_t_band": f"[{lo:g}, {hi:g})", "frac_tokens": float(m.mean()),
                      "mean_H_t": float(Ht[m].mean()), "mean_H_s": float(Hs[m].mean()),
                      "mean_H_s_minus_H_t": float((Hs[m] - Ht[m]).mean()),
                      "frac_student_more_uncertain": float((Hs[m] > Ht[m]).mean()),
                      "teacher_top1_prob": float(D["t_top1"][m].mean()),
                      "argmax_disagree": float(1 - D["agree"][m].mean()),
                      "fkl_mean": float(D["fkl"][m].mean()), "fkl_p90": float(np.percentile(D["fkl"][m], 90)),
                      "share_of_total_fkl": float(D["fkl"][m].sum() / tot_fkl),
                      "jsd_mean": float(D["jsd"][m].mean()), "rkl_mean": float(D["rkl"][m].mean())})
    S["by_teacher_entropy_band"] = bands
    cands = []
    for tau in (1e-4, 1e-3, 3e-3, 0.01, 0.03, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5):
        m = Ht < tau
        cands.append({"tau_low": tau, "frac_tokens_gated": float(m.mean()),
                      "teacher_top1_prob_min": float(D["t_top1"][m].min()) if m.any() else None,
                      "fkl_mean_on_gated": float(D["fkl"][m].mean()) if m.any() else 0.0,
                      "fkl_per_token_contribution": float(D["fkl"][m].sum() / n),
                      "argmax_disagree_on_gated": float(1 - D["agree"][m].mean()) if m.any() else 0.0,
                      "frac_gated_where_student_H_gt_0.5": float(((Hs > 0.5) & m).sum() / max(m.sum(), 1))})
    S["low_entropy_threshold_candidates"] = cands
    S["by_style_fkl_low_H_t_lt_0.1"] = {
        st: float(D["fkl"][(D["style"] == st) & (Ht < 0.1)].mean())
        for st in sorted(set(D["style"].tolist())) if ((D["style"] == st) & (Ht < 0.1)).any()}
    return S


if __name__ == "__main__":
    main()
