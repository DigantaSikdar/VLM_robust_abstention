"""
caad/eval/overlap_probe.py
==========================
Student/teacher alignment metrics of Li et al. (2026), "Rethinking On-Policy
Distillation" (Sec. 2.3), measured on student-generated rollouts, no training:

  overlap ratio      |TopK(p_t) n TopK(q_t)| / k                      (k = 16)
  overlap mass       sum of p_t (student) / q_t (teacher) over the shared top-k
  overlap advantage  mean over v in the shared set of p_bar(v) (log q_bar(v) - log p_bar(v)),
                     p_bar / q_bar renormalized over the shared set (also reported
                     summed over v, which equals -KL(p_bar || q_bar))
  entropy gap        |H(p_t) - H(q_t)|

The student generates on its view (default: the corrupted image, temperature 1.0,
as in training); the teacher scores the same tokens on its view (default: clean).
Rollout prompts are drawn with a fixed seed, so every configuration probes the
same questions.

    # step 0, 4B self-teacher            # step 1000 of an 8B-teacher run
    python -m caad.eval.overlap_probe --out outputs/overlap/self_s0
    python -m caad.eval.overlap_probe --teacher-model Qwen/Qwen3-VL-8B-Instruct \
        --student-lora .../t8b_s4_rkl/checkpoints/step_1000/student --out ...
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


@torch.no_grad()
def overlap_stats(s_logits, t_logits, k):
    ls = s_logits.float().log_softmax(-1)
    lt = t_logits.float().log_softmax(-1)
    ps, pt = ls.exp(), lt.exp()
    sp = ls.topk(k, -1).indices                                 # (T, k)
    sq = lt.topk(k, -1).indices
    shared = (sp[:, :, None] == sq[:, None, :]).any(-1)         # (T, k): student top-k token also in teacher top-k
    n_shared = shared.sum(-1)
    lp_k, lq_k = ls.gather(-1, sp), lt.gather(-1, sp)           # log-probs on the student's top-k
    neg = torch.finfo(torch.float32).min
    lp_bar = torch.where(shared, lp_k, neg).log_softmax(-1)     # renormalized over the shared set
    lq_bar = torch.where(shared, lq_k, neg).log_softmax(-1)
    adv_v = torch.where(shared, lp_bar.exp() * (lq_bar - lp_bar), 0.0)
    return dict(
        overlap=n_shared.float() / k,
        mass_s=(lp_k.exp() * shared).sum(-1),
        mass_t=(lq_k.exp() * shared).sum(-1),
        adv_sum=adv_v.sum(-1),
        adv_mean=adv_v.sum(-1) / n_shared.clamp(min=1),
        H_s=-(ps * ls).sum(-1), H_t=-(pt * lt).sum(-1),
        top1_agree=(sp[:, 0] == sq[:, 0]).float())


def load_adapter(model, path, name):
    from peft import PeftModel
    if isinstance(model, PeftModel):
        model.load_adapter(path, adapter_name=name)
        return model
    return PeftModel.from_pretrained(model, path, adapter_name=name)


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="/gscratch/dsikdar/caad_data/gqa_train_30k_s4/manifest.jsonl")
    ap.add_argument("--model", default="Qwen/Qwen3-VL-4B-Instruct")
    ap.add_argument("--student-lora", default=None, help="student adapter dir (none = base model)")
    ap.add_argument("--teacher-model", default=None, help="separate frozen teacher (none = the 4B itself)")
    ap.add_argument("--teacher-lora", default=None, help="EMA teacher adapter on the 4B (self-teacher runs)")
    ap.add_argument("--student-view", default="corrupted", choices=["corrupted", "clean"])
    ap.add_argument("--teacher-view", default="clean", choices=["corrupted", "clean"])
    ap.add_argument("--teacher-device", default="cuda", help="e.g. cuda:1 when both models do not fit one GPU")
    ap.add_argument("--prompt-format", default="inline")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--topk", type=int, default=16)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    from transformers import AutoModelForImageTextToText, AutoProcessor
    load = lambda name, dev="cuda": AutoModelForImageTextToText.from_pretrained(
        name, dtype=torch.bfloat16, attn_implementation="sdpa", device_map=dev).eval()
    model = load(a.model)
    proc = AutoProcessor.from_pretrained(a.model)
    if a.student_lora:
        model = load_adapter(model, a.student_lora, "student")
    if a.teacher_lora:
        model = load_adapter(model, a.teacher_lora, "teacher")
    teacher = load(a.teacher_model, a.teacher_device) if a.teacher_model else None
    system, user_tpl = PROMPT_FORMATS[a.prompt_format]

    def use(adapter):
        """Select the 4B's adapter; None = the bare base model."""
        from peft import PeftModel
        if isinstance(model, PeftModel):
            if adapter is None:
                return model.disable_adapter()
            model.set_adapter(adapter)
        return _null()

    s_adapter = "student" if a.student_lora else None
    t_adapter = "teacher" if a.teacher_lora else None

    rows = [json.loads(l) for l in open(a.manifest)]
    by_style = defaultdict(list)
    for r in rows:
        by_style[r["corruption"]["style"]].append(r)
    rng = random.Random(a.seed)
    per = math.ceil(a.n / len(by_style))
    picks = [r for st in sorted(by_style) for r in rng.sample(by_style[st], per)][:a.n]
    img = lambda r, view: r["corrupted_image"] if view == "corrupted" else r["clean_image"]

    torch.manual_seed(a.seed)
    rec, meta, t0 = defaultdict(list), [], time.time()
    for i, r in enumerate(picks):
        user = user_tpl.format(question=r["question"])
        msgs = build_messages(img(r, a.student_view), user, system)
        text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inp = process_image(proc, text, msgs).to("cuda")
        with use(s_adapter):
            gen = model.generate(**inp, do_sample=True, temperature=a.temperature, top_p=1.0,
                                 max_new_tokens=a.max_new_tokens)
        completion = proc.decode(gen[0, inp["input_ids"].shape[1]:], skip_special_tokens=True)
        sb, smask, _ = build_inputs(proc, system, user, img(r, a.student_view), completion, "cuda")
        tb, _, _ = build_inputs(proc, system, user, img(r, a.teacher_view), completion, "cuda")
        assert tb["input_ids"].shape == sb["input_ids"].shape, "clean/corrupted prompt lengths differ"
        pos = smask[:, 1:].bool()[0].nonzero().squeeze(-1)
        with use(s_adapter):
            s_logits = model(**sb).logits[0, :-1][pos]
        if teacher is not None:
            t_logits = teacher(**{k: v.to(a.teacher_device) for k, v in tb.items()}).logits[0, :-1][pos.to(a.teacher_device)].to(s_logits.device)
        else:
            with use(t_adapter):
                t_logits = model(**tb).logits[0, :-1][pos]
        parts = defaultdict(list)
        for c in range(0, pos.numel(), 64):
            for k, v in overlap_stats(s_logits[c:c + 64], t_logits[c:c + 64], a.topk).items():
                parts[k].append(v.cpu().numpy())
        n_tok = int(pos.numel())
        for k, v in parts.items():
            rec[k].append(np.concatenate(v))
        rec["rollout"].append(np.full(n_tok, i))
        rec["pos"].append(np.arange(n_tok))
        meta.append({"rollout": i, "question": r["question"], "answer": r["answer"],
                     "style": r["corruption"]["style"], "completion": completion})
        del s_logits, t_logits, gen
        if (i + 1) % 25 == 0:
            print(f"[overlap] {i + 1}/{len(picks)} rollouts, {time.time() - t0:.0f}s", flush=True)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    D = {k: np.concatenate(v) for k, v in rec.items()}
    np.savez_compressed(out / "tokens.npz", **D)
    with (out / "rollouts.jsonl").open("w") as fh:
        for m_ in meta:
            fh.write(json.dumps(m_) + "\n")
    S = summarize(D)
    S["config"] = vars(a)
    (out / "summary.json").write_text(json.dumps(S, indent=2))
    print(json.dumps(S, indent=2))


class _null:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def summarize(D):
    gap = np.abs(D["H_s"] - D["H_t"])
    S = {"tokens": int(len(gap)), "rollouts": int(D["rollout"].max() + 1),
         "overlap_ratio": float(D["overlap"].mean()),
         "student_overlap_mass": float(D["mass_s"].mean()),
         "teacher_overlap_mass": float(D["mass_t"].mean()),
         "overlap_adv_mean_over_v": float(D["adv_mean"].mean()),
         "overlap_adv_sum_over_v": float(D["adv_sum"].mean()),
         "abs_entropy_gap": float(gap.mean()),
         "H_student": float(D["H_s"].mean()), "H_teacher": float(D["H_t"].mean()),
         "top1_agree": float(D["top1_agree"].mean())}
    # overlap on the tokens where the distributions carry real choice (the paper's
    # math traces are far less peaked than short VQA answers)
    for lo in (0.1, 0.5):
        m = np.maximum(D["H_s"], D["H_t"]) > lo
        S[f"overlap_ratio_where_maxH_gt_{lo}"] = float(D["overlap"][m].mean()) if m.any() else None
        S[f"frac_tokens_maxH_gt_{lo}"] = float(m.mean())
    return S


if __name__ == "__main__":
    main()
