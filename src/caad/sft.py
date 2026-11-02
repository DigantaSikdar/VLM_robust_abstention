"""
caad/sft.py
===========
Supervised cold start (Li et al. 2026, Sec. 5.1): fine-tune the student LoRA on
teacher-written completions (caad.data.coldstart -> sft.jsonl) with token-level
cross-entropy on the completion only. Image view per row: data.image_view
(clean by default, so teacher and student both saw the clean image).

The adapter is saved in the same layout as distillation checkpoints
(<output_dir>/checkpoints/final/student), so it can be evaluated with
eval_vqa_c.sbatch --lora and used as lora.init_student_from for distillation.

    python -m caad.sft --config configs/sft/coldstart_8b_qwen3vl4b.yaml [key=value ...]
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import time
from pathlib import Path

import torch

from .utils.config import load_config
from .trainers.caad import build_inputs

log = logging.getLogger("caad.sft")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("overrides", nargs="*")
    a = ap.parse_args()
    cfg = load_config(a.config, overrides=a.overrides)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    from transformers import get_cosine_schedule_with_warmup
    from .eval.tasks.vqa_c import PROMPT_FORMATS
    from .models.model_utils import load_base_model
    from .utils.logging import MetricsLogger

    torch.manual_seed(cfg["seed"])
    out = Path(cfg["output_dir"])
    (out / "results").mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(cfg, indent=2))
    model, proc = load_base_model(cfg)                  # method: sft -> student adapter only
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model.train()
    dev = "cuda"
    model.to(dev)

    d, t = cfg["data"], cfg["train"]
    rows = [json.loads(l) for l in open(d["manifest"])]
    if d.get("limit"):
        rows = rows[:d["limit"]]
    view = d.get("image_view", "clean")
    system, user_tpl = PROMPT_FORMATS[d.get("prompt_format", "inline")]
    accum = t["gradient_accumulation_steps"]
    total = math.ceil(len(rows) * t.get("num_train_epochs", 1) / accum)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=t["learning_rate"], weight_decay=t.get("weight_decay", 0.0))
    sched = get_cosine_schedule_with_warmup(opt, int(t.get("warmup_ratio", 0.03) * total), total)
    metrics = MetricsLogger(out, wandb_cfg=cfg.get("wandb"), full_config=cfg)
    log.info("sft rows=%d view=%s optimizer_steps=%d trainable=%.2fM", len(rows), view, total,
             sum(p.numel() for p in params) / 1e6)

    rng = random.Random(cfg["seed"])
    step, t0, win = 0, time.time(), {"loss": 0.0, "tok": 0, "acc": 0.0}
    for epoch in range(t.get("num_train_epochs", 1)):
        order = list(range(len(rows)))
        rng.shuffle(order)
        for i, j in enumerate(order):
            r = rows[j]
            img = r["clean_image"] if view == "clean" else r["corrupted_image"]
            batch, cmask, _ = build_inputs(proc, system, user_tpl.format(question=r["question"]), img,
                                           r["completion"], dev, cfg["model"].get("max_pixels"))
            logits = model(**batch).logits[:, :-1].float()
            tgt = batch["input_ids"][:, 1:]
            m = cmask[:, 1:].bool()
            nll = torch.nn.functional.cross_entropy(logits[m], tgt[m])     # token mean within the sample
            (nll / accum).backward()
            win["loss"] += nll.item()
            win["tok"] += int(m.sum())
            win["acc"] += (logits[m].argmax(-1) == tgt[m]).float().mean().item()
            if (i + 1) % accum and i + 1 != len(order):
                continue
            gn = torch.nn.utils.clip_grad_norm_(params, t.get("max_grad_norm", 1.0))
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            s_per = (time.time() - t0) / step
            stats = {"sft/loss": win["loss"] / accum, "sft/token_acc": win["acc"] / accum,
                     "sft/tokens": win["tok"] / accum, "train/grad_norm": float(gn),
                     "train/lr": sched.get_last_lr()[0], "time/s_per_step": s_per}
            metrics.log(step, stats)
            if step % t.get("logging_steps", 10) == 0 or step == 1:
                log.info("step %d/%d | loss %.4f | tok_acc %.3f | grad %.3f | lr %.2e | %.1fs/step | eta %.2fh",
                         step, total, stats["sft/loss"], stats["sft/token_acc"], stats["train/grad_norm"],
                         stats["train/lr"], s_per, (total - step) * s_per / 3600)
            win = {"loss": 0.0, "tok": 0, "acc": 0.0}
            if t.get("save_steps") and step % t["save_steps"] == 0:
                model.save_pretrained(out / "checkpoints" / f"step_{step}")
    model.save_pretrained(out / "checkpoints" / "final")
    log.info("saved %s", out / "checkpoints" / "final")


if __name__ == "__main__":
    main()
