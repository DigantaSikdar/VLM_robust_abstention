"""
caad/train.py
=============
Training entrypoint. ONE command, everything from the YAML:

    accelerate launch -m caad.train --config configs/train/caad_lora_qwen25vl7b.yaml

The script: resolves the config (self-contained recipe + CLI overrides), creates
the recipe's output_dir tree, freezes the exact config there, constructs the
model / rollout engine / data / trainer, and hands off to TrainLoop. A new
experiment is a new YAML, never an edit here.
"""

from __future__ import annotations

import argparse

from .trainers.base import TrainLoop
from .trainers.caad import CAADTrainer
from .trainers.grpo import GRPOTrainer
from .utils.config import freeze_config, load_config
from .utils.logging import MetricsLogger, run_dir, setup_logging
from .utils.seeding import seed_everything

# method -> trainer class. Add a variant here, select it via the recipe's `method`.
TRAINERS = {"caad": CAADTrainer, "grpo": GRPOTrainer}


def parse_args():
    p = argparse.ArgumentParser(description="CAAD training")
    p.add_argument("--config", required=True, help="path to a self-contained train recipe (YAML)")
    p.add_argument("--output_dir", default=None,
                   help="override the recipe's output_dir (where this run is written)")
    p.add_argument("overrides", nargs="*", help="dotted overrides: caad.lambda_l2=0.5")
    return p.parse_args()


def _total_optimizer_steps(cfg, dataset_len):
    """Total optimizer steps: train.max_steps if set, else derived from
    num_train_epochs over the dataset."""
    t = cfg["train"]
    if t.get("max_steps", 0) and t["max_steps"] > 0:
        return t["max_steps"]
    accum = t.get("per_device_train_batch_size", 1) * t["gradient_accumulation_steps"]
    per_epoch = max(1, -(-dataset_len // accum))      # ceil(dataset_len / accum)
    return t.get("num_train_epochs", 1) * per_epoch


def _build_scheduler(optimizer, cfg, total_steps):
    """LR schedule from config, warmup_ratio of total_steps as warmup.
    Honors train.lr_scheduler_type."""
    from transformers import get_scheduler
    t = cfg["train"]
    warmup = int(t.get("warmup_ratio", 0.0) * total_steps)
    return get_scheduler(t.get("lr_scheduler_type", "cosine"),
                         optimizer=optimizer, num_warmup_steps=warmup,
                         num_training_steps=total_steps)


def build_run(args):
    cfg = load_config(args.config, overrides=args.overrides)
    # output location is the recipe's output_dir (override with --output_dir).
    output_dir = args.output_dir or cfg["output_dir"]
    path = run_dir(output_dir)                 # creates checkpoints/ results/ plots/ logs/
    run_id = path.name                         # run identity for logs / W&B fallback
    freeze_config(cfg, path)
    return cfg, run_id, path


def main():
    import logging
    args = parse_args()
    cfg, run_id, path = build_run(args)
    level = getattr(logging, str(cfg.get("log_level", "info")).upper(), logging.INFO)
    log = setup_logging(path, level=level)
    log.info("run_id=%s  ->  %s", run_id, path)
    seed_everything(cfg["seed"], deterministic=cfg.get("deterministic", False))

    # --- heavy deps imported lazily so --help / config resolution stay fast ---
    from accelerate import Accelerator
    from torch.optim import AdamW
    from torch.utils.data import DataLoader

    from .data.dataset import ManifestDataset
    from .models.model_utils import external_teacher, load_base_model, load_teacher
    from .rollout import RolloutEngine  # vLLM wrapper; see module docstring

    acc = Accelerator(gradient_accumulation_steps=cfg["train"]["gradient_accumulation_steps"])
    model, processor = load_base_model(cfg)
    if cfg["lora"].get("init_student_from"):
        # start the student adapter from a saved one (e.g. the SFT cold start)
        from .models.model_utils import load_student_init
        load_student_init(model, cfg["lora"]["init_student_from"])
        log.info("student adapter initialized from %s", cfg["lora"]["init_student_from"])
    if cfg["model"].get("gradient_checkpointing"):
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs=cfg["model"].get(
                "gradient_checkpointing_kwargs", {"use_reentrant": False}))
        # HF only checkpoints in train mode; from_pretrained returns eval mode.
        # Dropout is 0 for Qwen3-VL and the LoRA adapters, so this changes memory
        # use only, not the computation.
        model.train()
    dataset = ManifestDataset(cfg["data"]["manifest"],
                              image_root=cfg["data"].get("image_root"))
    import torch
    loader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=lambda b: b[0],
                        generator=torch.Generator().manual_seed(cfg["seed"]))
    own_opt = cfg["train"].get("optimizer", "adamw") == "bf16_adamw"
    if own_opt:
        # full FT: fp32 grads/moments, stochastic rounding. Built on the training
        # device and NOT passed through acc.prepare: AcceleratedOptimizer reloads the
        # state dict, which copies every state buffer and casts it to the param dtype
        # (bf16), i.e. doubles the optimizer memory and drops its fp32 precision.
        from .utils.optim import BF16AdamW
        model.to(acc.device)
        opt = BF16AdamW([p for p in model.parameters() if p.requires_grad],
                        lr=cfg["train"]["learning_rate"],
                        weight_decay=cfg["train"].get("weight_decay", 0.01))
    else:
        opt = AdamW((p for p in model.parameters() if p.requires_grad),
                    lr=cfg["train"]["learning_rate"])
    sched = _build_scheduler(opt, cfg, _total_optimizer_steps(cfg, len(dataset)))
    resume = cfg["train"].get("resume_step", 0)
    if resume:
        # continue a run from its step-`resume` checkpoint (model.name): replay the seeded
        # shuffle, drop the samples the first `resume` optimizer steps consumed, and move the
        # LR schedule to step `resume`. The optimizer moments start fresh (checkpoints hold
        # weights only).
        import warnings
        from torch.utils.data import Subset
        accum = cfg["train"].get("per_device_train_batch_size", 1) * cfg["train"]["gradient_accumulation_steps"]
        # replay the original run's loader exactly (accelerate-prepared, seeded); a bare
        # RandomSampler with the same seed gives a different order
        index = {r["question_id"]: i for i, r in enumerate(dataset.rows)}
        order = [index[x["question_id"]] for x in acc.prepare(DataLoader(
            dataset, batch_size=1, shuffle=True, collate_fn=lambda b: b[0],
            generator=torch.Generator().manual_seed(cfg["seed"])))]
        loader = DataLoader(Subset(dataset, order[resume * accum:]), batch_size=1, shuffle=False,
                            collate_fn=lambda b: b[0])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")              # "scheduler.step() before optimizer.step()"
            for _ in range(resume):
                sched.step()
        log.info("resuming at step %d from %s: skipped %d seen samples, %d remain, lr %.3e",
                 resume, cfg["model"]["name"], resume * accum, len(order) - resume * accum,
                 sched.get_last_lr()[0])
    if own_opt:
        model, loader = acc.prepare(model, loader)
    else:
        model, opt, loader = acc.prepare(model, opt, loader)

    if cfg["method"] not in TRAINERS:
        raise SystemExit(f"unknown method '{cfg['method']}'; known: {sorted(TRAINERS)}")
    teacher = None
    if external_teacher(cfg):
        teacher = load_teacher(cfg, cfg["teacher"].get("device") or acc.device)
        log.info("external frozen teacher: %s (%.2fB params)", cfg["teacher"]["name"],
                 sum(p.numel() for p in teacher.parameters()) / 1e9)
    ref = None
    g = cfg.get("grpo") or {}
    if cfg["method"] == "grpo" and g.get("kl_beta", 0) > 0:
        # frozen copy of the starting policy for GRPO's KL penalty (same loader as a teacher)
        ref_name = g.get("ref_model") or cfg["model"]["name"]    # set when resuming from a checkpoint
        ref = load_teacher({"teacher": {"name": ref_name, "dtype": cfg["model"].get("dtype", "bfloat16"),
                                        "attn_implementation": cfg["model"].get("attn_implementation", "sdpa")}},
                           g.get("ref_device") or acc.device)
        log.info("GRPO KL reference: frozen %s on %s, beta=%g", ref_name,
                 g.get("ref_device") or acc.device, g["kl_beta"])
    trainer = TRAINERS[cfg["method"]](model, processor, cfg, acc, frozen_teacher=teacher, frozen_init=ref)
    rollout = RolloutEngine(cfg, processor, model=model)
    metrics = MetricsLogger(path, wandb_cfg=cfg.get("wandb"), full_config=cfg)

    loop = TrainLoop(trainer, rollout, loader, opt, acc, cfg, path,
                     scheduler=sched, metrics_logger=metrics)
    loop.total_steps = _total_optimizer_steps(cfg, len(dataset))
    loop.start_step = resume
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    dtypes = sorted({str(p.dtype) for p in model.parameters() if p.requires_grad})
    log.info("dataset=%d samples  total_optimizer_steps=%d  trainable_params=%.2fM %s",
             len(dataset), loop.total_steps, n_train / 1e6, dtypes)
    try:
        loop.run()
    finally:
        metrics.close()
    log.info("done: %s", run_id)


if __name__ == "__main__":
    main()
