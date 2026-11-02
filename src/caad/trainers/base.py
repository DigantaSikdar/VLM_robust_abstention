"""
caad/trainers/base.py
=====================
Shared training scaffolding: the rollout -> step -> backward -> EMA loop that is
common to CAAD and any future distillation variant. Algorithm-specific behavior
lives in a per-sample ``step(sample, completion_text) -> (loss, stats)`` method
(see CAADTrainer); this loop owns batching/grad-accum, optimization, LR
scheduling, gradient clipping, EMA cadence, logging, checkpointing, optional HF
hub push, and an in-training eval hook — all driven from the recipe so nothing
is hard-coded.

Counting convention (matches TRL/HF): ``max_steps``, ``save_steps``,
``ema_every``, ``logging_steps`` and ``eval.eval_steps`` are all in **optimizer
steps**, not samples.

Effective batch per optimizer step:
    per_device_train_batch_size x gradient_accumulation_steps
        x num_generations x num_train_gpus

VLM samples have variable visual-token counts, so we accumulate
sequentially (one sample's forwards at a time) rather than padding a true device
batch; ``per_device_train_batch_size`` folds into the accumulation count
alongside ``gradient_accumulation_steps``. ``num_generations`` is the number of
student completions sampled per (image, question) — each is a backward.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from pathlib import Path

_LOG = logging.getLogger("caad")


class TrainLoop:
    def __init__(self, step_trainer, rollout_engine, dataloader, optimizer,
                 accelerator, cfg, run_path, scheduler=None, metrics_logger=None,
                 evaluator=None):
        self.step_trainer = step_trainer          # .step() and .update_ema_teacher()
        self.rollout = rollout_engine             # .generate(sample) -> list[str]
        self.loader = dataloader
        self.opt = optimizer
        self.sched = scheduler
        self.acc = accelerator
        self.cfg = cfg
        self.run_path = Path(run_path)
        self.metrics = metrics_logger
        self.evaluator = evaluator                # callable(model, cfg) -> dict, or None (seam)
        t = cfg["train"]
        self.per_device_bs = t.get("per_device_train_batch_size", 1)
        self.grad_accum = t["gradient_accumulation_steps"]
        self.accum_samples = self.per_device_bs * self.grad_accum
        self.max_steps = t.get("max_steps", 0)            # optimizer steps; <=0 -> use epochs
        # stop early WITHOUT changing the LR schedule (which still spans max_steps /
        # the full epochs) -- for comparing against an intermediate checkpoint
        self.stop_after = t.get("stop_after_steps", 0)
        self.num_train_epochs = t.get("num_train_epochs", 1)
        self.ema_every = t["ema_every"]
        self.save_steps = t["save_steps"]
        self.logging_steps = t.get("logging_steps", 10)
        self.logging_first_step = t.get("logging_first_step", False)
        self.max_grad_norm = t.get("max_grad_norm", 0.0)
        self.save_total_limit = t.get("save_total_limit", 0)
        self.rollout_sync_every = (cfg.get("rollout", {}) or {}).get("sync_every", 0)
        self.hub = cfg.get("hub", {}) or {}
        ev = cfg.get("eval", {}) or {}
        self.do_eval = ev.get("do_eval", False)
        self.eval_steps = ev.get("eval_steps", 0)
        wb = cfg.get("wandb", {}) or {}
        self.log_completions = wb.get("log_completions", False)
        self.table_every = wb.get("rollout_table_every", 0)      # opt steps; 0 = off
        self.total_steps = None                                  # set by caller for ETA
        self.start_step = 0                                      # set by caller when resuming

    @staticmethod
    def _rollout_stats(sample, completions):
        """Rollout quality vs the gold answer (monitoring only; not in the loss)."""
        from ..eval.tasks.vqa_c import GQA, format_check
        out = []
        for c in completions:
            f = format_check(c)
            ok = bool(sample.get("answer")) and GQA.score({"gold": sample["answer"]}, c)
            out.append({"rollout/answer_tag_rate": float(f["answer"]),
                        "rollout/accuracy_vs_gold": float(ok),
                        "rollout/chars": float(len(c))})
        return out

    def run(self):
        import torch
        micro = 0          # samples since the last optimizer update
        opt_step = self.start_step   # optimizer steps taken (including a resumed run's)
        epoch = 0
        done = False
        window = defaultdict(list)                  # stats averaged over one optimizer step
        token_mean = getattr(self.step_trainer, "token_mean", False)
        step_tokens = 0.0
        table_rows = []
        t_run = time.time()
        # make the rollout policy match the starting weights before step 1
        self.rollout.sync_weights(self.step_trainer.model, self.acc)
        self.opt.zero_grad()
        while not done:
            for sample in self.loader:
                t0 = time.time()
                completions = self.rollout.generate(sample)    # list, len = num_generations
                t1 = time.time()
                if hasattr(self.step_trainer, "prepare_group"):  # e.g. GRPO: group rewards/advantages
                    self.step_trainer.prepare_group(sample, completions)
                g = max(len(completions), 1)
                for completion, rs in zip(completions, self._rollout_stats(sample, completions)):
                    loss, stats = self.step_trainer.step(sample, completion)
                    if token_mean:
                        # loss is a token SUM; gradients are divided by the step's
                        # total token count before the update (token-mean)
                        self.acc.backward(loss)
                        step_tokens += float(stats["loss/n_tokens"])
                    else:
                        self.acc.backward(loss / (self.accum_samples * g))
                    for k, v in {**stats, **rs}.items():
                        window[k].append(float(v))
                    if self.table_every and len(table_rows) < 4:
                        table_rows.append((sample, completion, rs))
                window["time/rollout_s"].append(t1 - t0)
                window["time/train_s"].append(time.time() - t1)
                micro += 1
                if micro % self.accum_samples != 0:
                    continue

                # ---- optimizer update ----
                inner = getattr(self.opt, "optimizer", self.opt)   # BF16AdamW keeps grads in its own fp32 buffers
                own_grads = hasattr(inner, "clip_grad_norm")
                if token_mean:
                    inv = 1.0 / max(step_tokens, 1.0)
                    if own_grads:
                        inner.scale_grads(inv)
                    else:
                        for prm in self.step_trainer.model.parameters():
                            if prm.grad is not None:
                                prm.grad.mul_(inv)
                    window["loss/step_tokens"].append(step_tokens)
                    step_tokens = 0.0
                grad_norm = None
                if self.max_grad_norm and self.max_grad_norm > 0:
                    grad_norm = (inner.clip_grad_norm(self.max_grad_norm) if own_grads else
                                 self.acc.clip_grad_norm_(self.step_trainer.model.parameters(),
                                                          self.max_grad_norm))
                self.opt.step()
                if self.sched is not None:
                    self.sched.step()
                self.opt.zero_grad()
                opt_step += 1

                if opt_step % self.ema_every == 0:
                    self.step_trainer.update_ema_teacher()
                if self.rollout_sync_every and opt_step % self.rollout_sync_every == 0:
                    self.rollout.sync_weights(self.step_trainer.model, self.acc)

                stats = {k: sum(v) / len(v) for k, v in window.items()}
                if grad_norm is not None:
                    stats["train/grad_norm"] = float(grad_norm)
                el = time.time() - t_run
                stats.update({"train/epoch": epoch + micro / max(len(self.loader), 1),
                              "train/samples_seen": micro + epoch * len(self.loader),
                              "time/elapsed_h": el / 3600,
                              "time/s_per_step": el / (opt_step - self.start_step)})
                end = self.stop_after or self.total_steps        # where this run actually stops
                if end:
                    stats["time/eta_h"] = (end - opt_step) * el / (opt_step - self.start_step) / 3600
                if torch.cuda.is_available():
                    stats["gpu/max_mem_gb"] = torch.cuda.max_memory_allocated() / 2 ** 30
                self._maybe_log(opt_step, stats, completions[-1] if completions else "")
                if self.table_every and opt_step % self.table_every == 0:
                    self._log_table(opt_step, table_rows)
                window.clear()
                table_rows = []
                if self.do_eval and self.eval_steps > 0 and opt_step % self.eval_steps == 0:
                    self._run_eval(opt_step)
                if opt_step % self.save_steps == 0:
                    self.save_checkpoint(opt_step)
                if (self.max_steps > 0 and opt_step >= self.max_steps) or \
                        (self.stop_after and opt_step >= self.stop_after):
                    done = True
                    break

            epoch += 1
            if self.max_steps <= 0 and epoch >= self.num_train_epochs:
                done = True

        self.save_checkpoint(opt_step, final=True)

    def _log_table(self, opt_step, rows):
        if self.metrics is None or not self.acc.is_main_process or not rows:
            return
        self.metrics.log_rollouts(opt_step, [
            {"question": smp["question"], "gold": smp.get("answer", ""),
             "corruption": f'{smp["corruption"]["style"]}@{smp["corruption"]["severity"]}',
             "correct": bool(rs["rollout/accuracy_vs_gold"]),
             "answer_tag": bool(rs["rollout/answer_tag_rate"]),
             "completion": comp[:2000], "image": smp["corrupted_image"]}
            for smp, comp, rs in rows])

    # ------------------------------------------------------------------ #
    def _maybe_log(self, opt_step, stats, completion):
        if self.metrics is None or not self.acc.is_main_process:
            return
        first = self.logging_first_step and opt_step == 1
        if not first and opt_step % self.logging_steps != 0:
            return
        lr = self.sched.get_last_lr()[0] if self.sched is not None else None
        # one human-readable line per logged step: lands in the job log, logs/train.log
        # and the W&B run's "Logs" tab (wandb captures the console)
        total = self.stop_after or self.total_steps or 0
        parts = [f"step {opt_step}" + (f"/{total}" if total else ""),
                 f"loss {stats.get('loss/total', float('nan')):.5f}"]
        for key, label in (("eopd/jsd_mean", "jsd"), ("eopd/rkl_mean", "rkl_pg"),
                           ("eopd/rkl_full_mean", "rkl_full"),
                           ("eopd/rkl_sample_estimate", "rkl_sample"),
                           ("eopd/rkl_exact", "rkl_exact"), ("eopd/fkl_topk_mean", "fkl_gated"),
                           ("distill/fkl_frac", "fkl_frac"), ("distill/jsd_mean", "jsd")):
            if key in stats and label not in " ".join(parts):
                parts.append(f"{label} {stats[key]:.5f}" if label != "fkl_frac" else f"{label} {stats[key]:.3f}")
        parts += [f"rollout_acc {stats.get('rollout/accuracy_vs_gold', float('nan')):.3f}",
                  f"answer_tag {stats.get('rollout/answer_tag_rate', float('nan')):.3f}",
                  *([f"retok_mismatch {stats['rollout/retok_mismatch']:.3f}"]
                    if "rollout/retok_mismatch" in stats else []),
                  f"grad {stats.get('train/grad_norm', float('nan')):.4f}",
                  f"lr {lr:.2e}" if lr is not None else "lr -",
                  f"{stats.get('time/s_per_step', float('nan')):.1f}s/step"]
        if total and "time/s_per_step" in stats:
            parts.append(f"eta {(total - opt_step) * stats['time/s_per_step'] / 3600:.2f}h")
        _LOG.info(" | ".join(parts))
        row = {**stats, "lr": lr, "epoch_step": opt_step}
        if self.log_completions:
            row["rollout/text"] = completion[:1000]
        self.metrics.log(opt_step, row)

    def _run_eval(self, opt_step):
        """In-training eval hook. The standalone harness (caad.eval) is the main
        path; this fires only if an evaluator was injected and do_eval is set."""
        if self.evaluator is None:
            _LOG.warning("eval.do_eval is set but no evaluator was injected; "
                         "skipping (use the standalone caad.eval harness)")
            return
        if not self.acc.is_main_process:
            return
        res = self.evaluator(self.acc.unwrap_model(self.step_trainer.model), self.cfg)
        if self.metrics is not None:
            self.metrics.log(opt_step, {f"eval/{k}": v for k, v in res.items()})

    # ------------------------------------------------------------------ #
    def save_checkpoint(self, opt_step, final=False):
        if not self.acc.is_main_process:
            return
        name = "final" if final else f"step_{opt_step}"
        out = self.run_path / "checkpoints" / name
        out.mkdir(parents=True, exist_ok=True)
        self.acc.unwrap_model(self.step_trainer.model).save_pretrained(out)
        if not self.cfg["lora"]["enabled"] and hasattr(self.step_trainer, "processor"):
            self.step_trainer.processor.save_pretrained(out)   # full-FT checkpoint loads standalone (vLLM eval)
        if not final:
            self._prune_checkpoints()
        self._maybe_push_hub(out, opt_step, final)

    def _maybe_push_hub(self, out, opt_step, final):
        if not self.hub.get("enabled"):
            return
        every = self.hub.get("push_every_n_steps", 0)
        if not final and (every <= 0 or opt_step % every != 0):
            return
        repo = self.hub.get("model_id")
        if not repo:
            _LOG.warning("hub.enabled but hub.model_id is empty; skipping push")
            return
        try:
            from huggingface_hub import HfApi
            api = HfApi()
            api.create_repo(repo, private=self.hub.get("private", True),
                            exist_ok=True)
            api.upload_folder(folder_path=str(out), repo_id=repo,
                              path_in_repo=out.name)
            _LOG.info("pushed %s -> hf.co/%s", out.name, repo)
        except Exception as e:                       # best-effort; never crash training
            _LOG.warning("hub push failed (%s): %s", out.name, e)

    def _prune_checkpoints(self):
        """Keep only the most recent ``save_total_limit`` step checkpoints."""
        if not self.save_total_limit or self.save_total_limit <= 0:
            return
        ckpts = sorted((self.run_path / "checkpoints").glob("step_*"),
                       key=lambda p: int(p.name.split("_")[1]))
        for old in ckpts[:-self.save_total_limit]:
            for f in sorted(old.rglob("*"), reverse=True):
                f.unlink() if f.is_file() else f.rmdir()
            old.rmdir()
