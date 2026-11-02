"""
caad/trainers/caad.py
=====================
The CAAD training step. V is an image. Per sample, up to four forwards on one
base model (the clean-feature and anchor forwards are skipped when their loss
weight is 0):

  rollout      vLLM (GPU 0)         y ~ pi_student( . | q, V_corrupted )
  student      train GPUs, grad     logits + post-merger visual feats, V_corrupted
  clean feats  train GPUs, no grad  post-merger visual feats, V_clean (L2 target)
  teacher      train GPUs, no grad  EMA-adapter logits on SAME tokens, V_clean
  anchor       train GPUs, no grad  adapter-disabled logits on SAME tokens, V_corr

Loss = gated_divergence + lambda_l2 * visual_L2 + anchor_beta * KL(pi||pi_init)

Cross-input alignment invariant: clean and corrupted prompts must tokenize to
the SAME length so that teacher logits at position t score the student's token
y_t under an identical prefix structure. Corruptions preserve resolution, so
visual token counts match; we assert it anyway.
"""

from __future__ import annotations

import torch

from ..data.vision import GRID_KEY, PIXEL_KEY, build_messages, process_image
from ..eval.tasks.vqa_c import PROMPT_FORMATS
from ..losses.divergence import (anchor_kl_loss, entropy_gate, eopd_loss,
                                 gated_divergence_loss, teacher_entropy,
                                 visual_l2_loss)
from ..models.model_utils import (STUDENT, TEACHER, get_visual_features,
                                  sync_adapter, use_adapter, use_anchor)


def build_inputs(processor, system_prompt, user_text, image_path,
                 completion_text, device, max_pixels=None):
    """
    Tokenize (prompt + completion + end-of-turn) with the given image. Returns
    the batch and a completion mask that is 1 exactly on the completion tokens
    (including the end-of-turn token, so the student also learns where to stop).
    ``system_prompt`` may be None (no system turn, as in evaluation).
    If ``completion_text`` carries the sampled ``token_ids`` (rollout.Completion),
    those are used as is instead of re-tokenizing the text.
    """
    messages = build_messages(image_path, user_text, system_prompt, max_pixels)
    prompt_text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)

    prompt_batch = process_image(processor, prompt_text, messages)
    tok = processor.tokenizer
    vocab = tok.get_vocab()                  # end-of-turn: Qwen / InternVL, Gemma 3, else EOS
    end = next((t for t in ("<|im_end|>", "<end_of_turn>") if t in vocab), tok.eos_token or "")
    end_ids = tok(end, add_special_tokens=False).input_ids
    sampled = getattr(completion_text, "token_ids", None)
    if sampled is not None:
        ids = list(sampled)
        if ids[-len(end_ids):] != end_ids:   # truncated, or the server dropped the stop token
            ids = ids + end_ids
        completion_ids = torch.tensor([ids], dtype=torch.long)
    else:
        completion_ids = tok(completion_text + end, return_tensors="pt",
                             add_special_tokens=False).input_ids

    input_ids = torch.cat([prompt_batch.input_ids, completion_ids], dim=1)
    attn = torch.ones_like(input_ids)
    completion_mask = torch.zeros_like(input_ids)
    plen = prompt_batch.input_ids.shape[1]
    completion_mask[:, plen:] = 1

    batch = {"input_ids": input_ids.to(device), "attention_mask": attn.to(device)}
    for k, v in prompt_batch.items():        # image tensors: Qwen-VL pixel_values + image_grid_thw,
        if k not in batch and torch.is_tensor(v):   # InternVL pixel_values, ...
            if v.dim() == 2 and v.shape[1] == plen and not v.is_floating_point():
                # per-token prompt tensors (Gemma 3 token_type_ids): completion tokens are text (0)
                v = torch.cat([v, torch.zeros_like(completion_ids)], dim=1)
            batch[k] = v.to(device)
    return batch, completion_mask.to(device), plen


def shift_for_next_token(logits, completion_mask):
    """
    Align logits with the tokens they predict: logits[:, t] predicts token t+1.
    Returns (pred_logits, target_mask) both of length L-1.
    """
    return logits[:, :-1], completion_mask[:, 1:]


class CAADTrainer:
    def __init__(self, model, processor, cfg, accelerator,
                 frozen_teacher=None, frozen_init=None):
        self.model = model
        self.processor = processor
        self.cfg = cfg
        self.acc = accelerator
        self.lora = cfg["lora"]["enabled"]
        self.frozen_teacher = frozen_teacher   # full-FT mode, or an external (e.g. larger) teacher
        # external teacher: a separate frozen model sees the clean image; no EMA
        self.external = bool((cfg.get("teacher") or {}).get("name"))
        if self.external and frozen_teacher is None:
            raise ValueError("cfg.teacher is set but no teacher model was passed")
        self.frozen_init = frozen_init
        c = cfg["caad"]
        self.q = c["fkl_quantile"]
        self.min_tok = c["min_tokens_for_quantile"]
        self.jsd_beta = c["jsd_beta"]
        self.lambda_l2 = c["lambda_l2"]
        self.anchor_beta = c["anchor_beta"]
        self.ema_decay = c["ema_decay"]
        self.chunk = c["chunk_size"]
        # distillation objective:
        #   jsd   gated FKL / JSD_beta, per-rollout mean (original CAAD)
        #   eopd  base divergence on every token (caad.eopd.base: jsd | rkl) +
        #         top-k forward KL added on the gated (lowest-entropy-quantile)
        #         tokens; token-mean over the optimizer step (Jin et al., 2026)
        self.loss_type = c.get("loss", "jsd")
        if self.loss_type not in ("jsd", "eopd"):
            raise ValueError(f"caad.loss must be 'jsd' or 'eopd', got {self.loss_type!r}")
        e = c.get("eopd", {}) or {}
        self.topk, self.clip_eps = e.get("topk", 16), e.get("clip_eps", 0.2)
        self.eopd_base = e.get("base", "jsd")
        # low_quantile | high_quantile | high_threshold | low_threshold |
        # low_or_high_threshold | abs_gap_threshold
        self.gate_mode = c.get("fkl_gate", "low_quantile")
        self.tau = c.get("fkl_tau", 0.8)                      # high-entropy threshold
        self.tau_low = c.get("fkl_tau_low", 0.2)              # low-entropy threshold
        self.tau_gap = c.get("fkl_tau_gap", 0.2)              # |H_student - H_teacher| threshold
        self.diagnostics = e.get("diagnostics", False)
        # the loop divides summed gradients by the step's token count when True
        self.token_mean = self.loss_type == "eopd"
        d = cfg["data"]
        # same prompt as evaluation unless the recipe says otherwise
        fmt_system, self.user_tpl = PROMPT_FORMATS[d.get("prompt_format", "inline")]
        self.system = d.get("system_prompt") or fmt_system
        # rollout.token_ids: "log" requests the sampled ids only to measure the mismatch and
        # still trains on the re-tokenized text (the pre-fix behaviour, as a control)
        self.score_sampled = (cfg.get("rollout") or {}).get("token_ids") is True

    # ------------------------------------------------------------------ #
    def step(self, sample, completion_text):
        """
        One sample -> scalar loss (caller handles grad-accum / backward).
        sample: manifest row with question / clean_image / corrupted_image.
        """
        dev = self.acc.device
        mk = lambda image: build_inputs(
            self.processor, self.system, self.user_tpl.format(question=sample["question"]),
            image, completion_text if self.score_sampled else str(completion_text), dev,
            self.cfg["model"].get("max_pixels"))

        corr_batch, corr_cmask, corr_plen = mk(sample["corrupted_image"])
        clean_batch, _, clean_plen = mk(sample["clean_image"])

        # cross-input alignment invariant
        assert corr_plen == clean_plen and \
            corr_batch["input_ids"].shape == clean_batch["input_ids"].shape, (
            "clean/corrupted prompt lengths differ -- teacher/student token "
            "positions would misalign; check corruption preserves resolution")
        # teacher must score the student's actual rollout under the clean view
        clean_batch["input_ids"] = corr_batch["input_ids"].clone()
        clean_batch["attention_mask"] = corr_batch["attention_mask"].clone()
        use_l2, use_anchor_kl = self.lambda_l2 > 0, self.anchor_beta > 0

        # ---- 1) student forward: corrupted view (grad) ----
        with use_adapter(self.model, STUDENT) if self.lora else _null():
            s_logits = self.model(**corr_batch).logits
            s_visual = get_visual_features(
                self.model, corr_batch[PIXEL_KEY], corr_batch[GRID_KEY]) if use_l2 else None

        # ---- 2) clean visual feats (L2 target, stop-grad) -- only if L2 is on ----
        clean_visual = None
        if use_l2:
            with torch.no_grad():
                with use_adapter(self.model, STUDENT) if self.lora else _null():
                    clean_visual = get_visual_features(
                        self.model, clean_batch[PIXEL_KEY], clean_batch[GRID_KEY]).detach()

        # ---- 3) teacher logits: EMA adapter, CLEAN view, same tokens ----
        with torch.no_grad():
            if self.external or not self.lora:
                t_dev = next(self.frozen_teacher.parameters()).device
                if t_dev == dev:
                    t_logits = self.frozen_teacher(**clean_batch).logits
                else:                        # teacher on another GPU (e.g. beside the vLLM server)
                    t_logits = self.frozen_teacher(**{k: v.to(t_dev) if torch.is_tensor(v) else v
                                                      for k, v in clean_batch.items()}).logits.to(dev)
            else:
                with use_adapter(self.model, TEACHER):
                    t_logits = self.model(**clean_batch).logits

        # ---- 4) anchor logits: pi_init, corrupted view -- only if anchor KL is on ----
        i_logits = None
        if use_anchor_kl:
            with torch.no_grad():
                with use_anchor(self.model, self.lora, self.frozen_init) as init_m:
                    i_logits = init_m(**corr_batch).logits

        # teacher and student may pad the vocabulary to different sizes (Qwen2.5-VL 3B 151,936 vs
        # 7B 152,064 rows for the same 151,665-token tokenizer); rows past the shared size are
        # padding no token id maps to, so the divergences compare the shared vocabulary only
        V = min(s_logits.shape[-1], t_logits.shape[-1])
        s_logits, t_logits = s_logits[..., :V], t_logits[..., :V]
        if i_logits is not None:
            i_logits = i_logits[..., :V]

        # ---- losses ----
        s_pred, cmask = shift_for_next_token(s_logits, corr_cmask)
        t_pred, _ = shift_for_next_token(t_logits, corr_cmask)

        H = teacher_entropy(t_pred, chunk=self.chunk)
        H_s = teacher_entropy(s_pred.detach(), chunk=self.chunk)   # student entropy (no grad)
        gate = entropy_gate(H, cmask, self.gate_mode, q=self.q, tau=self.tau,
                            min_tokens=self.min_tok, tau_low=self.tau_low,
                            student_entropy=H_s, tau_gap=self.tau_gap)

        if self.loss_type == "eopd":
            targets = corr_batch["input_ids"][:, 1:]               # token predicted at each position
            distill, n_tok, stats = eopd_loss(
                s_pred, t_pred, targets, cmask, gate,
                topk=self.topk, clip_eps=self.clip_eps, chunk=self.chunk,
                base=self.eopd_base, jsd_beta=self.jsd_beta,
                diag_masks=None if not self.diagnostics else {
                    "all": cmask.bool(),
                    "low10": entropy_gate(H, cmask, "low_quantile", q=0.10, min_tokens=self.min_tok),
                    "high10": entropy_gate(H, cmask, "high_quantile", q=0.10, min_tokens=self.min_tok),
                    "H_gt_0.8": entropy_gate(H, cmask, "high_threshold", tau=0.8),
                    "H_lt_0.2": entropy_gate(H, cmask, "low_threshold", tau_low=0.2),
                    "absgap_gt_0.2": entropy_gate(H, cmask, "abs_gap_threshold",
                                                  student_entropy=H_s, tau_gap=0.2)})
            scale = n_tok.float()          # per-sample terms below are token-weighted to match
        else:
            distill, stats = gated_divergence_loss(
                s_pred, t_pred, cmask, gate,
                jsd_beta=self.jsd_beta, chunk=self.chunk)
            n_tok, scale = cmask.sum(), 1.0
        loss = distill
        l2 = anchor = torch.zeros((), device=dev)
        if use_l2:
            l2 = visual_l2_loss(s_visual, clean_visual)
            loss = loss + self.lambda_l2 * l2 * scale
        if use_anchor_kl:
            i_pred, _ = shift_for_next_token(i_logits, corr_cmask)
            anchor = anchor_kl_loss(s_pred, i_pred, cmask, chunk=self.chunk)
            loss = loss + self.anchor_beta * anchor * scale

        stats.update({
            # logged per token for eopd (its returned loss is a token sum)
            "loss/total": loss.detach() / (n_tok.clamp(min=1) if self.token_mean else 1),
            "loss/n_tokens": n_tok.float(),
            "loss/visual_l2": l2.detach(),
            "loss/anchor_kl": anchor.detach(),
            "entropy/teacher_mean": (H * cmask).sum() / cmask.sum().clamp(min=1),
            "entropy/student_mean": (H_s * cmask).sum() / cmask.sum().clamp(min=1),
            "entropy/abs_gap_mean": ((H_s - H).abs() * cmask).sum() / cmask.sum().clamp(min=1),
            "rollout/scored_tokens": cmask.sum().float(),
            "inputs/prompt_tokens": float(corr_plen),
        })
        sampled = getattr(completion_text, "token_ids", None)
        if sampled is not None:              # how often re-tokenizing the text would have changed the sequence
            retok = self.processor.tokenizer(str(completion_text), add_special_tokens=False).input_ids
            stats["rollout/retok_mismatch"] = float(retok != list(sampled)[:len(retok)] or
                                                    len(sampled) - len(retok) > 1)
        return loss, stats

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def update_ema_teacher(self):
        if self.external:                    # frozen external teacher: nothing to update
            return
        if self.lora:
            sync_adapter(self.acc.unwrap_model(self.model),
                         src=STUDENT, dst=TEACHER, decay=self.ema_decay)
        else:
            student = self.acc.unwrap_model(self.model)
            for p_t, p_s in zip(self.frozen_teacher.parameters(),
                                student.parameters()):
                p_t.mul_(self.ema_decay).add_(p_s.detach(),
                                              alpha=1 - self.ema_decay)


class _null:
    def __enter__(self): return None
    def __exit__(self, *a): return False
