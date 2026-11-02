"""
caad/losses/divergence.py
=========================
Per-token adaptive divergence for CAAD.

Gate (revised proposal):
  - teacher entropy H(T_t) computed PER ROLLOUT over valid completion tokens
  - bottom `fkl_quantile` (default 10%) entropy tokens -> forward KL  D(T||S)
  - all remaining tokens                              -> JSD_0.5(S, T)
  - rollouts shorter than `min_tokens` fall back to the batch-level quantile
    (a per-rollout 10th percentile over e.g. 12 tokens is just the 2nd lowest
    token -- too granular to be meaningful)

All quantities are computed from full-vocab log-probs in CHUNKS along the
sequence axis: with a ~150k vocab, materializing teacher+student+mixture
distributions for a whole rollout at once is the main OOM risk.

Shapes: logits (B, L, V); masks (B, L) with 1 on completion tokens.
Teacher logits must be produced under no_grad upstream; gradients flow only
through student logits.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


@torch.no_grad()
def teacher_entropy(teacher_logits: torch.Tensor, chunk: int = 64) -> torch.Tensor:
    """H(T_t) per position. (B, L, V) -> (B, L), float32."""
    B, L, V = teacher_logits.shape
    out = torch.empty(B, L, dtype=torch.float32, device=teacher_logits.device)
    for s in range(0, L, chunk):
        lt = teacher_logits[:, s:s + chunk].float().log_softmax(-1)
        out[:, s:s + chunk] = -(lt.exp() * lt).sum(-1)
    return out


@torch.no_grad()
def fkl_gate_mask(entropy: torch.Tensor,
                  valid: torch.Tensor,
                  q: float = 0.10,
                  min_tokens: int = 24) -> torch.Tensor:
    """
    Boolean mask (B, L): True where forward-KL should be used.

    Per-rollout quantile threshold; short rollouts use the batch quantile so
    the gate isn't dominated by order statistics of tiny samples.
    """
    B, L = entropy.shape
    valid = valid.bool()
    if q <= 0:                       # gate off: every token uses JSD
        return torch.zeros_like(valid)
    masked = entropy.masked_fill(~valid, float("inf"))

    # batch-level fallback threshold over all valid tokens
    all_valid = entropy[valid]
    batch_thr = (torch.quantile(all_valid.float(), q)
                 if all_valid.numel() > 0 else torch.tensor(0.0, device=entropy.device))

    gate = torch.zeros_like(valid)
    for b in range(B):
        n = int(valid[b].sum())
        if n == 0:
            continue
        if n >= min_tokens:
            thr = torch.quantile(entropy[b][valid[b]].float(), q)
        else:
            thr = batch_thr
        gate[b] = (masked[b] <= thr) & valid[b]
    return gate


def entropy_gate(entropy: torch.Tensor, valid: torch.Tensor, mode: str = "low_quantile",
                 q: float = 0.10, tau: float = 0.8, min_tokens: int = 24,
                 tau_low: float = 0.2, student_entropy: torch.Tensor | None = None,
                 tau_gap: float = 0.2) -> torch.Tensor:
    """
    Which tokens get the extra forward-KL term (``entropy`` = teacher entropy):
      low_quantile          lowest-entropy q of each rollout (original CAAD gate)
      high_quantile         highest-entropy q of each rollout
      high_threshold        H_teacher > tau (EOPD, Jin et al. 2026; tau = 0.8)
      low_threshold         H_teacher < tau_low (teacher confident)
      low_or_high_threshold H_teacher < tau_low  or  H_teacher > tau
      abs_gap_threshold     |H_student - H_teacher| > tau_gap (needs student_entropy)
    """
    valid = valid.bool()
    if mode == "low_threshold":
        return (entropy < tau_low) & valid
    if mode == "low_or_high_threshold":
        return ((entropy < tau_low) | (entropy > tau)) & valid
    if mode == "abs_gap_threshold":
        if student_entropy is None:
            raise ValueError("abs_gap_threshold needs student_entropy")
        return ((student_entropy - entropy).abs() > tau_gap) & valid
    if mode == "low_quantile":
        return fkl_gate_mask(entropy, valid, q=q, min_tokens=min_tokens)
    if mode == "high_quantile":
        return fkl_gate_mask(-entropy, valid, q=q, min_tokens=min_tokens)
    if mode == "high_threshold":
        return (entropy > tau) & valid
    raise ValueError(f"unknown gate mode {mode!r}")


def gated_divergence_loss(student_logits: torch.Tensor,
                          teacher_logits: torch.Tensor,
                          completion_mask: torch.Tensor,
                          fkl_mask: torch.Tensor,
                          jsd_beta: float = 0.5,
                          chunk: int = 64):
    """
    Sum over tokens of  [ FKL(T||S) on gated tokens, JSD_beta(S,T) elsewhere ],
    averaged over valid tokens. Gradients flow through student_logits only.

    Returns (loss, stats_dict).
    """
    B, L, V = student_logits.shape
    valid = completion_mask.bool()
    n_valid = valid.sum().clamp(min=1)

    total = student_logits.new_zeros(())
    fkl_sum = torch.zeros((), device=student_logits.device)
    jsd_sum = torch.zeros((), device=student_logits.device)

    for s in range(0, L, chunk):
        sl = slice(s, s + chunk)
        ls = student_logits[:, sl].float().log_softmax(-1)          # grad
        with torch.no_grad():
            lt = teacher_logits[:, sl].float().log_softmax(-1)       # no grad
        pt = lt.exp()

        v = valid[:, sl]
        g = fkl_mask[:, sl] & v          # forward-KL tokens
        j = v & ~g                       # JSD tokens

        if g.any():
            # D_KL(T || S) = sum_v p_T (log p_T - log p_S)
            fkl_tok = (pt * (lt - ls)).sum(-1)
            fkl_chunk = (fkl_tok * g).sum()
            total = total + fkl_chunk
            fkl_sum += fkl_chunk.detach()

        if j.any():
            # JSD_beta(S,T) = beta KL(S||M) + (1-beta) KL(T||M),
            # M = beta S + (1-beta) T, computed stably in log space.
            log_m = torch.logsumexp(
                torch.stack([ls + torch.log(torch.tensor(jsd_beta)),
                             lt + torch.log(torch.tensor(1 - jsd_beta))], 0), dim=0)
            ps = ls.exp()
            kl_s_m = (ps * (ls - log_m)).sum(-1)
            kl_t_m = (pt * (lt - log_m)).sum(-1)
            jsd_tok = jsd_beta * kl_s_m + (1 - jsd_beta) * kl_t_m
            jsd_chunk = (jsd_tok * j).sum()
            total = total + jsd_chunk
            jsd_sum += jsd_chunk.detach()

    loss = total / n_valid
    stats = {
        "distill/loss": loss.detach(),
        "distill/fkl_frac": (fkl_mask & valid).sum() / n_valid,
        "distill/fkl_mean": fkl_sum / (fkl_mask & valid).sum().clamp(min=1),
        "distill/jsd_mean": jsd_sum / (valid & ~fkl_mask).sum().clamp(min=1),
    }
    return loss, stats


def anchor_kl_loss(student_logits: torch.Tensor,
                   init_logits: torch.Tensor,
                   completion_mask: torch.Tensor,
                   chunk: int = 64) -> torch.Tensor:
    """
    KL( pi_theta || pi_init ) per token, averaged over valid tokens.
    init_logits produced under no_grad (adapter-disabled base / frozen copy).
    """
    B, L, V = student_logits.shape
    valid = completion_mask.bool()
    n_valid = valid.sum().clamp(min=1)
    total = student_logits.new_zeros(())
    for s in range(0, L, chunk):
        sl = slice(s, s + chunk)
        ls = student_logits[:, sl].float().log_softmax(-1)
        with torch.no_grad():
            li = init_logits[:, sl].float().log_softmax(-1)
        kl = (ls.exp() * (ls - li)).sum(-1)
        total = total + (kl * valid[:, sl]).sum()
    return total / n_valid


def visual_l2_loss(student_visual: torch.Tensor,
                   clean_visual: torch.Tensor) -> torch.Tensor:
    """
    L2 alignment on post-merger visual tokens.
      student_visual: (N, D) from CORRUPTED view, WITH grad
      clean_visual:   (N, D) from CLEAN view -- caller must .detach() it
                      (stop-grad on the target prevents the model satisfying
                      the loss by degrading the clean representation)
    Both sides unit-normalized so the loss measures direction, not magnitude.
    """
    assert student_visual.shape == clean_visual.shape, (
        f"visual token mismatch {student_visual.shape} vs {clean_visual.shape}; "
        "corruptions must preserve resolution")
    s = F.normalize(student_visual.float(), dim=-1)
    t = F.normalize(clean_visual.float(), dim=-1)
    return ((s - t) ** 2).sum(-1).mean()


# --------------------------------------------------------------------------- #
# EOPD-style objective (Jin et al., 2026, "Entropy-Aware On-Policy
# Distillation", Eq. 8-10): a base divergence on EVERY token plus a top-k
# forward KL ADDED on gated tokens, token-summed. The gate is supplied by the
# caller; the base term is JSD_beta (default) or EOPD's clipped reverse KL.
# --------------------------------------------------------------------------- #
def eopd_loss(student_logits: torch.Tensor,
              teacher_logits: torch.Tensor,
              targets: torch.Tensor,
              completion_mask: torch.Tensor,
              fkl_mask: torch.Tensor,
              topk: int = 16,
              clip_eps: float = 0.2,
              chunk: int = 64,
              base: str = "jsd",
              jsd_beta: float = 0.5,
              diag_masks: dict | None = None):
    """
    Per completion token t (sampled token x_t = targets[:, t]):

      base="jsd", every token (full vocabulary)
        L_t^base = JSD_beta(S, T) = beta KL(S||M) + (1-beta) KL(T||M),
        M = beta S + (1-beta) T
      base="rkl", every token (EOPD's clipped policy-gradient form, Eq. 6-8)
        A_t = log pi_T(x_t) - log pi_old(x_t)                       (detached)
        r_t = pi_theta(x_t) / pi_old(x_t)
        L_t^RKL = max(-r_t A_t, -clip(r_t, 1-eps, 1+eps) A_t)
      base="rkl_full", every token (exact reverse KL over the full vocabulary)
        L_t^RKL = sum_x pi_theta(x) [log pi_theta(x) - log pi_T(x)]
        Unlike "rkl", tokens the student rarely samples are penalized directly: with the
        1-sample form, Gemma 3 4B grew a diffuse tail the teacher gives ~0 probability
        (exact KL 0.5 -> 3.8 while the sampled estimate read ~0) and emitted garbage.
      forward KL, ADDED on gated tokens (top-k approximation, Eq. 10)
        L_t^FKL = sum_{x in top-k(pi_T)} pt~(x) [log pt~(x) - log pi_theta(x)]
        pt~ = teacher renormalized over its top-k; pi_theta is NOT renormalized
      L_t = L_t^base + 1[gate_t] L_t^FKL

    pi_old is the student at the time of this forward (detached), as when the
    behaviour policy's log-probs are recomputed by the trainer; each rollout is
    used for one gradient contribution, so r_t == 1 in value and the clip is
    inactive, but the gradient of -r_t A_t is -A_t * grad log pi_theta(x_t).

    ``diag_masks`` ({name: bool mask}) only adds monitoring stats: the top-k
    forward KL averaged over each token set, without gradient.

    Returns (loss_SUM over valid tokens, n_valid_tokens, stats). The caller
    divides the summed gradient by the total token count of the optimizer
    step, i.e. EOPD's token-mean over the mini-batch.
    """
    if base not in ("jsd", "rkl", "rkl_full"):
        raise ValueError(f"base must be 'jsd', 'rkl' or 'rkl_full', got {base!r}")
    B, L, V = student_logits.shape
    valid = completion_mask.bool()
    gate = fkl_mask.bool() & valid
    n_valid = valid.sum()
    log_b, log_1mb = math.log(jsd_beta), math.log(1 - jsd_beta)
    total = student_logits.new_zeros(())
    acc = {k: torch.zeros((), device=student_logits.device) for k in
           ("base", "rkl_sample", "rkl_exact", "fkl", "topk_mass", "clipped", "adv")}
    diag_masks = {k: m.bool() & valid for k, m in (diag_masks or {}).items()}
    diag = {k: torch.zeros((), device=student_logits.device) for k in diag_masks}

    for s in range(0, L, chunk):
        sl = slice(s, s + chunk)
        v = valid[:, sl]
        if not v.any():
            continue
        ls = student_logits[:, sl].float().log_softmax(-1)            # grad
        with torch.no_grad():
            lt = teacher_logits[:, sl].float().log_softmax(-1)
        tgt = targets[:, sl].unsqueeze(-1)
        lp_s = ls.gather(-1, tgt).squeeze(-1)                          # log pi_theta(x_t)
        lp_t = lt.gather(-1, tgt).squeeze(-1)                          # log pi_T(x_t)
        lp_old = lp_s.detach()
        adv = lp_t - lp_old                                            # detached
        ratio = torch.exp(lp_s - lp_old)
        if base == "rkl":
            base_tok = torch.max(-ratio * adv, -ratio.clamp(1 - clip_eps, 1 + clip_eps) * adv)
        elif base == "rkl_full":
            base_tok = (ls.exp() * (ls - lt)).sum(-1)
        else:
            log_m = torch.logsumexp(torch.stack([ls + log_b, lt + log_1mb]), dim=0)
            base_tok = (jsd_beta * (ls.exp() * (ls - log_m)).sum(-1)
                        + (1 - jsd_beta) * (lt.exp() * (lt - log_m)).sum(-1))
        total = total + (base_tok * v).sum()

        g = gate[:, sl]
        if g.any():
            top_lt, top_idx = lt.topk(topk, dim=-1)                    # teacher top-k
            log_pt = top_lt.log_softmax(-1)                            # renormalized over top-k
            fkl_tok = (log_pt.exp() * (log_pt - ls.gather(-1, top_idx))).sum(-1)
            total = total + (fkl_tok * g).sum()
            acc["fkl"] += (fkl_tok.detach() * g).sum()
            acc["topk_mass"] += (top_lt.exp().sum(-1) * g).sum()

        with torch.no_grad():
            if diag_masks:
                top_all, idx_all = lt.topk(topk, dim=-1)
                lp_all = top_all.log_softmax(-1)
                fkl_all = (lp_all.exp() * (lp_all - ls.detach().gather(-1, idx_all))).sum(-1)
                for k, m in diag_masks.items():
                    diag[k] += (fkl_all * m[:, sl]).sum()
            acc["base"] += (base_tok.detach() * v).sum()
            acc["rkl_sample"] += ((lp_old - lp_t) * v).sum()           # 1-sample reverse-KL estimate
            acc["rkl_exact"] += ((ls.detach().exp() * (ls.detach() - lt)).sum(-1) * v).sum()
            acc["adv"] += (adv * v).sum()
            acc["clipped"] += (((ratio.detach() - 1).abs() > clip_eps) & v).sum()

    nv = n_valid.clamp(min=1)
    ng = gate.sum().clamp(min=1)
    stats = {
        "distill/loss": (total.detach() / nv),
        "distill/fkl_frac": gate.sum() / nv,
        f"eopd/{base}_mean": acc["base"] / nv,
        "eopd/rkl_sample_estimate": acc["rkl_sample"] / nv,
        "eopd/rkl_exact": acc["rkl_exact"] / nv,
        "eopd/advantage_mean": acc["adv"] / nv,
        "eopd/clip_frac": acc["clipped"] / nv,
        "eopd/fkl_topk_mean": acc["fkl"] / ng,
        "eopd/teacher_topk_mass": acc["topk_mass"] / ng,
    }
    for k, m in diag_masks.items():
        stats[f"diag/fkl_{k}"] = diag[k] / m.sum().clamp(min=1)
        stats[f"diag/frac_{k}"] = m.sum() / nv
    return total, n_valid, stats
