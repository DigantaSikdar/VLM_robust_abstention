"""
caad/trainers/grpo.py
=====================
GRPO baseline (Shao et al., 2024) on the same data / LoRA / rollout setup as
CAAD, for comparison with self-distillation. No teacher, no clean image: the
student answers from the CORRUPTED image and is rewarded against the gold
answer.

Per question, the loop samples G completions (rollout.num_generations), then:
  reward_i    = 1[extracted answer == gold (GQA exact match)] + format_reward * 1[<answer> tags]
  advantage_i = (reward_i - mean_G) / (std_G + eps)          (std_normalize: false -> no division)
  loss        = sum_t max(-r_t A_i, -clip(r_t, 1-eps, 1+eps) A_i)   over completion tokens
with r_t = pi_theta / pi_old, pi_old = the student at the time of the forward
(one update per rollout, as in our distillation runs, so r_t == 1 in value).
No KL penalty to the base model by default (beta = 0, as in DAPO / Dr. GRPO).
Returned as a token SUM; the loop divides gradients by the step's token count
(token-mean), the same aggregation as the reverse-KL distillation runs.

Full fine-tuning needs more guard rails (both full-FT runs without them collapsed
into "answer + junk" / repetition to max length, which the lenient reward still
paid for). Opt-in via the recipe:
  grpo.strict_reward  correctness counts only for a WELL-FORMED response:
                        answer_format tags  -> exactly one <answer>...</answer> block and
                                               the response ends right after it (a truncated
                                               response never does); format reward likewise
                        answer_format plain -> one short line (<= max_answer_words words)
  grpo.min_reasoning_words  answer_format tags: a well-formed response also needs at least
                      this many words of reasoning before <answer> (<think> tags not
                      counted, so plain-text reasoning qualifies). 0 (default) = off. Keeps
                      GRPO from collapsing to bare answers on backbones that often skip
                      reasoning (InternVL3.5-4B: 66% of base answers have < 15 words)
  rollout.stop        e.g. ["</answer>"]: rollouts end at the answer (no junk after it)
  grpo.kl_beta        per-token KL to a frozen copy of the starting policy (k3 estimator,
                      DeepSeekMath GRPO), on every completion token of informative groups
                      (zero-advantage groups are still skipped, as before)
"""

from __future__ import annotations

import re

import torch

from ..eval.tasks.vqa_c import GQA, PROMPT_FORMATS, format_check
from .caad import build_inputs, shift_for_next_token


_TAGS = re.compile(r"<answer>.*?</answer>", re.S | re.I)


def group_advantages(rewards, std_normalize=True, eps=1e-6):
    r = torch.tensor(rewards, dtype=torch.float32)
    a = r - r.mean()
    if std_normalize:
        a = a / (r.std(unbiased=False) + eps)
    return a.tolist()


class GRPOTrainer:
    token_mean = True                       # the loop normalizes by the step's token count

    def __init__(self, model, processor, cfg, accelerator, frozen_teacher=None, frozen_init=None):
        self.model, self.processor, self.cfg, self.acc = model, processor, cfg, accelerator
        g = cfg.get("grpo", {}) or {}
        self.format_reward = g.get("format_reward", 0.1)
        self.clip_eps = g.get("clip_eps", 0.2)
        self.std_normalize = g.get("std_normalize", True)
        self.strict = g.get("strict_reward", False)
        self.answer_format = g.get("answer_format", "tags")
        self.max_words = g.get("max_answer_words", 8)
        self.min_reason = g.get("min_reasoning_words", 0)
        self.kl_beta = g.get("kl_beta", 0.0)
        self.ref = frozen_init                       # frozen starting policy (train.py), if kl_beta > 0
        self.chunk = (cfg.get("caad") or {}).get("chunk_size", 64)
        d = cfg["data"]
        fmt_system, self.user_tpl = PROMPT_FORMATS[d.get("prompt_format", "inline")]
        self.system = d.get("system_prompt") or fmt_system
        self._adv, self._rew, self._i = [], [], 0

    @staticmethod
    def reasoning_words(completion):
        """Words before the first <answer> tag, <think> / </think> tags removed."""
        pre = re.split(r"<answer>", completion, maxsplit=1, flags=re.I)[0]
        return len(re.sub(r"</?think>", " ", pre, flags=re.I).split())

    def well_formed(self, completion):
        t = completion.strip()
        if self.answer_format == "plain":
            return bool(t) and "\n" not in t and len(t.split()) <= self.max_words
        return (len(_TAGS.findall(t)) == 1 and t.endswith("</answer>")
                and self.reasoning_words(t) >= self.min_reason)

    def reward(self, sample, completion):
        correct = bool(sample.get("answer")) and GQA.score({"gold": sample["answer"]}, completion)
        if not self.strict:
            return float(correct) + self.format_reward * float(format_check(completion)["answer"])
        ok = self.well_formed(completion)
        return float(ok and correct) + self.format_reward * float(ok)

    def prepare_group(self, sample, completions):
        """Called once per question, before the per-completion steps."""
        self._rew = [self.reward(sample, c) for c in completions]
        self._adv = group_advantages(self._rew, self.std_normalize)
        self._i = 0

    def step(self, sample, completion_text):
        adv = self._adv[self._i]
        rew = self._rew[self._i]
        self._i += 1
        dev = self.acc.device
        batch, cmask, plen = build_inputs(
            self.processor, self.system, self.user_tpl.format(question=sample["question"]),
            sample["corrupted_image"], completion_text, dev, self.cfg["model"].get("max_pixels"))
        mask = cmask[:, 1:].bool()
        n_tok = mask.sum()
        stats = {"grpo/reward": torch.tensor(rew), "grpo/advantage_abs": torch.tensor(abs(adv)),
                 "grpo/well_formed": torch.tensor(float(self.well_formed(completion_text))),
                 "grpo/zero_advantage": torch.tensor(float(adv == 0.0)),
                 "grpo/reasoning_words": torch.tensor(float(self.reasoning_words(completion_text))),
                 "loss/n_tokens": n_tok.float(), "inputs/prompt_tokens": float(plen)}
        if adv == 0.0:                       # no learning signal: skip the forward pass
            z = torch.zeros((), device=dev, requires_grad=True)
            stats["loss/total"] = torch.zeros(())
            return z, stats
        logits = self.model(**batch).logits
        pred, _ = shift_for_next_token(logits, cmask)
        targets = batch["input_ids"][:, 1:]
        ref_pred = None
        if self.kl_beta > 0 and self.ref is not None:
            with torch.no_grad():
                r_dev = next(self.ref.parameters()).device
                r_logits = self.ref(**{k: v.to(r_dev) if torch.is_tensor(v) else v
                                       for k, v in batch.items()}).logits.to(dev)
            ref_pred, _ = shift_for_next_token(r_logits, cmask)
        total = pred.new_zeros((), dtype=torch.float32)
        kl_sum = pred.new_zeros((), dtype=torch.float32)
        for s in range(0, pred.shape[1], self.chunk):
            sl = slice(s, s + self.chunk)
            m = mask[:, sl]
            if not m.any():
                continue
            lp = pred[:, sl].float().log_softmax(-1).gather(-1, targets[:, sl, None]).squeeze(-1)
            ratio = torch.exp(lp - lp.detach())
            obj = torch.max(-ratio * adv, -ratio.clamp(1 - self.clip_eps, 1 + self.clip_eps) * adv)
            if ref_pred is not None:                 # k3: exp(ref - lp) - (ref - lp) - 1 >= 0
                rlp = ref_pred[:, sl].float().log_softmax(-1).gather(-1, targets[:, sl, None]).squeeze(-1)
                d = rlp - lp
                kl = torch.exp(d) - d - 1
                obj = obj + self.kl_beta * kl
                kl_sum = kl_sum + (kl.detach() * m).sum()
            total = total + (obj * m).sum()
        stats["loss/total"] = total.detach() / n_tok.clamp(min=1)
        if ref_pred is not None:
            stats["grpo/kl"] = kl_sum / n_tok.clamp(min=1)
        return total, stats

    @torch.no_grad()
    def update_ema_teacher(self):            # no teacher in GRPO
        return
