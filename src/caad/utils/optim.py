"""
caad/utils/optim.py
===================
BF16AdamW: AdamW for full fine-tuning with bf16 parameters and no fp32 master copy.

Plain AdamW on bf16 weights silently drops most updates at full-FT learning
rates (an update of ~1e-6 relative to a weight is far below bf16's ~4e-3
resolution), and bf16 .grad loses precision when 8-64 backward passes are
accumulated into it. This optimizer keeps everything that needs precision in
fp32 and pays only for that:

  * gradients are accumulated in fp32 buffers (a post-accumulate-grad hook moves
    each bf16 .grad into its buffer and frees it), so .grad stays None;
  * Adam moments are fp32;
  * the update is computed in fp32 and written back to the bf16 weight with
    stochastic rounding (unbiased: E[round(w)] = w), so small updates survive
    in expectation.

Memory per parameter: 2 (bf16 weight) + 4 (fp32 grad) + 8 (fp32 moments) bytes,
vs 16+ for an fp32 master-weight setup. The training loop must use
``scale_grads`` / ``clip_grad_norm`` (not .grad) for this optimizer.
"""

from __future__ import annotations

import math

import torch


def stochastic_round_(dst: torch.Tensor, src: torch.Tensor) -> None:
    """Write fp32 ``src`` into bf16 ``dst`` with stochastic rounding: add uniform
    noise to the 16 low mantissa bits that bf16 drops, then truncate."""
    bits = src.view(torch.int32)
    noise = torch.randint_like(bits, 0, 1 << 16)
    dst.copy_(((bits + noise) & -65536).view(torch.float32))   # low bits are 0: bf16 cast is exact


class BF16AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay))
        for group in self.param_groups:
            for p in group["params"]:
                st = self.state[p]
                st["step"] = 0
                st["grad"] = torch.zeros_like(p, dtype=torch.float32)
                st["exp_avg"] = torch.zeros_like(p, dtype=torch.float32)
                st["exp_avg_sq"] = torch.zeros_like(p, dtype=torch.float32)
                p.register_post_accumulate_grad_hook(self._accumulate)

    def _accumulate(self, p):
        self.state[p]["grad"].add_(p.grad)
        p.grad = None

    def _bufs(self):
        return [self.state[p]["grad"] for g in self.param_groups for p in g["params"]]

    @torch.no_grad()
    def scale_grads(self, factor: float) -> None:
        torch._foreach_mul_(self._bufs(), factor)

    @torch.no_grad()
    def clip_grad_norm(self, max_norm: float) -> torch.Tensor:
        bufs = self._bufs()
        total = torch.linalg.vector_norm(torch.stack(torch._foreach_norm(bufs)))
        coef = (max_norm / (total + 1e-6)).clamp(max=1.0)
        torch._foreach_mul_(bufs, coef)
        return total

    @torch.no_grad()
    def zero_grad(self, set_to_none: bool = True) -> None:
        torch._foreach_zero_(self._bufs())
        super().zero_grad(set_to_none=set_to_none)

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            lr, (b1, b2), eps, wd = group["lr"], group["betas"], group["eps"], group["weight_decay"]
            for p in group["params"]:
                st = self.state[p]
                st["step"] += 1
                g, m, v = st["grad"], st["exp_avg"], st["exp_avg_sq"]
                m.mul_(b1).add_(g, alpha=1 - b1)
                v.mul_(b2).addcmul_(g, g, value=1 - b2)
                bc1, bc2 = 1 - b1 ** st["step"], 1 - b2 ** st["step"]
                w = p.float()
                if wd:
                    w.mul_(1 - lr * wd)
                w.addcdiv_(m, v.sqrt().div_(math.sqrt(bc2)).add_(eps), value=-lr / bc1)
                stochastic_round_(p, w)
                g.zero_()
        return None
