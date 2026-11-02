"""EOPD-style loss: clipped reverse KL on every token + top-k forward KL added
on gated tokens, returned as a token sum (Jin et al., 2026, Eq. 6-10)."""
import pytest

torch = pytest.importorskip("torch")

from caad.losses.divergence import eopd_loss, fkl_gate_mask, teacher_entropy  # noqa: E402


def _setup(B=1, L=12, V=40, seed=0):
    g = torch.Generator().manual_seed(seed)
    s = torch.randn(B, L, V, generator=g, requires_grad=True)
    t = torch.randn(B, L, V, generator=g) * 2
    tgt = torch.randint(0, V, (B, L), generator=g)
    mask = torch.zeros(B, L); mask[:, 3:] = 1
    return s, t, tgt, mask


def test_rkl_base_is_policy_gradient_with_teacher_advantage():
    s, t, tgt, mask = _setup()
    gate = torch.zeros_like(mask, dtype=torch.bool)              # no forward KL
    loss, n, _ = eopd_loss(s, t, tgt, mask, gate, topk=8, base="rkl")
    loss.backward()
    # manual: grad of sum_t -A_t log pi(x_t), A_t = log pi_T(x_t) - log pi(x_t) (detached)
    s2 = s.detach().clone().requires_grad_(True)
    ls, lt = s2.log_softmax(-1), t.log_softmax(-1)
    lp_s = ls.gather(-1, tgt[..., None]).squeeze(-1)
    lp_t = lt.gather(-1, tgt[..., None]).squeeze(-1)
    (-(lp_t - lp_s).detach() * lp_s * mask).sum().backward()
    assert torch.allclose(s.grad, s2.grad, atol=1e-6)
    assert n == mask.sum()
    # value: r == 1, so loss == -sum A_t
    assert torch.allclose(loss, (-(lp_t - lp_s).detach() * mask).sum(), atol=1e-5)


def test_forward_kl_added_only_on_gated_tokens_over_teacher_topk():
    s, t, tgt, mask = _setup()
    gate = torch.zeros_like(mask, dtype=torch.bool); gate[0, 5] = gate[0, 9] = True
    base, _, _ = eopd_loss(s, t, tgt, mask, torch.zeros_like(gate), topk=8, base="rkl")
    full, _, st = eopd_loss(s, t, tgt, mask, gate, topk=8, base="rkl")
    ls, lt = s.detach().log_softmax(-1), t.log_softmax(-1)
    exp = 0.0
    for pos in (5, 9):
        top, idx = lt[0, pos].topk(8)
        pt = top.softmax(-1)                                     # renormalized teacher top-k
        exp += (pt * (pt.log() - ls[0, pos, idx])).sum()        # student NOT renormalized
    assert torch.allclose(full - base, exp, atol=1e-5)          # the two terms are added
    assert abs(float(st["distill/fkl_frac"]) - 2 / 9) < 1e-6


def test_identical_student_teacher():
    s, _, tgt, mask = _setup()
    t = s.detach().clone()
    gate = torch.ones_like(mask, dtype=torch.bool)
    loss, n, st = eopd_loss(s, t, tgt, mask, gate, topk=8, base="rkl")
    # advantage is 0; forward KL reduces to -log(teacher top-k mass) >= 0
    lt = t.log_softmax(-1)
    mass = lt.topk(8, -1).values.exp().sum(-1)
    assert torch.allclose(loss, (-mass.log() * mask).sum(), atol=1e-5)
    assert float(st["eopd/rkl_exact"]) < 1e-6 and float(st["eopd/advantage_mean"]) == 0


def test_gate_selects_lowest_entropy_decile():
    V, L = 50, 60
    logits = torch.randn(1, L, V) * torch.linspace(0.1, 6, L)[None, :, None]   # rising sharpness
    H = teacher_entropy(logits)
    mask = torch.ones(1, L)
    gate = fkl_gate_mask(H, mask, q=0.10, min_tokens=24)
    k = int(gate.sum())
    assert 5 <= k <= 7                                          # ~10% of 60 tokens
    assert H[gate].max() <= H[~gate].min()                      # exactly the lowest-entropy ones


def test_jsd_base_on_every_token_plus_gated_forward_kl():
    s, t, tgt, mask = _setup()
    gate = torch.zeros_like(mask, dtype=torch.bool); gate[0, 4] = True
    loss, n, st = eopd_loss(s, t, tgt, mask, gate, topk=8, base="jsd", jsd_beta=0.5)
    ps, pt = s.detach().softmax(-1), t.softmax(-1)
    m = 0.5 * ps + 0.5 * pt
    jsd = 0.5 * (ps * (ps / m).log()).sum(-1) + 0.5 * (pt * (pt / m).log()).sum(-1)
    lt, ls = t.log_softmax(-1), s.detach().log_softmax(-1)
    top, idx = lt[0, 4].topk(8); ptk = top.softmax(-1)
    fkl = (ptk * (ptk.log() - ls[0, 4, idx])).sum()
    assert torch.allclose(loss, (jsd * mask).sum() + fkl, atol=1e-5)
    # identical student / teacher -> JSD part is 0
    l0, _, _ = eopd_loss(s, s.detach().clone(), tgt, mask, torch.zeros_like(gate), base="jsd")
    assert abs(float(l0)) < 1e-5


def test_gate_modes():
    from caad.losses.divergence import entropy_gate
    H = torch.linspace(0.0, 2.0, 40)[None]
    m = torch.ones(1, 40)
    lo, hi = entropy_gate(H, m, "low_quantile", q=0.1), entropy_gate(H, m, "high_quantile", q=0.1)
    th = entropy_gate(H, m, "high_threshold", tau=0.8)
    assert H[lo].max() < H[~lo].min() and H[hi].min() > H[~hi].max()
    assert torch.equal(th, H > 0.8)


def test_threshold_and_gap_gates():
    from caad.losses.divergence import entropy_gate
    Ht = torch.tensor([[0.01, 0.15, 0.5, 0.9, 1.6, 0.05]])
    Hs = torch.tensor([[0.01, 0.60, 0.5, 0.2, 1.5, 0.00]])
    m = torch.ones(1, 6); m[0, 5] = 0                       # last token is padding
    lo = entropy_gate(Ht, m, "low_threshold", tau_low=0.2)
    assert lo.tolist() == [[True, True, False, False, False, False]]
    both = entropy_gate(Ht, m, "low_or_high_threshold", tau_low=0.2, tau=0.8)
    assert both.tolist() == [[True, True, False, True, True, False]]
    gap = entropy_gate(Ht, m, "abs_gap_threshold", student_entropy=Hs, tau_gap=0.2)
    # |dH| = 0, .45, 0, .7, .1, (pad)  -> student confused (.45) and student overconfident (.7)
    assert gap.tolist() == [[False, True, False, True, False, False]]
    with pytest.raises(ValueError):
        entropy_gate(Ht, m, "abs_gap_threshold")


def test_grpo_group_advantages():
    from caad.trainers.grpo import group_advantages
    a = group_advantages([1.1, 0.1, 0.1, 1.1])
    assert abs(sum(a)) < 1e-6 and a[0] > 0 > a[1] and abs(a[0] - 1.0) < 1e-4
    assert group_advantages([1.0, 1.0, 1.0]) == [0.0, 0.0, 0.0]      # no signal when all equal
    b = group_advantages([1.0, 0.0], std_normalize=False)
    assert b == [0.5, -0.5]
