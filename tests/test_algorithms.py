"""Unit tests for advantages, losses, KL and the algorithm classes (CPU, milliseconds)."""

from __future__ import annotations

import math

import pytest
import torch

from kiln.algorithms import ForwardOutputs, build_algorithm
from kiln.algorithms.advantages import (
    gae_advantages,
    grpo_advantages,
    reinforce_pp_advantages,
    rloo_advantages,
    whiten,
)
from kiln.algorithms.kl import AdaptiveKLController, kl_penalty, masked_mean
from kiln.algorithms.losses import aggregate, dpo_loss, ppo_policy_loss, staleness_weights, value_loss
from kiln.config.schema import AlgorithmConfig, AsyncConfig
from kiln.rollout.types import Sample, collate_samples


def _batch(rewards, group_ids, T=6, resp_start=2):
    samples = [
        Sample(
            prompt_ids=list(range(resp_start)),
            response_ids=list(range(T - resp_start)),
            behavior_logprobs=[-1.0] * (T - resp_start),
            group_id=g,
            policy_version=0,
            reward=r,
        )
        for r, g in zip(rewards, group_ids)
    ]
    return collate_samples(samples)


# --------------------------------------------------------------- advantages
def test_grpo_advantages_group_normalised():
    rewards = torch.tensor([1.0, 0.0, 1.0, 0.0, 5.0, 5.0])
    groups = torch.tensor([0, 0, 0, 0, 1, 1])
    mask = torch.ones(6, 4)
    adv = grpo_advantages(rewards, groups, mask)
    # group 0: mean .5, std (unbiased) = sqrt(1/3) -> +-0.866
    assert torch.allclose(adv[:4, 0], torch.tensor([1, -1, 1, -1.0]) * 0.5 / math.sqrt(1 / 3), atol=1e-4)
    # group 1: zero variance -> zero advantage (eps prevents blow-up)
    assert torch.allclose(adv[4:], torch.zeros(2, 4))


def test_dr_grpo_skips_std():
    rewards = torch.tensor([1.0, 0.0])
    adv = grpo_advantages(rewards, torch.tensor([0, 0]), torch.ones(2, 3), normalize_std=False)
    assert torch.allclose(adv[:, 0], torch.tensor([0.5, -0.5]))


def test_rloo_leave_one_out_baseline():
    rewards = torch.tensor([3.0, 1.0, 2.0])
    adv = rloo_advantages(rewards, torch.tensor([0, 0, 0]), torch.ones(3, 2))
    # r_i - mean of others: 3-1.5, 1-2.5, 2-2
    assert torch.allclose(adv[:, 0], torch.tensor([1.5, -1.5, 0.0]))


def test_singleton_groups_carry_no_signal():
    adv = grpo_advantages(torch.tensor([7.0]), torch.tensor([0]), torch.ones(1, 3))
    assert adv.abs().sum() == 0
    adv = rloo_advantages(torch.tensor([7.0]), torch.tensor([0]), torch.ones(1, 3))
    assert adv.abs().sum() == 0


def test_reinforce_pp_global_normalisation():
    adv = reinforce_pp_advantages(torch.tensor([1.0, 2.0, 3.0]), torch.ones(3, 2))
    assert abs(adv[:, 0].mean().item()) < 1e-6
    assert abs(adv[:, 0].std(unbiased=False).item() - 1) < 1e-4


def test_gae_matches_reference_loop():
    torch.manual_seed(1)
    B, T = 3, 7
    rewards = torch.randn(B, T)
    values = torch.randn(B, T)
    mask = torch.ones(B, T)
    mask[1, 5:] = 0  # shorter sequence
    adv, ret = gae_advantages(rewards, values, mask, gamma=0.99, lam=0.9)
    # reference: plain python recursion per row, over valid positions only
    for b in range(B):
        L = int(mask[b].sum())
        last, nxt = 0.0, 0.0
        ref = [0.0] * T
        for t in reversed(range(L)):
            delta = rewards[b, t] + 0.99 * nxt - values[b, t]
            last = delta + 0.99 * 0.9 * last
            ref[t] = float(last)
            nxt = float(values[b, t])
        assert torch.allclose(adv[b], torch.tensor(ref), atol=1e-5)
    assert torch.allclose(ret, adv + values * mask)


def test_whiten_masked():
    adv = torch.tensor([[1.0, 2.0, 100.0], [3.0, 4.0, 100.0]])
    mask = torch.tensor([[1, 1, 0], [1, 1, 0]])
    w = whiten(adv, mask)
    assert abs((w * mask).sum()) < 1e-5
    assert (w[:, 2] == 0).all()


# -------------------------------------------------------------------- losses
def test_ppo_loss_zero_at_old_policy_is_vanilla_pg():
    torch.manual_seed(0)
    logp = torch.randn(2, 5, requires_grad=True)
    old = logp.detach().clone()
    adv = torch.randn(2, 5)
    mask = torch.ones(2, 5)
    out = ppo_policy_loss(logp, old, adv, mask, loss_agg="token")
    out.loss.backward()
    # at ratio == 1 the clipped surrogate's gradient equals -adv/N
    assert torch.allclose(logp.grad, -adv / 10, atol=1e-6)
    assert out.clipfrac == 0 and out.ratio_mean.item() == pytest.approx(1.0)


def test_ppo_clipping_kills_gradient_outside_trust_region():
    logp = torch.full((1, 4), 0.0, requires_grad=True)
    old = torch.full((1, 4), -1.0)  # ratio = e > 1.2 ; positive advantage -> clipped
    adv = torch.ones(1, 4)
    out = ppo_policy_loss(logp, old, adv, torch.ones(1, 4))
    out.loss.backward()
    assert torch.allclose(logp.grad, torch.zeros_like(logp.grad))
    assert out.clipfrac.item() == 1.0


def test_dual_clip_bounds_negative_advantage_loss():
    logp = torch.full((1, 1), 3.0)
    old = torch.zeros(1, 1)  # ratio = e^3 ~ 20
    adv = -torch.ones(1, 1)
    without = ppo_policy_loss(logp, old, adv, torch.ones(1, 1), dual_clip=None).loss
    with_dc = ppo_policy_loss(logp, old, adv, torch.ones(1, 1), dual_clip=3.0).loss
    assert with_dc.item() == pytest.approx(3.0)
    assert without.item() > with_dc.item()


def test_clip_higher_is_asymmetric():
    logp = torch.full((1, 1), math.log(1.25))
    old = torch.zeros(1, 1)
    adv = torch.ones(1, 1)
    sym = ppo_policy_loss(logp, old, adv, torch.ones(1, 1), clip_eps=0.2)
    asym = ppo_policy_loss(logp, old, adv, torch.ones(1, 1), clip_eps=0.2, clip_eps_high=0.28)
    assert sym.loss.item() == pytest.approx(-1.2)
    assert asym.loss.item() == pytest.approx(-1.25)


@pytest.mark.parametrize("mode", ["token", "sequence", "seq_mean_token_sum"])
def test_aggregate_modes(mode):
    x = torch.tensor([[1.0, 1.0, 1.0, 0.0], [2.0, 0.0, 0.0, 0.0]])
    m = torch.tensor([[1, 1, 1, 0], [1, 0, 0, 0]])
    v = aggregate(x, m, mode).item()
    expected = {"token": 5 / 4, "sequence": (1 + 2) / 2, "seq_mean_token_sum": (3 + 2) / 2}[mode]
    assert v == pytest.approx(expected)


def test_staleness_weights_token_and_sequence():
    old = torch.tensor([[0.0, 0.0, 0.0]])
    beh = torch.tensor([[-1.0, 1.0, 0.0]])  # ratios e, 1/e, 1
    mask = torch.ones(1, 3)
    tok = staleness_weights(old, beh, mask, mode="token", clip=10.0)
    assert torch.allclose(tok, torch.tensor([[math.e, 1 / math.e, 1.0]]), atol=1e-5)
    seq = staleness_weights(old, beh, mask, mode="sequence", clip=10.0)
    assert torch.allclose(seq, torch.ones(1, 3), atol=1e-5)  # log-ratios cancel
    clipped = staleness_weights(old, beh, mask, mode="token", clip=2.0)
    assert clipped.max().item() == pytest.approx(2.0)
    assert torch.equal(staleness_weights(old, None, mask), torch.ones(1, 3))


def test_value_loss_clipped():
    v = torch.tensor([[2.0]])
    old = torch.tensor([[0.0]])
    ret = torch.tensor([[0.0]])
    out = value_loss(v, old, ret, torch.ones(1, 1), clip=0.5)
    # clipped prediction 0.5 -> (0.5)^2 = .25 ; unclipped 4 -> max = 4 ; *0.5
    assert out[0].item() == pytest.approx(0.5 * 4.0)


def test_dpo_loss_prefers_chosen():
    pc, pr = torch.tensor([2.0]), torch.tensor([0.0])
    rc, rr = torch.tensor([0.0]), torch.tensor([0.0])
    loss, m = dpo_loss(pc, pr, rc, rr, beta=0.1)
    assert m["dpo/accuracy"].item() == 1.0
    loss2, _ = dpo_loss(pr, pc, rc, rr, beta=0.1)
    assert loss2 > loss


# ------------------------------------------------------------------------ KL
def test_kl_estimators():
    lp, ref = torch.tensor([0.0]), torch.tensor([-1.0])
    assert kl_penalty(lp, ref, "k1").item() == pytest.approx(1.0)
    assert kl_penalty(lp, ref, "k2").item() == pytest.approx(0.5)
    assert kl_penalty(lp, ref, "k3").item() == pytest.approx(math.exp(-1) + 1 - 1)
    assert kl_penalty(lp, lp, "k3").item() == 0.0
    assert kl_penalty(lp, ref, "k3").item() >= 0


def test_adaptive_kl_controller_moves_toward_target():
    c = AdaptiveKLController(0.1, target=0.01, horizon=10)
    for _ in range(5):
        c.update(0.1)  # KL far above target -> coefficient grows
    assert c.coef > 0.1
    c2 = AdaptiveKLController(0.1, target=0.01, horizon=10)
    for _ in range(5):
        c2.update(0.0)
    assert c2.coef < 0.1
    assert AdaptiveKLController(0.3, None).update(5.0) == 0.3


def test_masked_mean():
    x = torch.tensor([[1.0, 100.0], [3.0, 100.0]])
    m = torch.tensor([[1, 0], [1, 0]])
    assert masked_mean(x, m).item() == 2.0
    assert torch.equal(masked_mean(x, m, dim=-1), torch.tensor([1.0, 3.0]))


# ------------------------------------------------------------ algorithms
@pytest.mark.parametrize("name", ["grpo", "dr_grpo", "rloo", "reinforce_pp"])
def test_critic_free_algorithms_run(name):
    cfg = AlgorithmConfig(name=name, kl={"coef": 0.01})
    algo = build_algorithm(cfg, AsyncConfig())
    b = _batch([1.0, 0.0, 1.0, 0.0], [0, 0, 1, 1])
    adv = algo.compute_advantages(b)
    assert adv.shape == (4, b.seq_len - 1)
    logp = torch.randn(4, b.seq_len - 1, requires_grad=True)
    out = ForwardOutputs(
        logp=logp,
        entropy=torch.rand(4, b.seq_len - 1),
        old_logp=logp.detach() + 0.01,
        ref_logp=logp.detach() - 0.05,
    )
    loss, m = algo.loss(b, out, adv)
    loss.backward()
    assert torch.isfinite(loss) and logp.grad is not None
    assert "kl/ref" in m and m["kl/ref"] >= 0


def test_ppo_algorithm_needs_values():
    cfg = AlgorithmConfig(name="ppo")
    algo = build_algorithm(cfg, AsyncConfig())
    b = _batch([1.0, 0.0], [0, 0])
    with pytest.raises(AssertionError):
        algo.compute_advantages(b, None)
    values = torch.zeros(2, b.seq_len - 1)
    adv = algo.compute_advantages(b, values)
    assert "returns" in b.extra
    logp = torch.randn(2, b.seq_len - 1, requires_grad=True)
    out = ForwardOutputs(
        logp,
        torch.zeros_like(logp),
        logp.detach(),
        values=values.clone().requires_grad_(True),
        old_values=values,
    )
    loss, m = algo.loss(b, out, adv)
    loss.backward()
    assert "loss/value" in m


def test_kl_in_reward_shapes_per_token_rewards():
    cfg = AlgorithmConfig(name="ppo", kl={"coef": 0.1, "in_reward": True})
    algo = build_algorithm(cfg, AsyncConfig())
    b = _batch([1.0, 0.0], [0, 0])
    old = torch.zeros(2, b.seq_len - 1)
    shaped = algo.shape_rewards(b, ForwardOutputs(old, old, old, ref_logp=old))
    mask = b.token_mask()
    # with identical policies KL=0 so the shaped reward is just R on the last response token
    assert shaped.sum().item() == pytest.approx(1.0)
    last = (mask.cumsum(-1) * mask).argmax(-1)
    assert shaped[0, last[0]].item() == pytest.approx(1.0)
