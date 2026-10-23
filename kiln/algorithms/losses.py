"""Loss functions shared by the GRPO / PPO / DPO algorithm classes.

Terminology for the on-policy ratio in an *asynchronous* trainer, where
three distinct policies touch every sample:

* ``behavior_logp`` — log-prob the **rollout engine** (vLLM/SGLang) reported at
  generation time, under policy version ``v_gen``.
* ``old_logp``      — log-prob recomputed by the **trainer** with the weights
  it had when the sample entered the current optimisation window (version
  ``v_old``). For strictly on-policy GRPO with ``ppo_epochs=1`` this equals the
  current log-prob on the first (only) pass.
* ``logp``          — log-prob under the weights being optimised right now.

PPO clips ``exp(logp - old_logp)``. Staleness between ``v_gen`` and ``v_old``
(because rollouts were produced a few versions ago, or because vLLM's fused
kernels disagree numerically with the FSDP forward) is corrected with a
*separate* truncated importance weight ``exp(old_logp - behavior_logp)`` that
multiplies the loss but is never differentiated. Keeping the two ratios apart
is the whole point: conflating them lets the clip range silently absorb
staleness and destabilises training once the queue gets deep.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from kiln.algorithms.kl import masked_mean


def aggregate(per_token: torch.Tensor, mask: torch.Tensor, mode: str) -> torch.Tensor:
    """Reduce a ``[B, T]`` per-token loss to a scalar.

    * ``token``              — mean over all response tokens in the batch (DAPO "token-level").
    * ``sequence``           — mean over sequences of per-sequence token-means (GRPO paper).
    * ``seq_mean_token_sum`` — mean over sequences of per-sequence token-sums (Dr. GRPO, length-unbiased).
    """
    m = mask.float()
    if mode == "token":
        return (per_token * m).sum() / m.sum().clamp_min(1.0)
    if mode == "sequence":
        return masked_mean(per_token, m, dim=-1).mean()
    if mode == "seq_mean_token_sum":
        return (per_token * m).sum(-1).mean()
    raise ValueError(f"unknown loss aggregation {mode!r}")


def staleness_weights(
    old_logp: torch.Tensor,
    behavior_logp: torch.Tensor | None,
    mask: torch.Tensor,
    mode: str = "token",
    clip: float = 2.0,
) -> torch.Tensor:
    """Truncated importance weights correcting for rollout/trainer policy mismatch.

    Returned tensor is ``[B, T]``, detached, in ``[0, clip]``. ``mode="sequence"``
    uses the length-normalised sequence ratio broadcast to every token, which
    is what MoE training wants (GSPO) because per-token ratios are dominated by
    router noise.
    """
    if behavior_logp is None or mode == "none":
        return torch.ones_like(old_logp)
    m = mask.float()
    log_ratio = (old_logp - behavior_logp).detach() * m
    if mode == "token":
        w = torch.exp(log_ratio.clamp(-20, 20))
    elif mode == "sequence":
        seq_log_ratio = log_ratio.sum(-1) / m.sum(-1).clamp_min(1.0)
        w = torch.exp(seq_log_ratio.clamp(-20, 20)).unsqueeze(-1).expand_as(old_logp)
    else:
        raise ValueError(f"unknown IS mode {mode!r}")
    return (w.clamp(max=clip) * m).detach()


@dataclass
class PolicyLossOut:
    loss: torch.Tensor
    clipfrac: torch.Tensor
    approx_kl: torch.Tensor
    ratio_mean: torch.Tensor
    is_weight_mean: torch.Tensor


def ppo_policy_loss(
    logp: torch.Tensor,
    old_logp: torch.Tensor,
    advantages: torch.Tensor,
    mask: torch.Tensor,
    clip_eps: float = 0.2,
    clip_eps_high: float | None = None,
    dual_clip: float | None = 3.0,
    loss_agg: str = "token",
    is_weights: torch.Tensor | None = None,
) -> PolicyLossOut:
    """Clipped surrogate with optional asymmetric clipping and dual-clip.

    ``clip_eps_high`` (DAPO "clip-higher") widens the upper bound so low-prob
    tokens can still grow. ``dual_clip`` (Ye et al. 2020) bounds the loss for
    *negative* advantages when the ratio explodes, which otherwise produces an
    unbounded gradient the standard clip does not protect against.
    """
    hi = clip_eps_high if clip_eps_high is not None else clip_eps
    m = mask.float()
    log_ratio = (logp - old_logp) * m
    ratio = torch.exp(log_ratio.clamp(-20, 20))
    surr1 = -advantages * ratio
    surr2 = -advantages * ratio.clamp(1.0 - clip_eps, 1.0 + hi)
    per_token = torch.maximum(surr1, surr2)
    if dual_clip is not None:
        dual = -advantages * dual_clip
        per_token = torch.where(advantages < 0, torch.minimum(per_token, dual), per_token)
    if is_weights is not None:
        per_token = per_token * is_weights
    loss = aggregate(per_token, m, loss_agg)
    with torch.no_grad():
        clipfrac = masked_mean(((ratio - 1.0).abs() > clip_eps).float(), m)
        approx_kl = masked_mean(0.5 * log_ratio.pow(2), m)
        ratio_mean = masked_mean(ratio, m)
        is_mean = masked_mean(is_weights, m) if is_weights is not None else torch.ones((), device=logp.device)
    return PolicyLossOut(loss, clipfrac, approx_kl, ratio_mean, is_mean)


def value_loss(
    values: torch.Tensor,
    old_values: torch.Tensor,
    returns: torch.Tensor,
    mask: torch.Tensor,
    clip: float = 0.2,
) -> tuple[torch.Tensor, torch.Tensor]:
    """PPO clipped value loss. Returns ``(loss, clipfrac)``."""
    m = mask.float()
    v_clipped = old_values + (values - old_values).clamp(-clip, clip)
    l1 = (values - returns) ** 2
    l2 = (v_clipped - returns) ** 2
    per_token = torch.maximum(l1, l2)
    loss = 0.5 * masked_mean(per_token, m)
    with torch.no_grad():
        clipfrac = masked_mean((l2 > l1).float(), m)
    return loss, clipfrac


def entropy_bonus(entropy: torch.Tensor, mask: torch.Tensor, loss_agg: str = "token") -> torch.Tensor:
    """Negative mean entropy (so *adding* ``coef * bonus`` to the loss maximises entropy)."""
    return -aggregate(entropy, mask, loss_agg)


def dpo_loss(
    policy_chosen_logp: torch.Tensor,
    policy_rejected_logp: torch.Tensor,
    ref_chosen_logp: torch.Tensor,
    ref_rejected_logp: torch.Tensor,
    beta: float = 0.1,
    variant: str = "sigmoid",
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """DPO family on *sequence-summed* log-probs ``[B]``.

    * ``sigmoid`` — Rafailov et al. 2023
    * ``ipo``     — Azar et al. 2023 (squared hinge on the margin)
    * ``hinge``   — SLiC-HF style
    """
    chosen_rewards = beta * (policy_chosen_logp - ref_chosen_logp)
    rejected_rewards = beta * (policy_rejected_logp - ref_rejected_logp)
    margin = chosen_rewards - rejected_rewards
    if variant == "sigmoid":
        loss = -F.logsigmoid(margin)
    elif variant == "ipo":
        loss = (margin / beta - 1.0 / (2.0 * beta)) ** 2
    elif variant == "hinge":
        loss = torch.relu(1.0 - margin)
    else:
        raise ValueError(f"unknown DPO variant {variant!r}")
    metrics = {
        "dpo/chosen_reward": chosen_rewards.mean().detach(),
        "dpo/rejected_reward": rejected_rewards.mean().detach(),
        "dpo/accuracy": (margin > 0).float().mean().detach(),
        "dpo/margin": margin.mean().detach(),
    }
    return loss.mean(), metrics
