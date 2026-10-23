"""Advantage estimators.

All functions take sequence-level rewards ``[B]`` (or per-token ``[B, T]`` for
GAE) plus a ``group_ids [B]`` tensor mapping each response to the prompt it was
sampled from, and return per-token advantages ``[B, T]`` broadcast over the
``response_mask``. Keeping every estimator behind the same signature is what
lets ``AlgorithmConfig.name`` be a one-word switch.
"""

from __future__ import annotations

import torch


def _group_stats(values: torch.Tensor, group_ids: torch.Tensor, unbiased: bool = True):
    """Per-group mean/std scattered back to each element. Pure tensor ops, no Python loops."""
    n_groups = int(group_ids.max().item()) + 1 if group_ids.numel() else 0
    counts = torch.zeros(n_groups, device=values.device, dtype=values.dtype).scatter_add_(
        0, group_ids, torch.ones_like(values)
    )
    sums = torch.zeros_like(counts).scatter_add_(0, group_ids, values)
    mean = sums / counts.clamp_min(1)
    centered = values - mean[group_ids]
    sq = torch.zeros_like(counts).scatter_add_(0, group_ids, centered * centered)
    denom = (counts - 1).clamp_min(1) if unbiased else counts.clamp_min(1)
    std = torch.sqrt(sq / denom)
    return mean[group_ids], std[group_ids], counts[group_ids]


def grpo_advantages(
    rewards: torch.Tensor,
    group_ids: torch.Tensor,
    response_mask: torch.Tensor,
    eps: float = 1e-6,
    normalize_std: bool = True,
) -> torch.Tensor:
    """GRPO: ``(r - mean_group) / std_group``. ``normalize_std=False`` gives Dr. GRPO."""
    mean, std, counts = _group_stats(rewards.float(), group_ids)
    adv = rewards.float() - mean
    if normalize_std:
        adv = adv / (std + eps)
    # groups of size 1 carry no signal
    adv = torch.where(counts > 1, adv, torch.zeros_like(adv))
    return adv.unsqueeze(-1) * response_mask.float()


def rloo_advantages(
    rewards: torch.Tensor, group_ids: torch.Tensor, response_mask: torch.Tensor
) -> torch.Tensor:
    """REINFORCE leave-one-out baseline: ``r_i - mean_{j != i} r_j``."""
    r = rewards.float()
    mean, _, counts = _group_stats(r, group_ids)
    loo = (mean * counts - r) / (counts - 1).clamp_min(1)
    adv = torch.where(counts > 1, r - loo, torch.zeros_like(r))
    return adv.unsqueeze(-1) * response_mask.float()


def reinforce_pp_advantages(
    rewards: torch.Tensor, response_mask: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    """REINFORCE++: global (whole-batch) mean/std normalisation, no grouping."""
    r = rewards.float()
    adv = (r - r.mean()) / (r.std(unbiased=False) + eps) if r.numel() > 1 else torch.zeros_like(r)
    return adv.unsqueeze(-1) * response_mask.float()


def gae_advantages(
    rewards: torch.Tensor,
    values: torch.Tensor,
    response_mask: torch.Tensor,
    gamma: float = 1.0,
    lam: float = 0.95,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generalised advantage estimation over per-token rewards.

    ``rewards``/``values``/``response_mask`` are ``[B, T]``. Masked positions
    are skipped so padding never leaks into the recursion. Returns
    ``(advantages, returns)`` where ``returns = advantages + values``.
    """
    B, T = rewards.shape
    mask = response_mask.float()
    adv = torch.zeros_like(rewards, dtype=torch.float32)
    last = torch.zeros(B, device=rewards.device, dtype=torch.float32)
    next_value = torch.zeros(B, device=rewards.device, dtype=torch.float32)
    for t in reversed(range(T)):
        m = mask[:, t]
        delta = rewards[:, t].float() + gamma * next_value - values[:, t].float()
        last = delta + gamma * lam * last
        last = last * m  # reset at padding
        adv[:, t] = last
        next_value = torch.where(m > 0, values[:, t].float(), next_value)
    returns = adv + values.float() * mask
    return adv * mask, returns


def whiten(adv: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8, shift_mean: bool = True) -> torch.Tensor:
    """Masked batch whitening of per-token advantages."""
    m = mask.float()
    n = m.sum().clamp_min(1)
    mean = (adv * m).sum() / n
    var = (((adv - mean) ** 2) * m).sum() / n
    out = (adv - mean) if shift_mean else adv
    return out * torch.rsqrt(var + eps) * m
