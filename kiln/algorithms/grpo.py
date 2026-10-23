"""GRPO, Dr. GRPO, RLOO and REINFORCE++ — the critic-free family.

They differ only in the baseline/normalisation used to turn sequence rewards
into advantages; the surrogate loss, clipping and staleness correction are
shared. The policy-version-aware importance weights are computed from
``behavior_logp`` (engine) vs ``old_logp`` (trainer) exactly as described in
:mod:`kiln.algorithms.losses`.
"""

from __future__ import annotations

import torch

from kiln.algorithms.advantages import (
    grpo_advantages,
    reinforce_pp_advantages,
    rloo_advantages,
    whiten,
)
from kiln.algorithms.base import Algorithm, ForwardOutputs
from kiln.algorithms.losses import entropy_bonus, ppo_policy_loss, staleness_weights
from kiln.rollout.types import RolloutBatch


class GRPO(Algorithm):
    def compute_advantages(self, batch: RolloutBatch, values: torch.Tensor | None = None) -> torch.Tensor:
        mask = batch.token_mask()
        name = self.cfg.name
        if name == "grpo":
            adv = grpo_advantages(batch.rewards, batch.group_ids, mask, normalize_std=True)
        elif name == "dr_grpo":
            adv = grpo_advantages(batch.rewards, batch.group_ids, mask, normalize_std=False)
        elif name == "rloo":
            adv = rloo_advantages(batch.rewards, batch.group_ids, mask)
        elif name == "reinforce_pp":
            adv = reinforce_pp_advantages(batch.rewards, mask)
        else:
            raise ValueError(f"{name} is not a critic-free algorithm")
        if self.cfg.normalize_advantages and name in ("rloo",):
            adv = whiten(adv, mask)
        return adv

    def loss(self, batch: RolloutBatch, out: ForwardOutputs, advantages: torch.Tensor):
        mask = batch.token_mask()
        is_w = staleness_weights(
            out.old_logp,
            batch.behavior_logp,
            mask,
            mode=self.async_cfg.is_correction,
            clip=self.async_cfg.is_clip,
        )
        pl = ppo_policy_loss(
            out.logp,
            out.old_logp,
            advantages,
            mask,
            clip_eps=self.cfg.clip_eps,
            clip_eps_high=self.cfg.clip_eps_high,
            dual_clip=self.cfg.dual_clip,
            loss_agg=self.cfg.loss_agg,
            is_weights=is_w,
        )
        loss = pl.loss
        metrics = {
            "loss/policy": pl.loss.detach(),
            "policy/clipfrac": pl.clipfrac,
            "policy/approx_kl": pl.approx_kl,
            "policy/ratio": pl.ratio_mean,
            "async/is_weight": pl.is_weight_mean,
            "policy/entropy": (out.entropy * mask).sum().detach() / mask.sum().clamp_min(1),
            "reward/mean": batch.rewards.mean().detach(),
            "adv/abs_mean": (advantages.abs() * mask).sum().detach() / mask.sum().clamp_min(1),
        }
        kl_loss, kl_m = self.kl_term(out, mask)
        loss = loss + kl_loss
        metrics.update(kl_m)
        if self.cfg.entropy_coef > 0:
            ent = entropy_bonus(out.entropy, mask, self.cfg.loss_agg)
            loss = loss + self.cfg.entropy_coef * ent
            metrics["loss/entropy"] = ent.detach()
        metrics["loss/total"] = loss.detach()
        return loss, metrics
