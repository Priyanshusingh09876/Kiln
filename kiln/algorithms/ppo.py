"""PPO with a learned critic and GAE."""

from __future__ import annotations

import torch

from kiln.algorithms.advantages import gae_advantages, whiten
from kiln.algorithms.base import Algorithm, ForwardOutputs
from kiln.algorithms.losses import entropy_bonus, ppo_policy_loss, staleness_weights, value_loss
from kiln.rollout.types import RolloutBatch


class PPO(Algorithm):
    def compute_advantages(self, batch: RolloutBatch, values: torch.Tensor | None = None) -> torch.Tensor:
        assert values is not None, "PPO needs critic values"
        mask = batch.token_mask()
        per_tok_rewards = batch.extra.get("shaped_rewards")
        if per_tok_rewards is None:
            per_tok_rewards = torch.zeros_like(values)
            last_idx = (mask.float().cumsum(-1) * mask).argmax(-1)
            per_tok_rewards[torch.arange(mask.shape[0]), last_idx] = batch.rewards
        adv, returns = gae_advantages(
            per_tok_rewards, values.detach(), mask, self.cfg.gae_gamma, self.cfg.gae_lambda
        )
        batch.extra["returns"] = returns
        if self.cfg.normalize_advantages:
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
        assert out.values is not None and out.old_values is not None
        vl, vclip = value_loss(out.values, out.old_values, batch.extra["returns"], mask, self.cfg.value_clip)
        loss = pl.loss + self.cfg.value_coef * vl
        metrics = {
            "loss/policy": pl.loss.detach(),
            "loss/value": vl.detach(),
            "policy/clipfrac": pl.clipfrac,
            "value/clipfrac": vclip,
            "policy/approx_kl": pl.approx_kl,
            "async/is_weight": pl.is_weight_mean,
            "reward/mean": batch.rewards.mean().detach(),
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
