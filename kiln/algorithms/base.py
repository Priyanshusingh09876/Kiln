"""Algorithm interface: advantages + loss, nothing else.

An :class:`Algorithm` never touches a model, an optimizer, or a device. It is
handed tensors by the trainer and returns a scalar loss plus metrics. That
keeps the GRPO/PPO/DPO math unit-testable on CPU in milliseconds and makes the
trainer agnostic to which algorithm is running.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch

from kiln.algorithms.kl import AdaptiveKLController, kl_penalty, masked_mean
from kiln.config.schema import AlgorithmConfig, AsyncConfig
from kiln.rollout.types import RolloutBatch


@dataclass
class ForwardOutputs:
    logp: torch.Tensor  # [B, T-1] current policy
    entropy: torch.Tensor  # [B, T-1]
    old_logp: torch.Tensor  # [B, T-1] detached, trainer weights at window start
    ref_logp: torch.Tensor | None = None  # [B, T-1]
    values: torch.Tensor | None = None  # [B, T-1]
    old_values: torch.Tensor | None = None


class Algorithm(ABC):
    def __init__(self, cfg: AlgorithmConfig, async_cfg: AsyncConfig):
        self.cfg = cfg
        self.async_cfg = async_cfg
        self.kl_ctl = AdaptiveKLController(cfg.kl.coef, cfg.kl.target, cfg.kl.horizon)

    # ------------------------------------------------------------- hooks
    @abstractmethod
    def compute_advantages(self, batch: RolloutBatch, values: torch.Tensor | None = None) -> torch.Tensor:
        """Return per-token advantages ``[B, T-1]``; may also stash returns on ``batch.extra``."""

    @abstractmethod
    def loss(
        self, batch: RolloutBatch, out: ForwardOutputs, advantages: torch.Tensor
    ) -> tuple[torch.Tensor, dict]: ...

    # ----------------------------------------------------------- shared
    def kl_term(self, out: ForwardOutputs, mask: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """KL-to-reference loss term (zero tensor when disabled)."""
        if out.ref_logp is None or self.cfg.kl.coef <= 0 or self.cfg.kl.in_reward:
            return torch.zeros((), device=out.logp.device), {}
        kl = kl_penalty(out.logp, out.ref_logp, self.cfg.kl.estimator)
        kl_mean = masked_mean(kl, mask)
        coef = self.kl_ctl.update(float(kl_mean.detach()))
        return coef * kl_mean, {
            "kl/ref": kl_mean.detach(),
            "kl/coef": torch.tensor(coef, device=kl_mean.device),
        }

    def shape_rewards(self, batch: RolloutBatch, out: ForwardOutputs) -> torch.Tensor | None:
        """PPO-style per-token reward = -coef*KL on every token, +R on the last response token."""
        if not self.cfg.kl.in_reward or out.ref_logp is None:
            return None
        mask = batch.token_mask().float()
        kl = kl_penalty(out.old_logp, out.ref_logp, self.cfg.kl.estimator) * mask
        per_tok = -self.cfg.kl.coef * kl
        last_idx = (mask.cumsum(-1) * mask).argmax(-1)
        per_tok[torch.arange(mask.shape[0]), last_idx] += batch.rewards
        return per_tok
