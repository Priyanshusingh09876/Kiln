"""Offline DPO / IPO / hinge on preference pairs.

DPO does not use rollouts, so it does not go through :class:`RolloutBatch`.
The trainer passes chosen/rejected token log-probs directly.
"""

from __future__ import annotations

import torch

from kiln.algorithms.base import Algorithm, ForwardOutputs
from kiln.algorithms.losses import dpo_loss
from kiln.rollout.types import RolloutBatch


class DPO(Algorithm):
    def compute_advantages(self, batch: RolloutBatch, values: torch.Tensor | None = None) -> torch.Tensor:
        raise NotImplementedError("DPO is offline and has no advantages")

    def loss(self, batch: RolloutBatch, out: ForwardOutputs, advantages: torch.Tensor):
        raise NotImplementedError("use DPO.pair_loss")

    def pair_loss(
        self,
        chosen_logp: torch.Tensor,
        chosen_mask: torch.Tensor,
        rejected_logp: torch.Tensor,
        rejected_mask: torch.Tensor,
        ref_chosen_logp: torch.Tensor,
        ref_rejected_logp: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        """All inputs ``[B, T-1]`` per-token; sums over the response mask."""
        pc = (chosen_logp * chosen_mask).sum(-1)
        pr = (rejected_logp * rejected_mask).sum(-1)
        rc = (ref_chosen_logp * chosen_mask).sum(-1)
        rr = (ref_rejected_logp * rejected_mask).sum(-1)
        loss, m = dpo_loss(pc, pr, rc, rr, beta=self.cfg.dpo_beta, variant=self.cfg.dpo_variant)
        m["loss/total"] = loss.detach()
        return loss, m
