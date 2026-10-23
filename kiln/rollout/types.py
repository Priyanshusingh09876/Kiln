"""Data containers that flow between rollout engines, the experience buffer and the trainer.

Alignment convention used everywhere in Kiln
--------------------------------------------
``input_ids`` is ``[B, T]`` = prompt tokens followed by response tokens
(right-padded). ``response_mask`` is ``[B, T]`` with 1 on response tokens.
Any per-token quantity produced by a causal LM (log-prob, entropy, value,
advantage) is ``[B, T-1]`` and aligned to the *predicted* token
``input_ids[:, 1:]``; its mask is therefore ``response_mask[:, 1:]``. Helpers
below (:meth:`RolloutBatch.token_mask`) make this a one-liner so the
off-by-one lives in exactly one place.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class Sample:
    """One prompt/response pair as produced by a rollout engine."""

    prompt_ids: list[int]
    response_ids: list[int]
    behavior_logprobs: list[float]
    """Per-response-token log-prob reported by the engine (len == len(response_ids))."""
    group_id: int
    policy_version: int
    prompt_text: str = ""
    response_text: str = ""
    finish_reason: str = "stop"
    reward: float | None = None
    reward_breakdown: dict[str, float] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def total_len(self) -> int:
        return len(self.prompt_ids) + len(self.response_ids)


@dataclass
class RolloutBatch:
    input_ids: torch.Tensor  # [B, T] long
    attention_mask: torch.Tensor  # [B, T] long (1 = real token)
    response_mask: torch.Tensor  # [B, T] long (1 = response token)
    position_ids: torch.Tensor  # [B, T] long
    behavior_logp: torch.Tensor  # [B, T-1] float, aligned to input_ids[:, 1:]
    rewards: torch.Tensor  # [B] float (sequence-level)
    group_ids: torch.Tensor  # [B] long
    policy_version: torch.Tensor  # [B] long
    samples: list[Sample] = field(default_factory=list)
    extra: dict[str, torch.Tensor] = field(default_factory=dict)

    # ------------------------------------------------------------ helpers
    @property
    def batch_size(self) -> int:
        return int(self.input_ids.shape[0])

    @property
    def seq_len(self) -> int:
        return int(self.input_ids.shape[1])

    def token_mask(self) -> torch.Tensor:
        """Mask for ``[B, T-1]`` per-token tensors (response positions only)."""
        return self.response_mask[:, 1:]

    def num_response_tokens(self) -> int:
        return int(self.response_mask.sum().item())

    def to(self, device: torch.device | str) -> RolloutBatch:
        kw = {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in self.__dict__.items()
            if k != "extra"
        }
        kw["extra"] = {k: v.to(device) for k, v in self.extra.items()}
        return RolloutBatch(**kw)

    def slice(self, start: int, end: int) -> RolloutBatch:
        return RolloutBatch(
            input_ids=self.input_ids[start:end],
            attention_mask=self.attention_mask[start:end],
            response_mask=self.response_mask[start:end],
            position_ids=self.position_ids[start:end],
            behavior_logp=self.behavior_logp[start:end],
            rewards=self.rewards[start:end],
            group_ids=self.group_ids[start:end],
            policy_version=self.policy_version[start:end],
            samples=self.samples[start:end],
            extra={k: v[start:end] for k, v in self.extra.items()},
        )

    def micro_batches(self, size: int):
        for s in range(0, self.batch_size, size):
            yield self.slice(s, min(s + size, self.batch_size))

    def staleness(self, current_version: int) -> torch.Tensor:
        return current_version - self.policy_version


def collate_samples(samples: list[Sample], pad_id: int = 0, max_len: int | None = None) -> RolloutBatch:
    """Right-pad a list of :class:`Sample` into a :class:`RolloutBatch`."""
    assert samples, "cannot collate an empty list"
    lens = [s.total_len for s in samples]
    T = max(lens)
    if max_len is not None:
        T = min(T, max_len)
    B = len(samples)
    input_ids = torch.full((B, T), pad_id, dtype=torch.long)
    attn = torch.zeros((B, T), dtype=torch.long)
    resp = torch.zeros((B, T), dtype=torch.long)
    pos = torch.zeros((B, T), dtype=torch.long)
    blogp = torch.zeros((B, T - 1), dtype=torch.float32)
    for i, s in enumerate(samples):
        ids = (s.prompt_ids + s.response_ids)[:T]
        n = len(ids)
        p = min(len(s.prompt_ids), n)
        input_ids[i, :n] = torch.tensor(ids, dtype=torch.long)
        attn[i, :n] = 1
        resp[i, p:n] = 1
        pos[i, :n] = torch.arange(n)
        # response token k (0-based) sits at absolute index p+k and is predicted at logit index p+k-1
        r = n - p
        if r > 0:
            blogp[i, p - 1 : p - 1 + r] = torch.tensor(s.behavior_logprobs[:r], dtype=torch.float32)
    rewards = torch.tensor([s.reward if s.reward is not None else 0.0 for s in samples], dtype=torch.float32)
    group_ids = torch.tensor([s.group_id for s in samples], dtype=torch.long)
    versions = torch.tensor([s.policy_version for s in samples], dtype=torch.long)
    return RolloutBatch(
        input_ids, attn, resp, pos, blogp, rewards, group_ids, versions, samples=list(samples)
    )


def concat_batches(batches: list[RolloutBatch], pad_id: int = 0) -> RolloutBatch:
    """Concatenate batches of differing ``T`` by re-collating their samples."""
    samples = [s for b in batches for s in b.samples]
    return collate_samples(samples, pad_id=pad_id)
