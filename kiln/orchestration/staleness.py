"""Adaptive queue-depth controller.

The right amount of asynchrony is a function of the measured ratio between
rollout time and training time per batch, not a constant:

* If rollouts are much slower than training (the common case with long
  reasoning traces), depth 1 already keeps the trainer fed and extra depth
  only adds staleness.
* If training is the bottleneck, deeper queues let more rollout workers run
  ahead, but every extra batch of depth costs one version of staleness.

The controller keeps an EMA of both durations and sets ``depth`` so that
roughly ``ceil(train_time / rollout_time)`` batches are in flight, clamped to
``[1, max_depth]`` and never exceeding ``max_staleness``. It only adjusts
every ``cooldown`` steps to avoid thrashing the buffer capacity.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class StalenessController:
    depth: int
    max_depth: int
    max_staleness: int
    ema: float = 0.3
    cooldown: int = 10
    _rollout_s: float | None = None
    _train_s: float | None = None
    _since_adjust: int = 0

    def observe(self, rollout_seconds: float | None = None, train_seconds: float | None = None) -> None:
        if rollout_seconds is not None and rollout_seconds > 0:
            self._rollout_s = (
                rollout_seconds
                if self._rollout_s is None
                else (1 - self.ema) * self._rollout_s + self.ema * rollout_seconds
            )
        if train_seconds is not None and train_seconds > 0:
            self._train_s = (
                train_seconds
                if self._train_s is None
                else (1 - self.ema) * self._train_s + self.ema * train_seconds
            )

    def step(self) -> int:
        """Call once per optimizer step; returns the (possibly updated) depth."""
        self._since_adjust += 1
        if self._since_adjust < self.cooldown or self._rollout_s is None or self._train_s is None:
            return self.depth
        self._since_adjust = 0
        ratio = self._train_s / max(self._rollout_s, 1e-9)
        target = int(math.ceil(ratio)) if ratio > 1.0 else 1
        target = max(1, min(target, self.max_depth, self.max_staleness))
        # move one step at a time toward target
        if target > self.depth:
            self.depth += 1
        elif target < self.depth:
            self.depth -= 1
        return self.depth

    def metrics(self) -> dict[str, float]:
        return {
            "async/queue_depth": float(self.depth),
            "async/rollout_s_ema": self._rollout_s or 0.0,
            "async/train_s_ema": self._train_s or 0.0,
        }
