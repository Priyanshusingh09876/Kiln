"""Tiny CUDA-aware timers used for step breakdowns and the benchmark CLI."""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field

import torch


@dataclass
class Timer:
    name: str
    total: float = 0.0
    count: int = 0
    last: float = 0.0

    @property
    def mean(self) -> float:
        return self.total / max(self.count, 1)

    def add(self, seconds: float) -> None:
        self.total += seconds
        self.count += 1
        self.last = seconds


@dataclass
class Timers:
    sync_cuda: bool = True
    timers: dict[str, Timer] = field(default_factory=dict)

    def get(self, name: str) -> Timer:
        if name not in self.timers:
            self.timers[name] = Timer(name)
        return self.timers[name]

    @contextmanager
    def __call__(self, name: str):
        if self.sync_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        try:
            yield
        finally:
            if self.sync_cuda and torch.cuda.is_available():
                torch.cuda.synchronize()
            self.get(name).add(time.perf_counter() - t0)

    def metrics(self, prefix: str = "time/") -> dict[str, float]:
        return {f"{prefix}{k}_s": t.last for k, t in self.timers.items()}

    def reset(self) -> None:
        self.timers.clear()
