"""Bounded, version-aware experience buffer between rollout workers and the trainer.

This is the piece that makes training asynchronous. Rollout workers ``put``
batches tagged with the policy version they were generated under; the
trainer ``get``s the oldest batch whose staleness ``(current_version -
batch_version)`` is within ``max_staleness`` and drops anything older,
recording how much was discarded so the operator can see when the queue
depth is mis-sized.

Back-pressure: ``put`` blocks while the buffer holds ``capacity`` batches. In
a healthy run the buffer hovers near full (rollouts are the bottleneck, GPUs
on the trainer side are never starved) and drops are ~0. If drops climb,
either rollout throughput is too high relative to training (shrink
``capacity`` to save compute) or ``max_staleness`` is too tight.

Thread-safe; the same class is used in-process (threads) and inside a Ray
actor (one actor, many callers).
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field

from kiln.rollout.types import RolloutBatch


@dataclass
class BufferStats:
    puts: int = 0
    gets: int = 0
    dropped_batches: int = 0
    dropped_samples: int = 0
    put_wait_seconds: float = 0.0
    get_wait_seconds: float = 0.0
    staleness_hist: dict[int, int] = field(default_factory=dict)

    def as_metrics(self, size: int, capacity: int) -> dict[str, float]:
        total_gets = max(self.gets, 1)
        mean_stale = sum(k * v for k, v in self.staleness_hist.items()) / total_gets
        return {
            "buffer/size": float(size),
            "buffer/capacity": float(capacity),
            "buffer/fill": size / max(capacity, 1),
            "buffer/dropped_batches": float(self.dropped_batches),
            "buffer/dropped_samples": float(self.dropped_samples),
            "buffer/put_wait_s": self.put_wait_seconds,
            "buffer/get_wait_s": self.get_wait_seconds,
            "buffer/mean_staleness": mean_stale,
        }


class ExperienceBuffer:
    def __init__(self, capacity: int, max_staleness: int):
        self.capacity = max(capacity, 1)
        self.max_staleness = max_staleness
        self._q: deque[tuple[int, RolloutBatch]] = deque()
        self._cv = threading.Condition()
        self._closed = False
        self.stats = BufferStats()

    # ------------------------------------------------------------- sizing
    def resize(self, capacity: int) -> None:
        with self._cv:
            self.capacity = max(capacity, 1)
            self._cv.notify_all()

    def __len__(self) -> int:
        with self._cv:
            return len(self._q)

    def close(self) -> None:
        with self._cv:
            self._closed = True
            self._cv.notify_all()

    @property
    def closed(self) -> bool:
        return self._closed

    # ---------------------------------------------------------------- put
    def put(self, batch: RolloutBatch, version: int, timeout: float | None = None) -> bool:
        t0 = time.perf_counter()
        with self._cv:
            while len(self._q) >= self.capacity and not self._closed:
                if not self._cv.wait(timeout):
                    self.stats.put_wait_seconds += time.perf_counter() - t0
                    return False
            if self._closed:
                return False
            self._q.append((version, batch))
            self.stats.puts += 1
            self.stats.put_wait_seconds += time.perf_counter() - t0
            self._cv.notify_all()
            return True

    # ---------------------------------------------------------------- get
    def get(self, current_version: int, timeout: float | None = None) -> RolloutBatch | None:
        """Return the oldest sufficiently-fresh batch, discarding stale ones. ``None`` on timeout/close."""
        t0 = time.perf_counter()
        deadline = None if timeout is None else t0 + timeout
        with self._cv:
            while True:
                self._drop_stale_locked(current_version)
                if self._q:
                    version, batch = self._q.popleft()
                    self.stats.gets += 1
                    s = current_version - version
                    self.stats.staleness_hist[s] = self.stats.staleness_hist.get(s, 0) + 1
                    self.stats.get_wait_seconds += time.perf_counter() - t0
                    self._cv.notify_all()
                    return batch
                if self._closed:
                    return None
                remaining = None if deadline is None else deadline - time.perf_counter()
                if remaining is not None and remaining <= 0:
                    self.stats.get_wait_seconds += time.perf_counter() - t0
                    return None
                self._cv.wait(remaining)

    def _drop_stale_locked(self, current_version: int) -> None:
        while self._q and (current_version - self._q[0][0]) > self.max_staleness:
            _, b = self._q.popleft()
            self.stats.dropped_batches += 1
            self.stats.dropped_samples += b.batch_size

    def metrics(self) -> dict[str, float]:
        with self._cv:
            return self.stats.as_metrics(len(self._q), self.capacity)
