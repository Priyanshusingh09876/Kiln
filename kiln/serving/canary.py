"""SLO guardrails for canary and shadow rollouts of a new model / engine build.

A :class:`CanaryController` receives per-request observations for the
*baseline* and the *canary* (latency, success, and optionally a quality score
such as reward-model agreement), keeps sliding windows, and decides every
``evaluate()`` whether to promote, hold, or roll back. Decisions are made on
percentiles over a minimum sample size so a single slow request cannot
trigger a rollback, and rollback is sticky (a tripped canary stays tripped
until a human resets it).

Traffic split is a ramp: ``promote()`` moves the canary share through
``ramp_steps`` (e.g. 1% -> 5% -> 25% -> 50% -> 100%), holding at each step
until ``min_samples`` observations pass the SLO.
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum


class SLOVerdict(str, Enum):
    PASS = "pass"
    INSUFFICIENT = "insufficient_data"
    LATENCY = "latency_breach"
    ERRORS = "error_rate_breach"
    QUALITY = "quality_regression"


@dataclass
class SLO:
    p99_latency_s: float = 0.5
    p50_latency_s: float | None = None
    max_error_rate: float = 0.005
    max_quality_drop: float | None = 0.02
    """Canary quality may be at most this much below baseline (absolute)."""
    max_latency_ratio: float | None = 1.2
    """Canary p99 may be at most this multiple of baseline p99."""
    max_p50_ratio: float | None = 1.3
    """Canary median may be at most this multiple of baseline median. Tails are noisy; a
    regression that shifts the whole distribution can hide inside a noisy baseline p99
    but not inside its median."""
    min_samples: int = 200
    warmup_samples: int = 0
    """Canary observations to ignore before judging it at all. A new replica starts with a
    cold prefix/KV cache and, at a 1-5% traffic share, warms slowly; judging it immediately
    against warm baseline replicas reports a regression that is really a cold cache."""


@dataclass
class _Window:
    maxlen: int
    latencies: deque[float] = field(init=False)
    errors: deque[bool] = field(init=False)
    quality: deque[float] = field(init=False)

    def __post_init__(self) -> None:
        self.latencies = deque(maxlen=self.maxlen)
        self.errors = deque(maxlen=self.maxlen)
        self.quality = deque(maxlen=self.maxlen)

    def add(self, latency_s: float, ok: bool, quality: float | None) -> None:
        self.latencies.append(latency_s)
        self.errors.append(not ok)
        if quality is not None:
            self.quality.append(quality)

    def __len__(self) -> int:
        return len(self.latencies)

    def pct(self, p: float) -> float:
        if not self.latencies:
            return math.nan
        s = sorted(self.latencies)
        return s[min(int(round(p / 100 * (len(s) - 1))), len(s) - 1)]

    def error_rate(self) -> float:
        return sum(self.errors) / max(len(self.errors), 1)

    def mean_quality(self) -> float | None:
        return None if not self.quality else sum(self.quality) / len(self.quality)


@dataclass
class CanaryState:
    share: float
    verdict: SLOVerdict
    tripped: bool
    step_index: int
    canary_p99: float
    baseline_p99: float
    canary_err: float
    detail: str = ""


class CanaryController:
    """Ramp a canary through ``ramp_steps``, judging it against a baseline window.

    With ``matched_control=True`` the baseline window is fed only by a *control* arm: a
    freshly started replica of the current build that receives exactly the same share
    of traffic as the canary (the Kayenta pattern). Without it the canary is compared
    with the warm fleet, which is biased against any new replica whose performance
    depends on cache state — kiln's own load test shows a healthy canary being rolled
    back for that reason (docs/benchmarks/gateway_demo.md).
    """

    def __init__(
        self,
        slo: SLO,
        ramp_steps: tuple[float, ...] = (0.01, 0.05, 0.25, 0.5, 1.0),
        window: int = 5000,
        matched_control: bool = False,
    ):
        self.slo = slo
        self.matched_control = matched_control
        self.ramp = ramp_steps
        self.step_index = 0
        self.baseline = _Window(window)
        self.canary = _Window(window)
        self.tripped = False
        self.history: list[CanaryState] = []
        self._last_change = time.monotonic()
        self.warmup_seen = 0

    # ------------------------------------------------------------ inputs
    @property
    def canary_share(self) -> float:
        return 0.0 if self.tripped else self.ramp[self.step_index]

    @property
    def control_share(self) -> float:
        """Traffic share of the matched control arm (equal to the canary's, capped so both fit)."""
        if not self.matched_control or self.tripped:
            return 0.0
        return min(self.canary_share, 1.0 - self.canary_share)

    def observe(self, arm: str, latency_s: float, ok: bool = True, quality: float | None = None) -> None:
        if arm == "canary":
            if self.warmup_seen < self.slo.warmup_samples:
                self.warmup_seen += 1  # cold-cache period: count it, do not judge it
                return
            self.canary.add(latency_s, ok, quality)
        elif arm == "control" or not self.matched_control:
            self.baseline.add(latency_s, ok, quality)
        # with a matched control, fleet ("baseline") traffic is served but not used for judgement

    def choose_arm(self, u: float) -> str:
        """Route a request given a uniform sample ``u`` in [0,1): canary, matched control, or fleet."""
        c = self.canary_share
        if u < c:
            return "canary"
        if u < c + self.control_share:
            return "control"
        return "baseline"

    # ----------------------------------------------------------- decision
    def evaluate(self) -> CanaryState:
        c, b = self.canary, self.baseline
        verdict, detail = SLOVerdict.PASS, ""
        if len(c) < self.slo.min_samples:
            verdict, detail = SLOVerdict.INSUFFICIENT, f"{len(c)}/{self.slo.min_samples} canary samples"
        else:
            c99, b99 = c.pct(99), b.pct(99)
            if c99 > self.slo.p99_latency_s:
                verdict, detail = SLOVerdict.LATENCY, f"canary p99 {c99:.3f}s > {self.slo.p99_latency_s}s"
            elif (
                self.slo.max_latency_ratio
                and len(b) >= self.slo.min_samples
                and c99 > b99 * self.slo.max_latency_ratio
            ):
                verdict, detail = (
                    SLOVerdict.LATENCY,
                    f"canary p99 {c99:.3f}s > {self.slo.max_latency_ratio}x baseline {b99:.3f}s",
                )
            elif (
                self.slo.max_p50_ratio
                and len(b) >= self.slo.min_samples
                and c.pct(50) > b.pct(50) * self.slo.max_p50_ratio
            ):
                verdict, detail = (
                    SLOVerdict.LATENCY,
                    f"canary p50 {c.pct(50):.3f}s > {self.slo.max_p50_ratio}x baseline p50 {b.pct(50):.3f}s",
                )
            elif self.slo.p50_latency_s and c.pct(50) > self.slo.p50_latency_s:
                verdict, detail = (
                    SLOVerdict.LATENCY,
                    f"canary p50 {c.pct(50):.3f}s > {self.slo.p50_latency_s}s",
                )
            elif c.error_rate() > self.slo.max_error_rate:
                verdict, detail = (
                    SLOVerdict.ERRORS,
                    f"canary error rate {c.error_rate():.4f} > {self.slo.max_error_rate}",
                )
            elif self.slo.max_quality_drop is not None:
                cq, bq = c.mean_quality(), b.mean_quality()
                if cq is not None and bq is not None and (bq - cq) > self.slo.max_quality_drop:
                    verdict, detail = SLOVerdict.QUALITY, f"canary quality {cq:.3f} vs baseline {bq:.3f}"
        if verdict in (SLOVerdict.LATENCY, SLOVerdict.ERRORS, SLOVerdict.QUALITY):
            self.tripped = True
        elif verdict is SLOVerdict.PASS and not self.tripped and self.step_index < len(self.ramp) - 1:
            self.step_index += 1
            self.canary = _Window(self.canary.maxlen)  # fresh window at the new traffic level
            if self.matched_control:  # the control moves to the same share, so its window restarts too
                self.baseline = _Window(self.baseline.maxlen)
            detail = f"promoted to {self.ramp[self.step_index]:.0%}"
        state = CanaryState(
            self.canary_share,
            verdict,
            self.tripped,
            self.step_index,
            c.pct(99),
            b.pct(99),
            c.error_rate(),
            detail,
        )
        self.history.append(state)
        return state

    def reset(self) -> None:
        self.tripped = False
        self.step_index = 0
        self.canary = _Window(self.canary.maxlen)

    @property
    def fully_promoted(self) -> bool:
        return not self.tripped and self.step_index == len(self.ramp) - 1
