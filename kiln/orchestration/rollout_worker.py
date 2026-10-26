"""RolloutWorker: prompts -> engine -> rewards -> collated batch -> experience buffer.

One worker drives one rollout engine (which may itself be a TP group of
GPUs). The loop is deliberately simple::

    while not stopped:
        prompts = next(prompt_batches)
        samples = engine.generate(prompts)         # tagged with engine.policy_version
        rewards = reward_fn(samples)               # rules / HTTP server / reward model
        batch   = collate(samples)
        buffer.put(batch, version)                 # blocks when buffer is full (back-pressure)

The worker never talks to the trainer directly. Weight updates arrive via
``engine.update_weights`` (called by the controller's :class:`WeightSyncer`),
so a worker mid-``generate`` simply keeps going and the next batch carries
the new version. With ``partial_rollout`` the engine is paused at a token
boundary during the swap; without it the swap waits for ``generate`` to
return.

The same class runs as a thread (local / tests) or inside a Ray actor
(``kiln.orchestration.ray_runtime``) — the only difference is who constructs
the engine and how ``buffer`` is reached.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any

from kiln.config.schema import KilnConfig
from kiln.data.prompts import Prompt
from kiln.rollout.base import GenerationRequest, RolloutEngine
from kiln.rollout.types import Sample, collate_samples

log = logging.getLogger("kiln.rollout")

RewardFn = Callable[[list[dict[str, Any]]], tuple[list[float], list[dict[str, float]]]]


@dataclass
class WorkerStats:
    batches: int = 0
    samples: int = 0
    response_tokens: int = 0
    gen_seconds: float = 0.0
    reward_seconds: float = 0.0
    put_seconds: float = 0.0
    truncated: int = 0
    reward_hist: dict[str, float] = field(default_factory=dict)

    def as_metrics(self, prefix: str = "rollout/") -> dict[str, float]:
        n = max(self.samples, 1)
        return {
            f"{prefix}batches": float(self.batches),
            f"{prefix}samples": float(self.samples),
            f"{prefix}gen_s": self.gen_seconds / max(self.batches, 1),
            f"{prefix}reward_s": self.reward_seconds / max(self.batches, 1),
            f"{prefix}put_wait_s": self.put_seconds / max(self.batches, 1),
            f"{prefix}resp_tok_per_s": self.response_tokens / max(self.gen_seconds, 1e-9),
            f"{prefix}truncated_frac": self.truncated / n,
            **{f"reward/{k}": v / n for k, v in self.reward_hist.items()},
        }


class RolloutWorker:
    def __init__(
        self,
        worker_id: int,
        cfg: KilnConfig,
        engine: RolloutEngine,
        prompts: Iterator[list[Prompt]],
        reward_fn: RewardFn,
        buffer: Any,
        tokenizer: Any,
        prompts_per_batch: int,
    ):
        self.id = worker_id
        self.cfg = cfg
        self.engine = engine
        self.prompts = prompts
        self.reward_fn = reward_fn
        self.buffer = buffer
        self.tok = tokenizer
        self.prompts_per_batch = prompts_per_batch
        self.stats = WorkerStats()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.error: BaseException | None = None
        self.batches_produced = 0

    # ------------------------------------------------------------ one batch
    def produce_batch(self):
        """Generate, score and collate one batch. Returns ``(batch, version)``."""
        prompts = next(self.prompts)
        self.batches_produced += 1
        sampling = self.cfg.rollout.sampling
        reqs = [
            GenerationRequest(
                prompt_ids=p.ids,
                group_id=p.uid,
                prompt_text=p.text,
                n=sampling.n,
                meta={"answer": p.answer, **p.meta},
            )
            for p in prompts
        ]
        t0 = time.perf_counter()
        samples = self.engine.generate(reqs, sampling)
        t1 = time.perf_counter()
        self._score(samples)
        t2 = time.perf_counter()
        version = min(s.policy_version for s in samples)  # conservative under partial rollouts
        batch = collate_samples(samples, pad_id=self.tok.pad_id, max_len=self.cfg.train.max_seq_len)
        self.stats.batches += 1
        self.stats.samples += len(samples)
        self.stats.response_tokens += sum(len(s.response_ids) for s in samples)
        self.stats.truncated += sum(1 for s in samples if s.finish_reason == "length")
        self.stats.gen_seconds += t1 - t0
        self.stats.reward_seconds += t2 - t1
        return batch, version

    def _score(self, samples: list[Sample]) -> None:
        items = []
        for s in samples:
            if not s.response_text:
                s.response_text = self.tok.decode(s.response_ids)
            items.append(
                {
                    "response": s.response_text,
                    "answer": s.meta.get("answer"),
                    "prompt": s.prompt_text,
                    "meta": s.meta,
                }
            )
        rewards, breakdown = self.reward_fn(items)
        for s, r, b in zip(samples, rewards, breakdown):
            s.reward = float(r)
            s.reward_breakdown = dict(b)
            for k, v in b.items():
                self.stats.reward_hist[k] = self.stats.reward_hist.get(k, 0.0) + float(v)

    # ---------------------------------------------------------------- loop
    def run(self) -> None:
        try:
            while not self._stop.is_set():
                batch, version = self.produce_batch()
                t0 = time.perf_counter()
                ok = self.buffer.put(batch, version, timeout=1.0)
                while not ok and not self._stop.is_set():
                    ok = self.buffer.put(batch, version, timeout=1.0)
                self.stats.put_seconds += time.perf_counter() - t0
        except BaseException as e:  # noqa: BLE001 - surfaced to the controller
            self.error = e
            log.exception("rollout worker %d died", self.id)
        finally:
            log.debug("rollout worker %d exiting", self.id)

    def start(self) -> None:
        self._thread = threading.Thread(target=self.run, name=f"kiln-rollout-{self.id}", daemon=True)
        self._thread.start()

    def stop(self, join: bool = True) -> None:
        self._stop.set()
        if join and self._thread is not None:
            self._thread.join(timeout=30)

    # --------------------------------------------------------------- resume
    def state_dict(self) -> dict:
        """Position in the (deterministic, seeded) prompt stream + engine sampler state.

        In synchronous mode this makes a resumed run bit-identical to an
        uninterrupted one. In asynchronous mode batches that were in flight
        (generated, not yet trained on) at checkpoint time are skipped on resume.
        """
        return {"batches_produced": self.batches_produced, "engine": self.engine.state_dict()}

    def load_state_dict(self, state: dict) -> None:
        n = int(state.get("batches_produced", 0))
        for _ in range(n - self.batches_produced):
            next(self.prompts)
        self.batches_produced = max(n, self.batches_produced)
        self.engine.load_state_dict(state.get("engine", {}))

    def metrics(self) -> dict[str, float]:
        m = self.stats.as_metrics()
        m.update(self.engine.stats())
        return m
