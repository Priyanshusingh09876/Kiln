"""AsyncController: the training loop that ties everything together.

                 prompts                       rewards
   PromptDataset ───────► RolloutWorker ×K ◄──────── reward fn / server
                              │ engine.generate()
                              ▼  put(batch, version)
                        ExperienceBuffer  (bounded, staleness-aware)
                              │  get(current_version)
                              ▼
                        PolicyTrainer.train_step()  ──► metrics / checkpoints
                              │ policy_version += 1
                              ▼  WeightSyncer.sync()  (inproc / cpu / nccl)
                        RolloutEngine.update_weights()

The loop body is :meth:`AsyncController.train_one_step`; :meth:`run` wraps it
with checkpoint cadence, logging and the adaptive queue-depth controller.
Synchronous (on-policy) training is the degenerate case ``queue_depth=1,
max_staleness=0`` with the worker loop called inline, which is what the
tests use for exact-reproducibility checks.

The controller is intentionally free of Ray / torch.distributed specifics:
``workers`` is any list of objects with ``start/stop/metrics``, ``buffer``
anything with ``put/get/metrics``. :mod:`kiln.orchestration.ray_runtime`
provides Ray-backed versions of both.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from kiln.checkpoint import AsyncCheckpointer
from kiln.config.schema import KilnConfig
from kiln.data.prompts import ByteTokenizer, PromptDataset
from kiln.distributed import ParallelState, init_parallel
from kiln.model import load_policy
from kiln.orchestration.experience_buffer import ExperienceBuffer
from kiln.orchestration.rollout_worker import RolloutWorker
from kiln.orchestration.staleness import StalenessController
from kiln.reward import build_reward_fn
from kiln.rollout import WeightSyncer, build_engine
from kiln.rollout.types import RolloutBatch, concat_batches
from kiln.trainer.actor import PolicyTrainer
from kiln.utils.logging import MetricLogger, build_logger, setup_logging
from kiln.utils.mfu import MFUCalculator

log = logging.getLogger("kiln.controller")


@dataclass
class RunResult:
    steps: int
    final_metrics: dict[str, float]
    history: list[dict[str, float]] = field(default_factory=list)
    run_dir: Path | None = None

    def series(self, key: str) -> list[float]:
        return [m[key] for m in self.history if key in m]


class AsyncController:
    def __init__(
        self,
        cfg: KilnConfig,
        trainer: PolicyTrainer,
        workers: list[Any],
        buffer: Any,
        syncer: WeightSyncer,
        checkpointer: AsyncCheckpointer | None,
        logger: MetricLogger | None,
        pstate: ParallelState,
        on_step: Callable[[int, dict[str, float]], None] | None = None,
    ):
        self.cfg = cfg
        self.trainer = trainer
        self.workers = workers
        self.buffer = buffer
        self.syncer = syncer
        self.ckpt = checkpointer
        self.logger = logger
        self.pstate = pstate
        self.on_step = on_step
        a = cfg.async_
        self.depth_ctl = StalenessController(
            depth=a.queue_depth, max_depth=max(a.queue_depth, 1) * 4, max_staleness=max(a.max_staleness, 1)
        )
        self.mfu = MFUCalculator.from_model(
            trainer.model, pstate.world_size, has_reference=trainer.ref_model is not None
        )
        self.history: list[dict[str, float]] = []

    # ---------------------------------------------------------------- build
    @classmethod
    def from_config(
        cls,
        cfg: KilnConfig,
        tokenizer: Any | None = None,
        rows: list[dict[str, Any]] | None = None,
        reward_fn: Callable | None = None,
        on_step: Callable[[int, dict[str, float]], None] | None = None,
        logger: MetricLogger | None = None,
    ) -> AsyncController:
        """Local (single-process) wiring: workers are threads, engines are in-process."""
        setup_logging()
        torch.manual_seed(cfg.train.seed)
        pstate = init_parallel(cfg.parallel)
        if tokenizer is None:
            tokenizer = (
                ByteTokenizer(cfg.model.tiny_vocab)
                if cfg.model.name_or_path == "tiny"
                else _hf_tokenizer(cfg)
            )
        trainer = PolicyTrainer.build(cfg, pstate, pad_id=tokenizer.pad_id)
        dataset = PromptDataset(cfg.data, tokenizer, rows=rows)
        reward_fn = reward_fn or build_reward_fn(cfg.reward)

        buffer = ExperienceBuffer(capacity=cfg.async_.queue_depth, max_staleness=cfg.async_.max_staleness)
        n_workers = max(cfg.rollout.num_engines, 1)
        prompts_per_batch = max(cfg.train.rollout_batch_size // cfg.rollout.sampling.n, 1)
        if not cfg.async_.enabled and n_workers > 1:
            # synchronous mode splits each training batch across all engines and generates the
            # parts in parallel (as verl/OpenRLHF do); async engines each produce whole batches
            if prompts_per_batch % n_workers:
                raise ValueError(
                    f"sync mode with {n_workers} engines needs rollout_batch_size/n "
                    f"({prompts_per_batch} prompts) divisible by the number of engines"
                )
            prompts_per_batch //= n_workers
        # every (data-parallel rank, worker) pair gets a disjoint prompt shard and its own sampler seed
        dp_rank, dp_size = pstate.dp_rank, max(cfg.parallel.dp, 1)
        workers = []
        for i in range(n_workers):
            shard = dp_rank * n_workers + i
            engine = build_engine(
                cfg.rollout,
                # engines hold their own *unsharded* copy (like a real inference server); the
                # syncer below loads the trainer's current weights into it before any rollout
                model=load_policy(cfg.model, "cpu") if cfg.rollout.engine == "mock" else None,
                model_path=cfg.model.name_or_path,
                eos_id=tokenizer.eos_id,
                seed=cfg.train.seed + shard,
            )
            prompts = dataset.batches(
                prompts_per_batch, seed=cfg.train.seed, rank=shard, world_size=dp_size * n_workers
            )
            workers.append(
                RolloutWorker(i, cfg, engine, prompts, reward_fn, buffer, tokenizer, prompts_per_batch)
            )
        syncer = WeightSyncer([w.engine for w in workers], transport="inproc")
        # collective under FSDP (gathers full weights), so every rank calls it in lockstep here
        syncer.sync(trainer.model, trainer.policy_version, pause_engines=False)
        ckpt = AsyncCheckpointer(cfg.checkpoint) if cfg.checkpoint.interval > 0 else None
        if logger is None:
            logger = build_logger(cfg.logging, is_main=pstate.is_main)
        ctl = cls(cfg, trainer, workers, buffer, syncer, ckpt, logger, pstate, on_step=on_step)
        ctl.tokenizer = tokenizer
        ctl.dataset = dataset
        return ctl

    # ---------------------------------------------------------------- loop
    def maybe_resume(self) -> int:
        if self.ckpt is None or not self.cfg.checkpoint.resume:
            return 0
        step = self.ckpt.latest_step()
        if step is None:
            return 0
        model_state, extra, _meta = self.ckpt.load(step, map_location=self.trainer.device)
        self.trainer.load_state(model_state, extra)
        for w, ws in zip(self.workers, extra.get("workers", [])):
            if hasattr(w, "load_state_dict"):
                w.load_state_dict(ws)
        # engines were built from the *initial* weights: push the resumed policy before any rollout,
        # otherwise every new sample is tagged version 0 and dropped as stale (async) or mis-weighted (sync)
        self.syncer.sync(self.trainer.model, self.trainer.policy_version, pause_engines=False)
        log.info("resumed from step %d (policy version %d)", self.trainer.step, self.trainer.policy_version)
        return self.trainer.step

    def _extra_state(self) -> dict[str, Any]:
        extra = self.trainer.extra_state()
        extra["workers"] = [w.state_dict() for w in self.workers if hasattr(w, "state_dict")]
        return extra

    def _next_batch(self, timeout: float | None) -> RolloutBatch | None:
        return self.buffer.get(self.trainer.policy_version, timeout=timeout)

    def train_one_step(self, batch: RolloutBatch) -> dict[str, float]:
        t0 = time.perf_counter()
        metrics = self.trainer.train_step(batch)
        train_s = time.perf_counter() - t0
        sync = self.syncer.sync(
            self.trainer.model, self.trainer.policy_version, pause_engines=self.cfg.rollout.partial_rollout
        )
        metrics["sync/seconds"] = sync.seconds
        metrics["sync/gbps"] = self.syncer.gbps()
        metrics["sync/bytes"] = float(sync.bytes)
        tokens = int(batch.token_mask().sum().item())
        metrics["perf/mfu"] = self.mfu.mfu(tokens, metrics["time/step_s"], batch.seq_len)
        # adaptive depth: effective wall time per rollout batch with K workers in parallel is mean(gen_s)/K
        worker_gen = [w.metrics().get("rollout/gen_s", 0.0) for w in self.workers]
        k = max(len(worker_gen), 1)
        rollout_s = (sum(worker_gen) / k) / k if worker_gen else None
        self.depth_ctl.observe(rollout_seconds=rollout_s, train_seconds=train_s)
        if self.cfg.async_.adaptive_depth and self.cfg.async_.enabled:
            new_depth = self.depth_ctl.step()
            if new_depth != self.buffer.capacity:
                self.buffer.resize(new_depth)
        metrics.update(self.depth_ctl.metrics())
        metrics.update(self.buffer.metrics())
        for w in self.workers:
            metrics.update(w.metrics())
        return metrics

    def run(self, max_steps: int | None = None) -> RunResult:
        total = max_steps or self.cfg.train.total_steps
        start = self.maybe_resume()
        async_mode = self.cfg.async_.enabled
        if async_mode:
            for w in self.workers:
                w.start()
        metrics: dict[str, float] = {}
        try:
            busy_total = wait_total = 0.0
            t_run = time.perf_counter()
            for _ in range(start, total):
                # time the trainer spends *not* training: waiting on rollouts (sync: generating them inline)
                t_wait = time.perf_counter()
                if async_mode:
                    batch = None
                    while batch is None:
                        self._check_workers()
                        batch = self._next_batch(timeout=5.0)
                else:
                    batch = self._sync_rollout()
                t_busy = time.perf_counter()
                wait_s = t_busy - t_wait
                metrics = self.train_one_step(batch)
                step = self.trainer.step
                if (
                    self.ckpt is not None
                    and self.cfg.checkpoint.interval > 0
                    and step % self.cfg.checkpoint.interval == 0
                ):
                    st = self.ckpt.save(step, self.trainer.state_dict(), self._extra_state())
                    metrics["ckpt/copy_s"] = st.copy_seconds
                busy_s = time.perf_counter() - t_busy
                busy_total += busy_s
                wait_total += wait_s
                metrics["time/data_wait_s"] = wait_s
                metrics["perf/trainer_busy_frac"] = busy_s / max(busy_s + wait_s, 1e-9)
                metrics["perf/trainer_busy_frac_cum"] = busy_total / max(busy_total + wait_total, 1e-9)
                metrics["time/wall_s"] = time.perf_counter() - t_run
                self.history.append(dict(metrics))
                if self.logger is not None and step % self.cfg.train.log_interval == 0:
                    self.logger.log(step, metrics)
                if self.on_step is not None:
                    self.on_step(step, metrics)
            if self.ckpt is not None and self.trainer.step % max(self.cfg.checkpoint.interval, 1) != 0:
                self.ckpt.save(self.trainer.step, self.trainer.state_dict(), self._extra_state())
        finally:
            self.shutdown()
        return RunResult(self.trainer.step, metrics, self.history, getattr(self.logger, "run_dir", None))

    def _sync_rollout(self) -> RolloutBatch:
        """On-policy mode: every engine generates its slice of the batch in parallel; the step
        waits for the slowest slice (that wait is exactly what async mode removes)."""
        if len(self.workers) == 1:
            return self.workers[0].produce_batch()[0]
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(len(self.workers)) as ex:
            parts = list(ex.map(lambda w: w.produce_batch()[0], self.workers))
        return concat_batches(parts, pad_id=self.tokenizer.pad_id)

    def _check_workers(self) -> None:
        for w in self.workers:
            err = getattr(w, "error", None)
            if err is not None:
                raise RuntimeError(f"rollout worker {w.id} failed") from err

    def shutdown(self) -> None:
        self.buffer.close()
        for w in self.workers:
            w.stop(join=True)
            w.engine.shutdown()
        if self.ckpt is not None:
            self.ckpt.wait()
        if self.logger is not None:
            self.logger.close()


def _hf_tokenizer(cfg: KilnConfig):
    from kiln.data.prompts import HFTokenizerAdapter

    return HFTokenizerAdapter(cfg.model.name_or_path, chat_template=cfg.data.chat_template)
