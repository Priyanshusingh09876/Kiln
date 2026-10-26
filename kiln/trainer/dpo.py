"""Offline preference optimisation (DPO / IPO / hinge) on chosen-rejected pairs.

No rollouts: each step takes ``train.rollout_batch_size`` pairs from a
:class:`PreferenceDataset`, computes summed response log-probs under the policy
(with grad) and the frozen reference (no grad), and applies
:meth:`kiln.algorithms.DPO.pair_loss`. It reuses :class:`PolicyTrainer` for the
model, FSDP wrapping, optimizer, scheduler, packing and checkpoint state, so
DPO runs get the same sharding, async checkpointing and exact resume as RL runs.
"""

from __future__ import annotations

import logging
import random
import time
from typing import Any

import torch

from kiln.checkpoint import AsyncCheckpointer
from kiln.config.schema import KilnConfig
from kiln.data.prompts import ByteTokenizer, PreferenceDataset, PreferencePair
from kiln.distributed import all_reduce_metrics, clip_grad_norm, init_parallel
from kiln.orchestration.controller import RunResult
from kiln.rollout.types import Sample, collate_samples
from kiln.trainer.actor import PolicyTrainer
from kiln.utils.logging import MetricLogger, build_logger, setup_logging

log = logging.getLogger("kiln.dpo")


def _as_batch(pairs: list[PreferencePair], which: str, pad_id: int, max_len: int):
    samples = []
    for i, p in enumerate(pairs):
        resp = p.chosen_ids if which == "chosen" else p.rejected_ids
        samples.append(
            Sample(list(p.prompt_ids), list(resp), [0.0] * len(resp), group_id=i, policy_version=0)
        )
    return collate_samples(samples, pad_id=pad_id, max_len=max_len)


class DPORunner:
    def __init__(
        self,
        cfg: KilnConfig,
        trainer: PolicyTrainer,
        dataset: PreferenceDataset,
        ckpt: AsyncCheckpointer | None,
        logger: MetricLogger | None,
    ):
        assert cfg.algorithm.name == "dpo"
        assert trainer.ref_model is not None, "DPO needs a reference model"
        self.cfg, self.trainer, self.data, self.ckpt, self.logger = cfg, trainer, dataset, ckpt, logger
        self.history: list[dict[str, float]] = []
        self._order: list[int] = []

    @classmethod
    def from_config(
        cls,
        cfg: KilnConfig,
        rows: list[dict] | None = None,
        tokenizer: Any = None,
        logger: MetricLogger | None = None,
    ) -> DPORunner:
        setup_logging()
        torch.manual_seed(cfg.train.seed)
        pstate = init_parallel(cfg.parallel)
        if tokenizer is None:
            if cfg.model.name_or_path == "tiny":
                tokenizer = ByteTokenizer(cfg.model.tiny_vocab)
            else:
                from kiln.data.prompts import HFTokenizerAdapter

                tokenizer = HFTokenizerAdapter(cfg.model.name_or_path, chat_template=cfg.data.chat_template)
        trainer = PolicyTrainer.build(cfg, pstate, pad_id=tokenizer.pad_id)
        data = PreferenceDataset(
            cfg.data.train_path if rows is None else None, tokenizer, max_len=cfg.train.max_seq_len, rows=rows
        )
        ckpt = AsyncCheckpointer(cfg.checkpoint) if cfg.checkpoint.interval > 0 else None
        logger = logger or build_logger(cfg.logging, is_main=pstate.is_main)
        r = cls(cfg, trainer, data, ckpt, logger)
        r.pad_id = tokenizer.pad_id
        return r

    # -------------------------------------------------------------- data
    def _batch(self, step: int) -> list[PreferencePair]:
        """Deterministic epoch-shuffled order, sharded by DP rank; a pure function of ``step`` (exact resume)."""
        pst = self.trainer.pstate
        bs = self.cfg.train.rollout_batch_size // max(self.cfg.parallel.dp, 1)
        start = step * self.cfg.train.rollout_batch_size + pst.dp_rank * bs
        n = len(self.data)
        out = []
        for k in range(start, start + bs):
            epoch, idx = divmod(k, n)
            order = list(range(n))
            random.Random(self.cfg.train.seed + epoch).shuffle(order)
            out.append(self.data[order[idx]])
        return out

    # -------------------------------------------------------------- step
    def train_step(self, pairs: list[PreferencePair]) -> dict[str, float]:
        tr, cfg = self.trainer, self.cfg
        t0 = time.perf_counter()
        tr.model.train()
        mbs = cfg.train.micro_batch_size
        chosen = _as_batch(pairs, "chosen", self.pad_id, cfg.train.max_seq_len).to(tr.device)
        rejected = _as_batch(pairs, "rejected", self.pad_id, cfg.train.max_seq_len).to(tr.device)
        n_micro = (len(pairs) + mbs - 1) // mbs
        tr.optimizer.zero_grad(set_to_none=True)
        acc: dict[str, torch.Tensor] = {}
        for i in range(n_micro):
            s, e = i * mbs, min((i + 1) * mbs, len(pairs))
            c, r = chosen.slice(s, e), rejected.slice(s, e)
            with torch.no_grad():
                ref_c, _, _ = tr._forward_logprobs(tr.ref_model, c)
                ref_r, _, _ = tr._forward_logprobs(tr.ref_model, r)
            lp_c, _, _ = tr._forward_logprobs(tr.model, c)
            lp_r, _, _ = tr._forward_logprobs(tr.model, r)
            loss, m = tr.algorithm.pair_loss(
                lp_c, c.token_mask().float(), lp_r, r.token_mask().float(), ref_c, ref_r
            )
            (loss / n_micro).backward()
            for k, v in m.items():
                acc[k] = acc.get(k, 0.0) + v.detach().float() / n_micro
        grad_norm = clip_grad_norm(tr.model, cfg.train.optim.grad_clip)
        tr.optimizer.step()
        tr.scheduler.step()
        tr.step += 1
        tr.policy_version += 1
        metrics = all_reduce_metrics(acc, tr.pstate.dp_group)
        metrics.update(
            {
                "optim/grad_norm": float(grad_norm),
                "optim/lr": float(tr.scheduler.get_last_lr()[0]),
                "time/step_s": time.perf_counter() - t0,
            }
        )
        return metrics

    def run(self, max_steps: int | None = None) -> RunResult:
        total = max_steps or self.cfg.train.total_steps
        if self.ckpt is not None and self.cfg.checkpoint.resume and self.ckpt.latest_step() is not None:
            ms, extra, _ = self.ckpt.load(map_location=self.trainer.device)
            self.trainer.load_state(ms, extra)
            log.info("resumed DPO at step %d", self.trainer.step)
        metrics: dict[str, float] = {}
        try:
            while self.trainer.step < total:
                metrics = self.train_step(self._batch(self.trainer.step))
                step = self.trainer.step
                self.history.append(dict(metrics))
                if self.logger is not None and step % self.cfg.train.log_interval == 0:
                    self.logger.log(step, metrics)
                if self.ckpt is not None and step % self.cfg.checkpoint.interval == 0:
                    self.ckpt.save(step, self.trainer.state_dict(), self.trainer.extra_state())
            if self.ckpt is not None and self.trainer.step % max(self.cfg.checkpoint.interval, 1):
                self.ckpt.save(self.trainer.step, self.trainer.state_dict(), self.trainer.extra_state())
        finally:
            if self.ckpt is not None:
                self.ckpt.wait()
            if self.logger is not None:
                self.logger.close()
        return RunResult(self.trainer.step, metrics, self.history, getattr(self.logger, "run_dir", None))
