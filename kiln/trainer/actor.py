"""PolicyTrainer: owns the policy (and optional critic / reference) and does optimizer steps.

Responsibilities, in the order they happen inside :meth:`train_step`:

1. Recompute ``old_logp`` (trainer-side behaviour policy for the clip ratio)
   and ``ref_logp`` under ``torch.no_grad`` — micro-batched, packed.
2. Compute advantages via the algorithm (GRPO group stats / GAE).
3. ``ppo_epochs`` passes of micro-batched forward/backward with gradient
   accumulation, gradient clipping, one optimizer step per pass.
4. Bump ``policy_version``.

Everything that is *not* an optimizer step (rollouts, rewards, weight sync,
checkpoint cadence) lives in the orchestrator. The trainer is deliberately
synchronous and single-purpose so it is testable with the tiny model on CPU
and so FSDP/TP specifics stay inside ``kiln.distributed``.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn

from kiln.algorithms import Algorithm, ForwardOutputs, build_algorithm
from kiln.config.schema import KilnConfig
from kiln.data.packing import pack, unpack
from kiln.distributed import ParallelState, all_reduce_metrics, clip_grad_norm, shard_model
from kiln.kernels import sequence_logprobs
from kiln.model import ValueHead, hidden_size_of, load_policy
from kiln.rollout.types import RolloutBatch


@dataclass
class StepTimings:
    old_logp_s: float = 0.0
    ref_logp_s: float = 0.0
    fwd_bwd_s: float = 0.0
    optim_s: float = 0.0
    total_s: float = 0.0
    tokens: int = 0

    def as_metrics(self) -> dict[str, float]:
        return {
            "time/old_logp_s": self.old_logp_s,
            "time/ref_logp_s": self.ref_logp_s,
            "time/fwd_bwd_s": self.fwd_bwd_s,
            "time/optim_s": self.optim_s,
            "time/step_s": self.total_s,
            "throughput/tokens_per_s": self.tokens / max(self.total_s, 1e-9),
        }


@dataclass
class PolicyTrainer:
    cfg: KilnConfig
    pstate: ParallelState
    model: nn.Module
    algorithm: Algorithm
    optimizer: torch.optim.Optimizer
    scheduler: Any
    ref_model: nn.Module | None = None
    critic: nn.Module | None = None
    critic_optimizer: torch.optim.Optimizer | None = None
    policy_version: int = 0
    step: int = 0
    pad_id: int = 0
    device: torch.device = field(default_factory=lambda: torch.device("cpu"))

    # ---------------------------------------------------------- building
    @classmethod
    def build(cls, cfg: KilnConfig, pstate: ParallelState, pad_id: int = 0) -> PolicyTrainer:
        device = pstate.device()
        model = load_policy(cfg.model, device)
        # snapshot the unsharded initial weights: the reference policy is a frozen copy of these,
        # and copying from the model *after* sharding would hand it DTensor shards
        init_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        param_dtype = torch.bfloat16 if cfg.model.dtype == "bf16" else torch.float32
        model = shard_model(
            model,
            pstate,
            cfg.parallel,
            param_dtype=param_dtype,
            activation_checkpointing=cfg.model.gradient_checkpointing and cfg.model.name_or_path != "tiny",
        )
        algo = build_algorithm(cfg.algorithm, cfg.async_)
        o = cfg.train.optim
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=o.lr,
            betas=o.betas,
            eps=o.eps,
            weight_decay=o.weight_decay,
            fused=device.type == "cuda",
        )
        scheduler = build_scheduler(optimizer, o.schedule, o.warmup_steps, cfg.train.total_steps)
        ref_model = None
        if cfg.algorithm.needs_reference:
            ref_cfg = cfg.reference or cfg.model
            ref_model = load_policy(ref_cfg, device)
            if ref_cfg is cfg.model or ref_cfg.name_or_path == cfg.model.name_or_path:
                # reference = frozen copy of the initial policy
                ref_model.load_state_dict(init_state, strict=False)
            ref_model = shard_model(
                ref_model, pstate, cfg.parallel, param_dtype=param_dtype, reshard_after_forward=False
            )
            ref_model.eval()
            for p in ref_model.parameters():
                p.requires_grad_(False)
        critic = critic_opt = None
        if cfg.algorithm.needs_critic:
            critic = ValueHead(hidden_size_of(model)).to(device)
            critic_opt = torch.optim.AdamW(critic.parameters(), lr=o.lr * 10, weight_decay=0.0)
        return cls(
            cfg,
            pstate,
            model,
            algo,
            optimizer,
            scheduler,
            ref_model,
            critic,
            critic_opt,
            pad_id=pad_id,
            device=device,
        )

    # ------------------------------------------------------------ forward
    def _forward_logprobs(self, model: nn.Module, mb: RolloutBatch, need_hidden: bool = False):
        """Return ``(logp, entropy, hidden|None)`` for a micro-batch, packed if configured."""
        T = mb.seq_len
        if self.cfg.train.pack_sequences and mb.batch_size > 1:
            capacity = max(int(mb.attention_mask.sum(-1).max().item()), 1)
            capacity = min(max(capacity * 2, 64), self.cfg.train.max_seq_len)
            capacity = max(capacity, int(mb.attention_mask.sum(-1).max().item()))
            packed = pack(mb.input_ids, mb.attention_mask, capacity, pad_id=self.pad_id)
            logits = model(
                packed.input_ids,
                attention_mask=packed.attention_mask,
                position_ids=packed.position_ids,
                block_mask=packed.block_diag_mask(),
            )
            resp_mask_packed = torch.zeros_like(packed.attention_mask)
            valid = packed.seq_index >= 0
            resp_mask_packed[valid] = mb.response_mask[packed.seq_index[valid], packed.position_ids[valid]]
            lp_p, ent_p = sequence_logprobs(logits, packed.input_ids, resp_mask_packed)
            logp = unpack(packed, lp_p, T)
            ent = unpack(packed, ent_p, T)
            hidden = None
            if need_hidden:
                h = (
                    model.forward_hidden(packed.input_ids, packed.attention_mask, packed.position_ids)
                    if hasattr(model, "forward_hidden")
                    else None
                )
                hidden = None if h is None else h
            return logp, ent, hidden
        logits = model(mb.input_ids, attention_mask=mb.attention_mask, position_ids=mb.position_ids)
        logp, ent = sequence_logprobs(logits, mb.input_ids, mb.response_mask)
        hidden = None
        if need_hidden and hasattr(model, "forward_hidden"):
            hidden = model.forward_hidden(mb.input_ids, mb.attention_mask, mb.position_ids)
        return logp, ent, hidden

    @torch.no_grad()
    def _nograd_logprobs(self, model: nn.Module, batch: RolloutBatch) -> torch.Tensor:
        outs = []
        for mb in batch.micro_batches(self.cfg.train.micro_batch_size):
            lp, _, _ = self._forward_logprobs(model, mb)
            outs.append(lp)
        return torch.cat(outs, 0)

    @torch.no_grad()
    def _values(self, batch: RolloutBatch) -> torch.Tensor:
        assert self.critic is not None
        outs = []
        for mb in batch.micro_batches(self.cfg.train.micro_batch_size):
            logits = self.model(mb.input_ids, attention_mask=mb.attention_mask, position_ids=mb.position_ids)
            # critic reads the pre-head features; for TinyLM reuse the normalised hidden via lm_head weight trick
            h = (
                logits @ torch.pinverse(self.model.lm_head.weight.float()).t()
                if hasattr(self.model, "lm_head")
                else logits
            )
            outs.append(self.critic(h[:, :-1].float()))
        return torch.cat(outs, 0)

    # --------------------------------------------------------------- step
    def train_step(self, batch: RolloutBatch) -> dict[str, float]:
        t_start = time.perf_counter()
        timings = StepTimings()
        batch = batch.to(self.device)
        mask = batch.token_mask().float()
        timings.tokens = int(mask.sum().item())
        self.model.train()

        t0 = time.perf_counter()
        old_logp = self._nograd_logprobs(self.model, batch)
        timings.old_logp_s = time.perf_counter() - t0

        ref_logp = None
        if self.ref_model is not None:
            t0 = time.perf_counter()
            ref_logp = self._nograd_logprobs(self.ref_model, batch)
            timings.ref_logp_s = time.perf_counter() - t0

        values = old_values = None
        if self.critic is not None:
            old_values = self._values(batch)
            values = old_values

        shaped = self.algorithm.shape_rewards(
            batch, ForwardOutputs(old_logp, torch.zeros_like(old_logp), old_logp, ref_logp)
        )
        if shaped is not None:
            batch.extra["shaped_rewards"] = shaped
        advantages = self.algorithm.compute_advantages(batch, values)

        mbs = self.cfg.train.micro_batch_size
        n_micro = math.ceil(batch.batch_size / mbs)
        metrics_acc: dict[str, torch.Tensor] = {}
        t0 = time.perf_counter()
        grad_norm = torch.zeros(())
        for _epoch in range(self.cfg.train.ppo_epochs):
            self.optimizer.zero_grad(set_to_none=True)
            if self.critic_optimizer is not None:
                self.critic_optimizer.zero_grad(set_to_none=True)
            for i, mb in enumerate(batch.micro_batches(mbs)):
                s, e = i * mbs, i * mbs + mb.batch_size
                logp, ent, hidden = self._forward_logprobs(
                    self.model, mb, need_hidden=self.critic is not None
                )
                v = None
                if self.critic is not None:
                    if hidden is None:
                        v = (
                            old_values[s:e].detach() + 0 * logp.sum()
                        )  # no hidden states available: critic frozen this step
                    else:
                        v = self.critic(hidden[:, :-1].float())
                out = ForwardOutputs(
                    logp=logp,
                    entropy=ent,
                    old_logp=old_logp[s:e],
                    ref_logp=None if ref_logp is None else ref_logp[s:e],
                    values=v,
                    old_values=None if old_values is None else old_values[s:e],
                )
                sub = batch.slice(s, e)
                sub.extra = {k: val[s:e] for k, val in batch.extra.items()}
                loss, m = self.algorithm.loss(sub, out, advantages[s:e])
                (loss / n_micro).backward()
                for k, val in m.items():
                    metrics_acc[k] = (
                        metrics_acc.get(k, 0.0) + val.detach().float() / n_micro / self.cfg.train.ppo_epochs
                    )
            grad_norm = clip_grad_norm(self.model, self.cfg.train.optim.grad_clip)
            t1 = time.perf_counter()
            self.optimizer.step()
            self.scheduler.step()
            if self.critic_optimizer is not None:
                self.critic_optimizer.step()
            timings.optim_s += time.perf_counter() - t1
        timings.fwd_bwd_s = time.perf_counter() - t0 - timings.optim_s

        self.step += 1
        self.policy_version += 1
        timings.total_s = time.perf_counter() - t_start
        metrics = all_reduce_metrics(metrics_acc, self.pstate.dp_group)
        metrics.update(timings.as_metrics())
        metrics["optim/grad_norm"] = float(grad_norm)
        metrics["optim/lr"] = float(self.scheduler.get_last_lr()[0])
        metrics["async/mean_staleness"] = float(batch.staleness(self.policy_version - 1).float().mean())
        metrics["policy/version"] = float(self.policy_version)
        metrics["rollout/mean_response_len"] = float(mask.sum(-1).mean())
        return metrics

    # ------------------------------------------------------------- state
    def state_dict(self) -> dict[str, Any]:
        from kiln.distributed import sharded_state_dict

        return sharded_state_dict(self.model)

    def extra_state(self) -> dict[str, Any]:
        s: dict[str, Any] = {
            "step": self.step,
            "policy_version": self.policy_version,
            "scheduler": self.scheduler.state_dict(),
        }
        if self.cfg.checkpoint.save_optimizer:
            s["optimizer"] = self.optimizer.state_dict()
        if self.critic is not None:
            s["critic"] = self.critic.state_dict()
        return s

    def load_state(self, model_state: dict[str, Any], extra: dict[str, Any]) -> None:
        from kiln.distributed import load_sharded_state_dict, rewrap_optimizer_state

        missing = load_sharded_state_dict(self.model, model_state)
        if missing:
            raise KeyError(f"checkpoint is missing {len(missing)} model keys, e.g. {missing[:3]}")
        self.step = int(extra.get("step", 0))
        self.policy_version = int(extra.get("policy_version", 0))
        if "optimizer" in extra:
            self.optimizer.load_state_dict(extra["optimizer"])
            rewrap_optimizer_state(self.optimizer)
        if "scheduler" in extra:
            self.scheduler.load_state_dict(extra["scheduler"])
        if self.critic is not None and "critic" in extra:
            self.critic.load_state_dict(extra["critic"])


def build_scheduler(optimizer: torch.optim.Optimizer, kind: str, warmup: int, total: int):
    def lr_lambda(step: int) -> float:
        if warmup > 0 and step < warmup:
            return (step + 1) / warmup
        if kind == "constant":
            return 1.0
        progress = min(max(step - warmup, 0) / max(total - warmup, 1), 1.0)
        if kind == "cosine":
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        if kind == "linear":
            return 1.0 - progress
        raise ValueError(kind)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
