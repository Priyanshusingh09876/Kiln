"""Trainer -> rollout-engine weight synchronisation.

Three transports, chosen by where the engines live:

* ``inproc``  — engines in the same process (mock / tests): hand tensors over.
* ``cpu``     — gather full fp16/bf16 weights on the trainer, ship CPU tensors
                through Ray object store / RPC. Simple, bandwidth-bound by the
                host, fine for < 10B models or infrequent sync.
* ``nccl``    — a dedicated NCCL group containing trainer rank 0 and every
                engine TP worker. Weights are broadcast directly GPU->GPU in
                ~1 GB buckets. This is the production path.

All transports share :func:`iter_sync_tensors`, which handles the two
annoying details: FSDP2 parameters are DTensors that must be gathered before
they can be sent, and the parameter names a HF checkpoint uses
(``model.layers.0...``) differ from what ``nn.Module.state_dict`` yields once
the model is wrapped (``model.model.layers.0...``), so we strip wrapper
prefixes to canonical HF names before shipping.

LoRA-only sync is supported transparently: pass ``filter_fn`` selecting
adapter tensors and the transfer drops from tens of GB to tens of MB.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field

import torch
import torch.nn as nn

from kiln.distributed.comm import plan_buckets
from kiln.distributed.fsdp import full_state_dict
from kiln.rollout.base import RolloutEngine

_WRAPPER_PREFIXES = ("_fsdp_wrapped_module.", "_checkpoint_wrapped_module.", "module.", "_orig_mod.")


def canonical_name(name: str, strip_leading: str = "") -> str:
    for p in _WRAPPER_PREFIXES:
        name = name.replace(p, "")
    if strip_leading and name.startswith(strip_leading):
        name = name[len(strip_leading) :]
    return name


def iter_sync_tensors(
    model: nn.Module,
    dtype: torch.dtype | None = None,
    filter_fn: Callable[[str], bool] | None = None,
    strip_leading: str = "",
) -> Iterator[tuple[str, torch.Tensor]]:
    sd = full_state_dict(model)
    for name, t in sd.items():
        cname = canonical_name(name, strip_leading)
        if filter_fn is not None and not filter_fn(cname):
            continue
        if hasattr(t, "full_tensor"):
            t = t.full_tensor()
        t = t.detach()
        if dtype is not None and t.is_floating_point():
            t = t.to(dtype)
        yield cname, t


@dataclass
class SyncStats:
    version: int
    seconds: float
    bytes: int
    num_tensors: int
    num_buckets: int
    transport: str


@dataclass
class WeightSyncer:
    engines: list[RolloutEngine]
    transport: str = "inproc"
    dtype: torch.dtype | None = None
    bucket_bytes: int = 1 << 30
    filter_fn: Callable[[str], bool] | None = None
    strip_leading: str = ""
    history: list[SyncStats] = field(default_factory=list)

    def sync(self, model: nn.Module, version: int, pause_engines: bool = True) -> SyncStats:
        t0 = time.perf_counter()
        tensors = list(iter_sync_tensors(model, self.dtype, self.filter_fn, self.strip_leading))
        nbytes = sum(t.numel() * t.element_size() for _, t in tensors)
        nbuckets = len(plan_buckets(tensors, self.bucket_bytes))
        if pause_engines:
            for e in self.engines:
                e.pause()
        try:
            if self.transport in ("inproc", "cpu"):
                payload = tensors if self.transport == "inproc" else [(n, t.cpu()) for n, t in tensors]
                for e in self.engines:
                    e.update_weights(payload, version)
            elif self.transport == "nccl":
                self._sync_nccl(tensors, version)
            else:
                raise ValueError(f"unknown transport {self.transport!r}")
        finally:
            if pause_engines:
                for e in self.engines:
                    e.resume()
        stats = SyncStats(version, time.perf_counter() - t0, nbytes, len(tensors), nbuckets, self.transport)
        self.history.append(stats)
        return stats

    def _sync_nccl(self, tensors: Iterable[tuple[str, torch.Tensor]], version: int) -> None:
        """Trainer side of the NCCL broadcast; engines must implement ``recv_weight_nccl``."""
        import torch.distributed as dist

        group = getattr(self, "_nccl_group", None)
        for name, t in tensors:
            for e in self.engines:
                recv = getattr(e, "recv_weight_nccl", None)
                if recv is not None:
                    recv(name, t.dtype, t.shape)  # async rpc: engine workers post a matching broadcast
            dist.broadcast(t.contiguous().cuda(), src=0, group=group)
        for e in self.engines:
            e.policy_version = version

    def gbps(self) -> float:
        if not self.history:
            return 0.0
        s = self.history[-1]
        return (s.bytes * 8) / max(s.seconds, 1e-9) / 1e9


def lora_only(name: str) -> bool:
    return "lora_" in name
