"""Bucketed collectives and the weight-broadcast primitive used for weight sync.

Broadcasting a 32B-parameter model as ~500 individual tensors costs ~500 NCCL
launches plus a sync per launch; coalescing into ~1 GB buckets turns that
into a handful of bandwidth-bound transfers. :func:`bucketed_broadcast`
packs same-dtype tensors into flat buffers, broadcasts the buffers, and
unpacks on the receiving side. With ``torch.distributed`` uninitialised it
is a no-op that returns the tensors unchanged, so the same code path runs
in single-process tests.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass

import torch
import torch.distributed as dist


def _dist_ok() -> bool:
    return dist.is_available() and dist.is_initialized()


@dataclass
class Bucket:
    dtype: torch.dtype
    names: list[str]
    shapes: list[torch.Size]
    numels: list[int]

    @property
    def total(self) -> int:
        return sum(self.numels)


def plan_buckets(named: Iterable[tuple[str, torch.Tensor]], bucket_bytes: int = 1 << 30) -> list[Bucket]:
    """Group tensors by dtype into buckets of at most ``bucket_bytes`` (a single huge tensor gets its own)."""
    buckets: list[Bucket] = []
    current: dict[torch.dtype, Bucket] = {}
    for name, t in named:
        itemsize = t.element_size()
        b = current.get(t.dtype)
        if b is not None and (b.total + t.numel()) * itemsize > bucket_bytes:
            buckets.append(b)
            b = None
        if b is None:
            b = Bucket(t.dtype, [], [], [])
            current[t.dtype] = b
        b.names.append(name)
        b.shapes.append(t.shape)
        b.numels.append(t.numel())
    buckets.extend(current.values())
    return buckets


def flatten_bucket(bucket: Bucket, tensors: dict[str, torch.Tensor], device: torch.device) -> torch.Tensor:
    flat = torch.empty(bucket.total, dtype=bucket.dtype, device=device)
    off = 0
    for name, n in zip(bucket.names, bucket.numels):
        flat[off : off + n].copy_(tensors[name].reshape(-1), non_blocking=True)
        off += n
    return flat


def unflatten_bucket(bucket: Bucket, flat: torch.Tensor) -> Iterator[tuple[str, torch.Tensor]]:
    off = 0
    for name, shape, n in zip(bucket.names, bucket.shapes, bucket.numels):
        yield name, flat[off : off + n].view(shape)
        off += n


def bucketed_broadcast(
    tensors: dict[str, torch.Tensor],
    src: int,
    group=None,
    bucket_bytes: int = 1 << 30,
    device: torch.device | None = None,
) -> Iterator[tuple[str, torch.Tensor]]:
    """Yield ``(name, tensor)`` on every rank after broadcasting from ``src`` in coalesced buckets.

    On the source rank the yielded tensors alias the inputs' values; on
    receivers they are freshly-populated views into the flat buffer, valid
    until the next bucket is processed (consumers copy into their own storage).
    """
    if not _dist_ok():
        yield from tensors.items()
        return
    device = device or next(iter(tensors.values())).device
    for bucket in plan_buckets(tensors.items(), bucket_bytes):
        flat = flatten_bucket(bucket, tensors, device)
        dist.broadcast(flat, src=src, group=group)
        yield from unflatten_bucket(bucket, flat)


def all_gather_object(obj, group=None) -> list:
    if not _dist_ok():
        return [obj]
    out = [None] * dist.get_world_size(group)
    dist.all_gather_object(out, obj, group=group)
    return out


def all_reduce_metrics(metrics: dict[str, torch.Tensor], group=None) -> dict[str, float]:
    """Average scalar metrics across ranks; returns python floats."""
    if not metrics:
        return {}
    keys = sorted(metrics)
    stacked = torch.stack([metrics[k].detach().float().reshape(()) for k in keys])
    if _dist_ok():
        dist.all_reduce(stacked, op=dist.ReduceOp.SUM, group=group)
        stacked /= dist.get_world_size(group)
    return {k: float(v) for k, v in zip(keys, stacked.tolist())}
