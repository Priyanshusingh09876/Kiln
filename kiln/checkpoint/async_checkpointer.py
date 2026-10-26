"""Asynchronous, atomic, rank-sharded checkpointing.

Why this exists
---------------
A synchronous ``torch.save`` of a sharded 70B model + Adam state stalls every
GPU for the full serialisation + write time (tens of seconds per checkpoint
at interval 50 easily becomes >10% of wall-clock). The cost splits into two
very different pieces:

1. **Device -> host copy.** Unavoidable, but fast (NVLink/PCIe bandwidth,
   ~1-2 s for a shard) and the only part the GPU has to wait for.
2. **Serialisation + filesystem write.** Slow (seconds to minutes on shared
   NFS/Lustre) and does not need the GPU at all.

This checkpointer does (1) on the training stream into *pinned* host buffers
that are reused across checkpoints, then hands the host tensors to a
background thread that does (2). Training resumes as soon as the copy is
issued. If the previous write is still in flight when the next checkpoint
is requested we block (back-pressure) rather than pile up host memory.

Atomicity
---------
Every checkpoint is written to ``<dir>/step_<N>.tmp/`` and renamed to
``<dir>/step_<N>/`` only after every rank's shard and the metadata file are
complete, and a ``LATEST`` pointer file is updated last. A crash mid-write
therefore never leaves a checkpoint that *looks* complete. Multi-rank
coordination uses a barrier when ``torch.distributed`` is initialised; the
rank-0 process writes metadata.

Format
------
One ``rank<i>.pt`` per rank containing that rank's (sharded or full) state
dict, plus ``meta.json``. ``safetensors`` is used for the model weights when
available (zero-copy loading, no pickle); optimizer state goes through
``torch.save`` because it contains non-tensor structure.
"""

from __future__ import annotations

import json
import os
import shutil
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from kiln.config.schema import CheckpointConfig

try:
    import torch.distributed as dist
except Exception:  # pragma: no cover
    dist = None  # type: ignore


def _dist_ok() -> bool:
    return dist is not None and dist.is_available() and dist.is_initialized()


def _rank() -> int:
    return dist.get_rank() if _dist_ok() else 0


def _world() -> int:
    return dist.get_world_size() if _dist_ok() else 1


_CKPT_GROUP = None


def _ckpt_group():
    """A dedicated gloo process group for checkpoint coordination.

    The writer thread must not issue collectives on the training process group:
    two threads driving collectives on one group can interleave and deadlock.
    Created lazily on the *main* thread (``AsyncCheckpointer.__init__``) because
    ``new_group`` is itself a collective every rank must enter in order.
    """
    global _CKPT_GROUP
    if _CKPT_GROUP is None and _dist_ok():
        _CKPT_GROUP = dist.new_group(backend="gloo")
    return _CKPT_GROUP


def _barrier() -> None:
    if _dist_ok():
        dist.barrier(group=_ckpt_group())


@dataclass
class CheckpointStats:
    step: int
    copy_seconds: float
    write_seconds: float = 0.0
    bytes_written: int = 0
    path: str = ""


@dataclass
class _HostBuffers:
    """Pinned host staging buffers keyed by tensor name, reused across saves."""

    buffers: dict[str, torch.Tensor] = field(default_factory=dict)

    def get(self, name: str, like: torch.Tensor) -> torch.Tensor:
        buf = self.buffers.get(name)
        if buf is None or buf.shape != like.shape or buf.dtype != like.dtype:
            pin = like.is_cuda
            buf = torch.empty(like.shape, dtype=like.dtype, device="cpu", pin_memory=pin)
            self.buffers[name] = buf
        return buf


class AsyncCheckpointer:
    def __init__(self, cfg: CheckpointConfig, run_dir: str | Path | None = None):
        self.cfg = cfg
        self.root = Path(run_dir or cfg.dir)
        self.root.mkdir(parents=True, exist_ok=True)
        _ckpt_group()  # create on the main thread, in lockstep on every rank
        self._host = _HostBuffers()
        self._thread: threading.Thread | None = None
        self._error: BaseException | None = None
        self.history: list[CheckpointStats] = []
        self._lock = threading.Lock()

    # ------------------------------------------------------------ public
    def save(
        self, step: int, model_state: Mapping[str, Any], extra_state: Mapping[str, Any] | None = None
    ) -> CheckpointStats:
        """Snapshot ``model_state`` (+ optimizer/scheduler/RNG in ``extra_state``) at ``step``."""
        self.wait()  # back-pressure: one in-flight write at a time
        t0 = time.perf_counter()
        host_model = self._stage(model_state, "model")
        host_extra = self._stage_extra(extra_state or {})
        if torch.cuda.is_available():
            torch.cuda.synchronize()  # the non_blocking copies must land before the writer reads them
        copy_s = time.perf_counter() - t0
        stats = CheckpointStats(step=step, copy_seconds=copy_s)

        def _writer() -> None:
            try:
                self._write(step, host_model, host_extra, stats)
            except BaseException as e:  # noqa: BLE001 - surfaced on next wait()
                self._error = e

        if self.cfg.async_save:
            self._thread = threading.Thread(target=_writer, name=f"kiln-ckpt-{step}", daemon=True)
            self._thread.start()
        else:
            _writer()
            self._raise_if_failed()
        with self._lock:
            self.history.append(stats)
        return stats

    def wait(self) -> None:
        if self._thread is not None:
            self._thread.join()
            self._thread = None
        self._raise_if_failed()

    def latest_step(self) -> int | None:
        p = self.root / "LATEST"
        if not p.exists():
            return None
        return int(p.read_text().strip())

    def load(
        self, step: int | None = None, map_location: str | torch.device = "cpu"
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        """Return ``(model_state, extra_state, meta)`` for ``step`` (default: latest)."""
        step = self.latest_step() if step is None else step
        if step is None:
            raise FileNotFoundError(f"no checkpoint under {self.root}")
        d = self.root / f"step_{step}"
        meta = json.loads((d / "meta.json").read_text())
        shard = d / f"rank{_rank()}.pt"
        payload = torch.load(shard, map_location=map_location, weights_only=False)
        model_state = payload["model"]
        st_path = d / f"rank{_rank()}.safetensors"
        if st_path.exists():
            from safetensors.torch import load_file

            model_state = load_file(str(st_path), device=str(map_location))
        return model_state, payload.get("extra", {}), meta

    def overhead_fraction(self, step_seconds: float) -> float:
        """Blocking time per step attributable to checkpointing, as a fraction of step time."""
        if not self.history:
            return 0.0
        blocking = sum(s.copy_seconds for s in self.history)
        steps = max(self.history[-1].step, 1)
        return blocking / (steps * step_seconds)

    # ----------------------------------------------------------- internals
    def _stage(self, state: Mapping[str, Any], prefix: str) -> dict[str, torch.Tensor]:
        out = {}
        for k, v in state.items():
            if not isinstance(v, torch.Tensor):
                out[k] = v
                continue
            if hasattr(v, "to_local"):  # DTensor -> local shard
                v = v.to_local()
            buf = self._host.get(f"{prefix}.{k}", v)
            buf.copy_(v.detach(), non_blocking=True)
            out[k] = buf
        return out

    def _stage_extra(self, extra: Mapping[str, Any]) -> dict[str, Any]:
        """Deep-copy ``extra`` preserving its structure; tensors go to pinned host buffers."""

        def rec(obj: Any, path: str) -> Any:
            if isinstance(obj, torch.Tensor):
                if hasattr(obj, "to_local"):
                    obj = obj.to_local()
                buf = self._host.get(path, obj)
                buf.copy_(obj.detach(), non_blocking=True)
                return buf
            if isinstance(obj, Mapping):
                return {k: rec(v, f"{path}.{k}") for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                out = [rec(v, f"{path}.{i}") for i, v in enumerate(obj)]
                return out if isinstance(obj, list) else tuple(out)
            return obj

        return rec(dict(extra), "extra")

    def _write(
        self, step: int, host_model: dict[str, Any], host_extra: dict[str, Any], stats: CheckpointStats
    ) -> None:
        t0 = time.perf_counter()
        final = self.root / f"step_{step}"
        tmp = self.root / f"step_{step}.tmp"
        if _rank() == 0:
            tmp.mkdir(parents=True, exist_ok=True)
        _barrier()
        shard = tmp / f"rank{_rank()}.pt"
        tensors = {k: v.clone() for k, v in host_model.items() if isinstance(v, torch.Tensor)}
        non_tensors = {k: v for k, v in host_model.items() if not isinstance(v, torch.Tensor)}
        wrote_st = False
        try:
            from safetensors.torch import save_file

            save_file(
                {k: v.contiguous() for k, v in tensors.items()}, str(tmp / f"rank{_rank()}.safetensors")
            )
            wrote_st = True
        except Exception:
            pass
        payload = {
            "model": non_tensors if wrote_st else {**tensors, **non_tensors},
            "extra": _clone_tensors(host_extra),
        }
        torch.save(payload, shard)
        nbytes = sum(p.stat().st_size for p in tmp.glob(f"rank{_rank()}.*"))
        _barrier()
        if _rank() == 0:
            meta = {
                "step": step,
                "world_size": _world(),
                "time": time.time(),
                "safetensors": wrote_st,
                "torch": torch.__version__,
            }
            (tmp / "meta.json").write_text(json.dumps(meta, indent=2))
            os.replace(tmp, final)  # atomic on POSIX
            (self.root / "LATEST.tmp").write_text(str(step))
            os.replace(self.root / "LATEST.tmp", self.root / "LATEST")
            self._prune()
        _barrier()
        stats.write_seconds = time.perf_counter() - t0
        stats.bytes_written = nbytes
        stats.path = str(final)

    def _prune(self) -> None:
        steps = sorted(
            int(p.name.split("_")[1])
            for p in self.root.glob("step_*")
            if p.is_dir() and not p.name.endswith(".tmp")
        )
        for s in steps[: -self.cfg.keep_last] if self.cfg.keep_last > 0 else []:
            shutil.rmtree(self.root / f"step_{s}", ignore_errors=True)

    def _raise_if_failed(self) -> None:
        if self._error is not None:
            err, self._error = self._error, None
            raise RuntimeError("async checkpoint write failed") from err


def _clone_tensors(obj: Any) -> Any:
    """Clone tensors out of the reusable host buffers so the next ``save`` cannot overwrite an in-flight write."""
    if isinstance(obj, torch.Tensor):
        return obj.clone()
    if isinstance(obj, dict):
        return {k: _clone_tensors(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        out = [_clone_tensors(v) for v in obj]
        return out if isinstance(obj, list) else tuple(out)
    return obj
