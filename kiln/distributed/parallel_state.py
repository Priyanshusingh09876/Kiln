"""Process-group bookkeeping for (dp, tp, pp, cp) parallelism.

A single ``DeviceMesh`` of shape ``[pp, dp, cp, tp]`` is built once; every
other module asks this file for groups rather than calling
``torch.distributed`` directly. The ordering puts TP innermost so TP groups
land on the NVLink-connected GPUs of one node, then CP, then DP across nodes,
then PP outermost (PP traffic is point-to-point and smallest).

Everything degrades to a trivially-correct single-process view when
``torch.distributed`` is not initialised so unit tests and the mock rollout
loop run on a laptop.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import timedelta

import torch
import torch.distributed as dist

from kiln.config.schema import ParallelConfig


@dataclass
class ParallelState:
    cfg: ParallelConfig
    mesh: object | None = None  # torch.distributed.device_mesh.DeviceMesh
    rank: int = 0
    world_size: int = 1
    local_rank: int = 0

    @property
    def initialized(self) -> bool:
        return dist.is_available() and dist.is_initialized()

    def _dim(self, name: str) -> int:
        return getattr(self.cfg, name)

    def group(self, name: str):
        if self.mesh is None or self._dim(name) == 1:
            return None
        return self.mesh[name].get_group()

    def coord(self, name: str) -> int:
        if self.mesh is None:
            return 0
        return int(self.mesh[name].get_local_rank())

    @property
    def dp_group(self):
        return self.group("dp")

    @property
    def tp_group(self):
        return self.group("tp")

    @property
    def pp_group(self):
        return self.group("pp")

    @property
    def cp_group(self):
        return self.group("cp")

    @property
    def dp_rank(self) -> int:
        return self.coord("dp")

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    def device(self) -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda", self.local_rank)
        return torch.device("cpu")

    def barrier(self) -> None:
        if self.initialized:
            dist.barrier()

    def all_reduce_mean(self, t: torch.Tensor) -> torch.Tensor:
        if not self.initialized or self.dp_group is None:
            return t
        t = t.clone()
        dist.all_reduce(t, op=dist.ReduceOp.SUM, group=self.dp_group)
        return t / self.cfg.dp


_STATE: ParallelState | None = None


def init_parallel(cfg: ParallelConfig, backend: str | None = None, timeout_s: int = 1800) -> ParallelState:
    """Initialise torch.distributed (if launched via torchrun/Ray) and build the device mesh."""
    global _STATE
    if _STATE is not None:
        return _STATE
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world > 1 and not dist.is_initialized():
        backend = backend or ("nccl" if torch.cuda.is_available() else "gloo")
        dist.init_process_group(backend=backend, timeout=timedelta(seconds=timeout_s))
    state = ParallelState(cfg=cfg)
    if dist.is_available() and dist.is_initialized():
        state.rank = dist.get_rank()
        state.world_size = dist.get_world_size()
        state.local_rank = int(os.environ.get("LOCAL_RANK", state.rank % max(torch.cuda.device_count(), 1)))
        if torch.cuda.is_available():
            torch.cuda.set_device(state.local_rank)
        if cfg.world_size != state.world_size:
            raise ValueError(
                f"parallel config implies world size {cfg.world_size} but launched with {state.world_size}"
            )
        from torch.distributed.device_mesh import init_device_mesh

        dev_type = "cuda" if torch.cuda.is_available() else "cpu"
        state.mesh = init_device_mesh(
            dev_type, (cfg.pp, cfg.dp, cfg.cp, cfg.tp), mesh_dim_names=("pp", "dp", "cp", "tp")
        )
    _STATE = state
    return state


def get_parallel_state() -> ParallelState:
    if _STATE is None:
        return init_parallel(ParallelConfig())
    return _STATE


def destroy_parallel() -> None:
    global _STATE
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()
    _STATE = None
