"""FSDP2 (``fully_shard``) wrapping with mixed precision and activation checkpointing.

We wrap per transformer block (so each block's parameters are gathered just
in time and freed right after) and then wrap the root. ``reshard_after_forward``
is on by default for the policy (memory) and off for the reference model
(it only runs forward, no backward, and we would rather spend memory than
re-gather). On CPU / single process the model is returned unchanged.
"""

from __future__ import annotations

import functools
from collections.abc import Iterable

import torch
import torch.nn as nn

from kiln.config.schema import ParallelConfig
from kiln.distributed.parallel_state import ParallelState


def transformer_blocks(model: nn.Module) -> Iterable[nn.Module]:
    """Best-effort discovery of the per-layer modules to shard individually."""
    for name, mod in model.named_modules():
        if (
            isinstance(mod, nn.ModuleList)
            and len(mod) > 0
            and name.split(".")[-1] in {"blocks", "layers", "h"}
        ):
            yield from mod
            return


def apply_activation_checkpointing(model: nn.Module) -> None:
    try:
        from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
            CheckpointImpl,
            checkpoint_wrapper,
        )
        from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
            apply_activation_checkpointing as _apply,
        )
    except Exception:  # pragma: no cover
        return
    blocks = set(id(b) for b in transformer_blocks(model))
    if not blocks:
        return
    wrapper = functools.partial(checkpoint_wrapper, checkpoint_impl=CheckpointImpl.NO_REENTRANT)
    _apply(model, checkpoint_wrapper_fn=wrapper, check_fn=lambda m: id(m) in blocks)


def shard_model(
    model: nn.Module,
    pstate: ParallelState,
    pcfg: ParallelConfig,
    param_dtype: torch.dtype = torch.bfloat16,
    reshard_after_forward: bool | None = None,
    activation_checkpointing: bool = False,
) -> nn.Module:
    if not pstate.initialized or pstate.mesh is None or pcfg.dp == 1:
        if activation_checkpointing:
            apply_activation_checkpointing(model)
        return model
    from torch.distributed.fsdp import CPUOffloadPolicy, MixedPrecisionPolicy, fully_shard

    if not torch.cuda.is_available():
        # FSDP2 runs on CPU over gloo (used by the multi-rank tests); bf16 compute on CPU is slow and lossy
        param_dtype = torch.float32

    mp = MixedPrecisionPolicy(param_dtype=param_dtype, reduce_dtype=torch.float32, output_dtype=param_dtype)
    reshard = pcfg.fsdp_reshard_after_forward if reshard_after_forward is None else reshard_after_forward
    kwargs = dict(mesh=pstate.mesh["dp"], mp_policy=mp, reshard_after_forward=reshard)
    if pcfg.fsdp_cpu_offload:
        kwargs["offload_policy"] = CPUOffloadPolicy()
    if activation_checkpointing:
        apply_activation_checkpointing(model)
    for blk in transformer_blocks(model):
        fully_shard(blk, **kwargs)
    fully_shard(model, **kwargs)
    return model


def full_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Gather an unsharded fp32 state dict on every rank (for weight sync / export)."""
    try:
        from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict

        return get_model_state_dict(model, options=StateDictOptions(full_state_dict=True, cpu_offload=False))
    except Exception:
        return {k: v.detach() for k, v in model.state_dict().items()}


def sharded_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """This rank's shard only (what the checkpointer stores)."""
    try:
        from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict

        return get_model_state_dict(model, options=StateDictOptions(full_state_dict=False))
    except Exception:
        return model.state_dict()


def _is_dtensor(t: object) -> bool:
    return hasattr(t, "to_local") and hasattr(t, "device_mesh")


@torch.no_grad()
def load_sharded_state_dict(model: nn.Module, state: dict[str, torch.Tensor]) -> list[str]:
    """Load a state dict saved as *local shards* (what the checkpointer writes) into a model.

    For FSDP2 parameters (DTensors) each saved tensor is this rank's local shard and is
    copied into ``param.to_local()``; plain parameters are copied directly. Returns the
    list of model keys that were not found in ``state`` (should be empty).
    """
    own = model.state_dict()
    missing = []
    for k, dst in own.items():
        if k not in state:
            missing.append(k)
            continue
        src = state[k]
        local = dst.to_local() if _is_dtensor(dst) else dst
        if src.shape != local.shape:
            raise ValueError(
                f"shape mismatch for {k}: checkpoint {tuple(src.shape)} vs model shard {tuple(local.shape)}"
            )
        local.copy_(src.to(local.device, local.dtype))
    return missing


def rewrap_optimizer_state(optimizer: torch.optim.Optimizer) -> None:
    """After ``optimizer.load_state_dict`` from local shards, turn state tensors back into DTensors
    matching their parameter, so the next ``step`` does not mix DTensor params with plain state."""
    try:
        from torch.distributed.tensor import DTensor
    except ImportError:  # pragma: no cover
        return
    for group in optimizer.param_groups:
        for p in group["params"]:
            if not _is_dtensor(p) or p not in optimizer.state:
                continue
            st = optimizer.state[p]
            local_shape = p.to_local().shape
            for name, v in list(st.items()):
                if isinstance(v, torch.Tensor) and not _is_dtensor(v) and v.shape == local_shape:
                    st[name] = DTensor.from_local(
                        v.to(p.device), p.device_mesh, p.placements, run_check=False
                    )


def clip_grad_norm(model: nn.Module, max_norm: float) -> torch.Tensor:
    """Works for both FSDP2 DTensor params and plain tensors."""
    return torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
