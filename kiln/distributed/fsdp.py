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
    if not pstate.initialized or pstate.mesh is None or pcfg.dp == 1 or not torch.cuda.is_available():
        if activation_checkpointing:
            apply_activation_checkpointing(model)
        return model
    from torch.distributed.fsdp import CPUOffloadPolicy, MixedPrecisionPolicy, fully_shard

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


def clip_grad_norm(model: nn.Module, max_norm: float) -> torch.Tensor:
    """Works for both FSDP2 DTensor params and plain tensors."""
    return torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
