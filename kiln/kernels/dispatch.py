"""Backend dispatch for kernels.

Every kernel in :mod:`kiln.kernels` has a Triton implementation and a pure
PyTorch reference implementation with identical semantics. The reference path
is used (a) on CPU / non-CUDA devices, (b) when Triton is not installed, and
(c) when ``KILN_DISABLE_TRITON=1`` is set, which is how the numerical parity
tests pin the two implementations against each other.
"""

from __future__ import annotations

import functools
import os

import torch

try:  # pragma: no cover - import guard
    import triton  # noqa: F401
    import triton.language as tl  # noqa: F401

    _TRITON_IMPORTABLE = True
except Exception:  # pragma: no cover
    _TRITON_IMPORTABLE = False


def triton_enabled() -> bool:
    if os.environ.get("KILN_DISABLE_TRITON", "0") == "1":
        return False
    return _TRITON_IMPORTABLE and torch.cuda.is_available()


def use_triton(*tensors: torch.Tensor) -> bool:
    """True if every tensor is on CUDA and Triton is usable."""
    if not triton_enabled():
        return False
    return all(t.is_cuda for t in tensors if isinstance(t, torch.Tensor))


@functools.cache
def next_pow2(n: int) -> int:
    return 1 << (n - 1).bit_length()


def num_warps_for(block: int) -> int:
    if block >= 8192:
        return 16
    if block >= 2048:
        return 8
    return 4
