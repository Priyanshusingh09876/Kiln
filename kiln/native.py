"""Loader for the optional native extensions (``kiln._C`` C++, ``kiln._cuda`` CUDA).

Both are optional: everything in Kiln has a Python / Triton / PyTorch path.
Importing ``torch`` first is required so the extensions can resolve
``libc10``/``libtorch`` symbols.

    KILN_KERNEL_BACKEND = auto | cuda | triton | torch   (default: auto)

``auto`` prefers the CUDA extension when built, then Triton, then PyTorch.
"""

from __future__ import annotations

import importlib
import os
from types import ModuleType

import torch  # noqa: F401  - must be imported before the extensions

_cpp: ModuleType | None | bool = False
_cuda: ModuleType | None | bool = False


def _try(name: str) -> ModuleType | None:
    try:
        return importlib.import_module(name)
    except ImportError:
        return None


def cpp() -> ModuleType | None:
    """``kiln._C`` if built, else ``None``."""
    global _cpp
    if _cpp is False:
        _cpp = _try("kiln._C")
    return _cpp  # type: ignore[return-value]


def cuda() -> ModuleType | None:
    """``kiln._cuda`` if built and a CUDA device is present, else ``None``."""
    global _cuda
    if _cuda is False:
        _cuda = _try("kiln._cuda") if torch.cuda.is_available() else None
    return _cuda  # type: ignore[return-value]


def backend() -> str:
    """Resolved kernel backend name for the current process."""
    want = os.environ.get("KILN_KERNEL_BACKEND", "auto").lower()
    if want == "torch":
        return "torch"
    if want == "cuda":
        return "cuda" if cuda() is not None else "torch"
    from kiln.kernels.dispatch import triton_enabled

    if want == "triton":
        return "triton" if triton_enabled() else "torch"
    if cuda() is not None:
        return "cuda"
    if triton_enabled():
        return "triton"
    return "torch"


def use_cuda_ext(*tensors: torch.Tensor) -> bool:
    return backend() == "cuda" and all(t.is_cuda for t in tensors if isinstance(t, torch.Tensor))


def summary() -> dict[str, str | bool]:
    return {
        "cpp_ext": cpp() is not None,
        "cuda_ext": cuda() is not None,
        "backend": backend(),
    }
