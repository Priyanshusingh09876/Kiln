"""Triton kernels with PyTorch reference fallbacks.

Public API:

* :func:`fused_logprob_entropy` / :func:`sequence_logprobs` — one-pass token
  log-prob + entropy (the hot path of every RL step).
* :func:`rmsnorm` / :class:`RMSNorm`
* :func:`apply_rope` / :func:`rope_tables`
* :func:`fp8_gemm` + block quantisers / :class:`FP8Linear`
"""

from kiln.kernels.dispatch import triton_enabled, use_triton
from kiln.kernels.fp8_gemm import (
    FP8Linear,
    dequantize_activations,
    dequantize_weights,
    fp8_gemm,
    quantize_activations,
    quantize_weights,
)
from kiln.kernels.fused_logprob import fused_logprob_entropy, sequence_logprobs
from kiln.kernels.rmsnorm import RMSNorm, rmsnorm
from kiln.kernels.rope import apply_rope, rope_tables

__all__ = [
    "FP8Linear",
    "RMSNorm",
    "apply_rope",
    "dequantize_activations",
    "dequantize_weights",
    "fp8_gemm",
    "fused_logprob_entropy",
    "quantize_activations",
    "quantize_weights",
    "rmsnorm",
    "rope_tables",
    "sequence_logprobs",
    "triton_enabled",
    "use_triton",
]
