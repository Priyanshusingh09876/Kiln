"""Block-scaled FP8 (e4m3) GEMM with fp32 accumulation.

Scaling scheme (DeepSeek-V3 style, which keeps outliers from blowing the
tiny e4m3 range):

* activations ``A [M, K]`` are quantised per ``(1, BLOCK_K)`` group -> scale ``sa [M, K/BLOCK_K]``
* weights     ``B [N, K]`` are quantised per ``(BLOCK_N, BLOCK_K)`` tile -> scale ``sb [N/BLOCK_N, K/BLOCK_K]``

The kernel accumulates each K-block in fp32, multiplies the partial product by
``sa[m, kb] * sb[nb, kb]`` and adds it into the running accumulator, so the
dequantisation is folded into the main loop and never touches HBM. On
hardware without FP8 tensor cores (or without CUDA) we fall back to
dequantise-then-matmul which is bit-for-bit the same math.

``quantize_blockwise`` is the single source of truth for the scale layout so
that the Triton path and the reference path cannot drift apart.
"""

from __future__ import annotations

import torch

from kiln.kernels.dispatch import use_triton

FP8_DTYPE = getattr(torch, "float8_e4m3fn", None)
FP8_MAX = 448.0

try:  # pragma: no cover
    import triton
    import triton.language as tl

    @triton.jit
    def _fp8_gemm_kernel(
        a_ptr,
        b_ptr,
        c_ptr,
        sa_ptr,
        sb_ptr,
        M,
        N,
        K,
        stride_am,
        stride_ak,
        stride_bn,
        stride_bk,
        stride_cm,
        stride_cn,
        stride_sam,
        stride_sak,
        stride_sbn,
        stride_sbk,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for kb in range(0, tl.cdiv(K, BLOCK_K)):
            k = kb * BLOCK_K + offs_k
            a_mask = (offs_m[:, None] < M) & (k[None, :] < K)
            b_mask = (offs_n[:, None] < N) & (k[None, :] < K)
            a = tl.load(a_ptr + offs_m[:, None] * stride_am + k[None, :] * stride_ak, mask=a_mask, other=0.0)
            b = tl.load(b_ptr + offs_n[:, None] * stride_bn + k[None, :] * stride_bk, mask=b_mask, other=0.0)
            part = tl.dot(a, tl.trans(b), out_dtype=tl.float32)
            sa = tl.load(sa_ptr + offs_m * stride_sam + kb * stride_sak, mask=offs_m < M, other=0.0)
            sb = tl.load(sb_ptr + pid_n * stride_sbn + kb * stride_sbk)
            acc += part * sa[:, None] * sb
        c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        tl.store(
            c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
            acc.to(c_ptr.dtype.element_ty),
            mask=c_mask,
        )

    _HAS_TRITON = True
except Exception:  # pragma: no cover
    _HAS_TRITON = False


def _to_fp8(x: torch.Tensor) -> torch.Tensor:
    if FP8_DTYPE is None:  # very old torch: emulate with clamped bf16
        return x.clamp(-FP8_MAX, FP8_MAX).to(torch.bfloat16)
    return x.clamp(-FP8_MAX, FP8_MAX).to(FP8_DTYPE)


def quantize_activations(a: torch.Tensor, block_k: int = 128) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-``(1, block_k)`` quantisation. Returns ``(a_fp8 [M,K], scale [M, K/block_k])`` (fp32 scales)."""
    M, K = a.shape
    assert K % block_k == 0, "K must be a multiple of block_k"
    a32 = a.float().view(M, K // block_k, block_k)
    amax = a32.abs().amax(-1, keepdim=True).clamp_min(1e-12)
    scale = amax / FP8_MAX
    q = _to_fp8(a32 / scale)
    return q.view(M, K), scale.view(M, K // block_k)


def quantize_weights(
    b: torch.Tensor, block_n: int = 128, block_k: int = 128
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-``(block_n, block_k)`` tile quantisation of ``b [N, K]``."""
    N, K = b.shape
    assert N % block_n == 0 and K % block_k == 0
    b32 = b.float().view(N // block_n, block_n, K // block_k, block_k).permute(0, 2, 1, 3)
    amax = b32.abs().amax(dim=(-1, -2), keepdim=True).clamp_min(1e-12)
    scale = amax / FP8_MAX
    q = _to_fp8(b32 / scale).permute(0, 2, 1, 3).reshape(N, K)
    return q, scale.view(N // block_n, K // block_k)


def dequantize_activations(q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    M, K = q.shape
    G = scale.shape[1]
    return (q.float().view(M, G, K // G) * scale.view(M, G, 1)).view(M, K)


def dequantize_weights(q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    N, K = q.shape
    NB, KB = scale.shape
    bn, bk = N // NB, K // KB
    x = q.float().view(NB, bn, KB, bk).permute(0, 2, 1, 3)
    x = x * scale.view(NB, KB, 1, 1)
    return x.permute(0, 2, 1, 3).reshape(N, K)


def fp8_gemm(
    a_q: torch.Tensor,
    a_scale: torch.Tensor,
    b_q: torch.Tensor,
    b_scale: torch.Tensor,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """``C = A @ B.T`` for block-quantised ``A [M,K]`` and ``B [N,K]``."""
    M, K = a_q.shape
    N = b_q.shape[0]
    block_k = K // a_scale.shape[1]
    block_n = N // b_scale.shape[0]
    fp8_ok = FP8_DTYPE is not None and a_q.dtype == FP8_DTYPE
    if _HAS_TRITON and fp8_ok and use_triton(a_q, b_q) and torch.cuda.get_device_capability()[0] >= 9:
        c = torch.empty(M, N, device=a_q.device, dtype=out_dtype)
        BLOCK_M = 64
        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, block_n))
        _fp8_gemm_kernel[grid](
            a_q,
            b_q,
            c,
            a_scale,
            b_scale,
            M,
            N,
            K,
            a_q.stride(0),
            a_q.stride(1),
            b_q.stride(0),
            b_q.stride(1),
            c.stride(0),
            c.stride(1),
            a_scale.stride(0),
            a_scale.stride(1),
            b_scale.stride(0),
            b_scale.stride(1),
            BLOCK_M=BLOCK_M,
            BLOCK_N=block_n,
            BLOCK_K=block_k,
            num_warps=8,
            num_stages=3,
        )
        return c
    a = dequantize_activations(a_q, a_scale)
    b = dequantize_weights(b_q, b_scale)
    return (a @ b.t()).to(out_dtype)


class FP8Linear(torch.nn.Module):
    """Drop-in ``nn.Linear`` whose forward runs the block-scaled FP8 GEMM.

    Weights are quantised once at ``freeze()`` time (inference / frozen
    reference-model use). Training a layer in FP8 needs the delayed-scaling
    recipe which lives in the trainer, not here.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = False, block: int = 128):
        super().__init__()
        self.in_features, self.out_features, self.block = in_features, out_features, block
        self.weight = torch.nn.Parameter(torch.empty(out_features, in_features))
        self.bias = torch.nn.Parameter(torch.zeros(out_features)) if bias else None
        torch.nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        self.register_buffer("w_q", None, persistent=False)
        self.register_buffer("w_scale", None, persistent=False)

    @classmethod
    def from_linear(cls, lin: torch.nn.Linear, block: int = 128) -> FP8Linear:
        m = cls(lin.in_features, lin.out_features, bias=lin.bias is not None, block=block)
        m.weight.data.copy_(lin.weight.data)
        if lin.bias is not None:
            m.bias.data.copy_(lin.bias.data)
        m.freeze()
        return m

    @torch.no_grad()
    def freeze(self) -> None:
        self.w_q, self.w_scale = quantize_weights(self.weight.data, self.block, self.block)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.w_q is None:
            return torch.nn.functional.linear(x, self.weight, self.bias)
        lead = x.shape[:-1]
        a_q, a_s = quantize_activations(x.reshape(-1, self.in_features), self.block)
        y = fp8_gemm(
            a_q, a_s, self.w_q, self.w_scale, out_dtype=x.dtype if x.dtype != torch.float32 else torch.float32
        )
        if self.bias is not None:
            y = y + self.bias
        return y.view(*lead, self.out_features)
