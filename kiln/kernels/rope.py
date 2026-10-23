"""Rotary position embeddings (GPT-NeoX / Llama "rotate_half" layout).

Applies RoPE in place of the usual ``q*cos + rotate_half(q)*sin`` expression,
which in eager PyTorch materialises three intermediates of shape ``[B, T, H, D]``.
The kernel reads q once and writes it once. The backward is the same kernel
with the rotation angle negated (rotation matrices are orthogonal), so no
extra state is saved beyond ``cos``/``sin`` which are tiny.

Layout: ``x [B, T, H, D]``, ``cos/sin [T, D/2]``. The same tables serve any
position mapping via ``positions [B, T]`` (needed for packed sequences).
"""

from __future__ import annotations

import torch

from kiln.kernels.dispatch import next_pow2, use_triton

try:  # pragma: no cover
    import triton
    import triton.language as tl

    @triton.jit
    def _rope_kernel(
        x_ptr,
        cos_ptr,
        sin_ptr,
        pos_ptr,
        out_ptr,
        T,
        H,
        HALF,
        stride_b,
        stride_t,
        stride_h,
        sign,
        BLOCK: tl.constexpr,
    ):
        pid = tl.program_id(0)  # over B*T*H
        h = pid % H
        bt = pid // H
        t = bt % T
        b = bt // T
        pos = tl.load(pos_ptr + b * T + t)
        offs = tl.arange(0, BLOCK)
        mask = offs < HALF
        base = b * stride_b + t * stride_t + h * stride_h
        x1 = tl.load(x_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
        x2 = tl.load(x_ptr + base + HALF + offs, mask=mask, other=0.0).to(tl.float32)
        c = tl.load(cos_ptr + pos * HALF + offs, mask=mask, other=0.0).to(tl.float32)
        s = tl.load(sin_ptr + pos * HALF + offs, mask=mask, other=0.0).to(tl.float32) * sign
        o1 = x1 * c - x2 * s
        o2 = x2 * c + x1 * s
        tl.store(out_ptr + base + offs, o1.to(out_ptr.dtype.element_ty), mask=mask)
        tl.store(out_ptr + base + HALF + offs, o2.to(out_ptr.dtype.element_ty), mask=mask)

    _HAS_TRITON = True
except Exception:  # pragma: no cover
    _HAS_TRITON = False


def rope_tables(max_seq: int, head_dim: int, base: float = 10000.0, device=None, dtype=torch.float32):
    """Return ``cos, sin`` of shape ``[max_seq, head_dim // 2]``."""
    half = head_dim // 2
    inv_freq = 1.0 / (base ** (torch.arange(0, half, device=device, dtype=torch.float32) / half))
    t = torch.arange(max_seq, device=device, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    return freqs.cos().to(dtype), freqs.sin().to(dtype)


def _rope_ref(x, cos, sin, positions, sign: float):
    half = x.shape[-1] // 2
    acc = torch.float64 if x.dtype == torch.float64 else torch.float32
    c = cos[positions].unsqueeze(2).to(acc)  # [B,T,1,half]
    s = sin[positions].unsqueeze(2).to(acc) * sign
    x1, x2 = x[..., :half].to(acc), x[..., half:].to(acc)
    o1 = x1 * c - x2 * s
    o2 = x2 * c + x1 * s
    return torch.cat([o1, o2], dim=-1).to(x.dtype)


def _rope_apply(x, cos, sin, positions, sign: float):
    if use_triton(x, cos, sin, positions) and _HAS_TRITON:
        x = x.contiguous()
        B, T, H, D = x.shape
        half = D // 2
        out = torch.empty_like(x)
        BLOCK = next_pow2(half)
        _rope_kernel[(B * T * H,)](
            x,
            cos.contiguous(),
            sin.contiguous(),
            positions.contiguous().to(torch.int32),
            out,
            T,
            H,
            half,
            x.stride(0),
            x.stride(1),
            x.stride(2),
            sign,
            BLOCK=BLOCK,
        )
        return out
    return _rope_ref(x, cos, sin, positions, sign)


class _RoPE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, cos, sin, positions):
        ctx.save_for_backward(cos, sin, positions)
        return _rope_apply(x, cos, sin, positions, 1.0)

    @staticmethod
    def backward(ctx, g):
        cos, sin, positions = ctx.saved_tensors
        return _rope_apply(g, cos, sin, positions, -1.0), None, None, None


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, positions: torch.Tensor | None = None):
    """``x [B, T, H, D]`` -> rotated ``x``. ``positions`` defaults to ``arange(T)``."""
    B, T, H, D = x.shape
    assert D % 2 == 0, "head_dim must be even"
    if positions is None:
        positions = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
    return _RoPE.apply(x, cos, sin, positions.long())
