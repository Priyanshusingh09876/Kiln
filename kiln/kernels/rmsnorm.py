"""RMSNorm with a single-pass Triton forward and a fused backward.

    y = x * rsqrt(mean(x^2) + eps) * w

Backward (per row, r = rsqrt(mean(x^2)+eps), xhat = x * r):
    dw  += sum_rows dy * xhat
    dx   = r * (dy * w - xhat * mean(dy * w * xhat))

The weight gradient is accumulated in fp32 via a per-row partial buffer that
is reduced in PyTorch, which avoids atomics and keeps the kernel deterministic.
"""

from __future__ import annotations

import torch

from kiln.kernels.dispatch import next_pow2, num_warps_for, use_triton

try:  # pragma: no cover
    import triton
    import triton.language as tl

    @triton.jit
    def _rmsnorm_fwd(x_ptr, w_ptr, y_ptr, rstd_ptr, N, stride, eps, BLOCK: tl.constexpr):
        row = tl.program_id(0)
        offs = tl.arange(0, BLOCK)
        mask = offs < N
        x = tl.load(x_ptr + row * stride + offs, mask=mask, other=0.0).to(tl.float32)
        var = tl.sum(x * x, axis=0) / N
        rstd = 1.0 / tl.sqrt(var + eps)
        w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        y = x * rstd * w
        tl.store(rstd_ptr + row, rstd)
        tl.store(y_ptr + row * stride + offs, y.to(y_ptr.dtype.element_ty), mask=mask)

    @triton.jit
    def _rmsnorm_bwd(x_ptr, w_ptr, dy_ptr, rstd_ptr, dx_ptr, dw_partial_ptr, N, stride, BLOCK: tl.constexpr):
        row = tl.program_id(0)
        offs = tl.arange(0, BLOCK)
        mask = offs < N
        x = tl.load(x_ptr + row * stride + offs, mask=mask, other=0.0).to(tl.float32)
        dy = tl.load(dy_ptr + row * stride + offs, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        rstd = tl.load(rstd_ptr + row)
        xhat = x * rstd
        wdy = w * dy
        c = tl.sum(wdy * xhat, axis=0) / N
        dx = rstd * (wdy - xhat * c)
        tl.store(dx_ptr + row * stride + offs, dx.to(dx_ptr.dtype.element_ty), mask=mask)
        tl.store(dw_partial_ptr + row * N + offs, dy * xhat, mask=mask)

    _HAS_TRITON = True
except Exception:  # pragma: no cover
    _HAS_TRITON = False


def _acc_dtype(x: torch.Tensor) -> torch.dtype:
    return torch.float64 if x.dtype == torch.float64 else torch.float32


def _fwd_ref(x: torch.Tensor, w: torch.Tensor, eps: float):
    xf = x.to(_acc_dtype(x))
    rstd = torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    y = (xf * rstd * w.to(xf.dtype)).to(x.dtype)
    return y, rstd.squeeze(-1)


class _RMSNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, w: torch.Tensor, eps: float):
        shape = x.shape
        N = shape[-1]
        x2 = x.reshape(-1, N)
        if use_triton(x2, w) and _HAS_TRITON:
            x2 = x2.contiguous()
            y = torch.empty_like(x2)
            rstd = torch.empty(x2.shape[0], device=x.device, dtype=torch.float32)
            BLOCK = next_pow2(N)
            _rmsnorm_fwd[(x2.shape[0],)](
                x2, w, y, rstd, N, x2.stride(0), eps, BLOCK=BLOCK, num_warps=num_warps_for(BLOCK)
            )
            ctx.triton = True
        else:
            y, rstd = _fwd_ref(x2, w, eps)
            ctx.triton = False
        ctx.save_for_backward(x2, w, rstd)
        return y.view(shape)

    @staticmethod
    def backward(ctx, dy: torch.Tensor):
        x2, w, rstd = ctx.saved_tensors
        N = x2.shape[-1]
        dy2 = dy.reshape(-1, N)
        if ctx.triton:
            dy2 = dy2.contiguous()
            dx = torch.empty_like(x2)
            dw_partial = torch.empty(x2.shape[0], N, device=x2.device, dtype=torch.float32)
            BLOCK = next_pow2(N)
            _rmsnorm_bwd[(x2.shape[0],)](
                x2, w, dy2, rstd, dx, dw_partial, N, x2.stride(0), BLOCK=BLOCK, num_warps=num_warps_for(BLOCK)
            )
            dw = dw_partial.sum(0)
        else:
            xf = x2.to(_acc_dtype(x2))
            xhat = xf * rstd.unsqueeze(-1)
            wdy = w.to(xf.dtype) * dy2.to(xf.dtype)
            c = (wdy * xhat).mean(-1, keepdim=True)
            dx = (rstd.unsqueeze(-1) * (wdy - xhat * c)).to(x2.dtype)
            dw = (dy2.to(xf.dtype) * xhat).sum(0)
        return dx.view(dy.shape), dw.to(w.dtype), None


def rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return _RMSNorm.apply(x, weight, eps)


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return rmsnorm(x, self.weight, self.eps)
