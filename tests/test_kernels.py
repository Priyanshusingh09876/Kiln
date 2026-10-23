"""Numerical parity tests for every kernel: reference path vs. naive PyTorch, and Triton vs. reference on CUDA.

Forward *and* backward are checked, because a kernel whose gradient is wrong
trains a model that looks fine for a few hundred steps and then drifts.
"""

from __future__ import annotations

import os

import pytest
import torch

from kiln.kernels import (
    FP8Linear,
    RMSNorm,
    apply_rope,
    dequantize_activations,
    dequantize_weights,
    fp8_gemm,
    fused_logprob_entropy,
    quantize_activations,
    quantize_weights,
    rmsnorm,
    rope_tables,
    sequence_logprobs,
    triton_enabled,
)

CUDA = torch.cuda.is_available()
DEVICES = ["cpu"] + (["cuda"] if CUDA else [])


def _naive_logprob_entropy(logits, targets):
    lp = torch.log_softmax(logits.float(), -1)
    tok = lp.gather(-1, targets[..., None]).squeeze(-1)
    ent = -(lp.exp() * lp).sum(-1)
    return tok, ent


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("vocab", [100, 1024, 4097])
def test_fused_logprob_entropy_matches_naive(device, dtype, vocab):
    if device == "cpu" and dtype is torch.bfloat16:
        pytest.skip("bf16 log_softmax on CPU reference is too lossy for a tight tolerance")
    N = 37
    logits = (torch.randn(N, vocab, device=device) * 3).to(dtype).requires_grad_(True)
    targets = torch.randint(0, vocab, (N,), device=device)
    logits_ref = logits.detach().clone().requires_grad_(True)

    lp, ent = fused_logprob_entropy(logits, targets)
    lp_ref, ent_ref = _naive_logprob_entropy(logits_ref, targets)
    tol = 1e-5 if dtype is torch.float32 else 2e-2
    assert lp.dtype == torch.float32 and ent.dtype == torch.float32
    assert torch.allclose(lp, lp_ref, atol=tol, rtol=tol)
    assert torch.allclose(ent, ent_ref, atol=tol, rtol=tol)

    g1, g2 = torch.randn_like(lp), torch.randn_like(ent)
    (lp * g1 + ent * g2).sum().backward()
    (lp_ref * g1 + ent_ref * g2).sum().backward()
    assert torch.allclose(logits.grad.float(), logits_ref.grad.float(), atol=tol * 10, rtol=tol * 10)


def test_sequence_logprobs_alignment_and_masking():
    B, T, V = 2, 5, 11
    logits = torch.randn(B, T, V)
    ids = torch.randint(0, V, (B, T))
    resp = torch.tensor([[0, 0, 1, 1, 1], [0, 1, 1, 0, 0]])
    lp, ent = sequence_logprobs(logits, ids, resp)
    assert lp.shape == (B, T - 1)
    # token at absolute index t is predicted by logits[t-1]
    ref = torch.log_softmax(logits[:, :-1].float(), -1).gather(-1, ids[:, 1:, None]).squeeze(-1)
    m = resp[:, 1:].float()
    assert torch.allclose(lp, ref * m, atol=1e-6)
    assert (lp[m == 0] == 0).all() and (ent[m == 0] == 0).all()


def test_logprob_handles_extreme_logits_without_nan():
    logits = torch.tensor([[1e4, -1e4, 0.0]], requires_grad=True)
    lp, ent = fused_logprob_entropy(logits, torch.tensor([0]))
    assert torch.isfinite(lp).all() and torch.isfinite(ent).all()
    (lp + ent).sum().backward()
    assert torch.isfinite(logits.grad).all()


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("hidden", [64, 1000, 4096])
def test_rmsnorm_matches_naive(device, hidden):
    x = torch.randn(9, hidden, device=device, requires_grad=True)
    w = (1 + 0.1 * torch.randn(hidden, device=device)).requires_grad_(True)
    xr, wr = x.detach().clone().requires_grad_(True), w.detach().clone().requires_grad_(True)
    y = rmsnorm(x, w, eps=1e-6)
    yr = xr * torch.rsqrt(xr.pow(2).mean(-1, keepdim=True) + 1e-6) * wr
    assert torch.allclose(y, yr, atol=1e-5)
    g = torch.randn_like(y)
    (y * g).sum().backward()
    (yr * g).sum().backward()
    assert torch.allclose(x.grad, xr.grad, atol=1e-5)
    assert torch.allclose(w.grad, wr.grad, atol=1e-4)


def test_rmsnorm_module_roundtrip():
    m = RMSNorm(16)
    y = m(torch.randn(2, 3, 16))
    assert y.shape == (2, 3, 16)
    assert torch.allclose(y.pow(2).mean(-1), torch.ones(2, 3), atol=1e-4)


@pytest.mark.parametrize("device", DEVICES)
def test_rope_matches_naive_and_is_orthogonal(device):
    B, T, H, D = 2, 10, 3, 16
    cos, sin = rope_tables(32, D, device=device)
    x = torch.randn(B, T, H, D, device=device, requires_grad=True)
    pos = torch.stack([torch.arange(T, device=device), torch.arange(T, device=device) + 5])
    y = apply_rope(x, cos, sin, pos)
    # naive rotate-half
    x1, x2 = x[..., : D // 2], x[..., D // 2 :]
    c, s = cos[pos][:, :, None, :], sin[pos][:, :, None, :]
    yr = torch.cat([x1 * c - x2 * s, x1 * s + x2 * c], -1)
    assert torch.allclose(y, yr, atol=1e-5)
    # rotation preserves norm
    assert torch.allclose(y.norm(dim=-1), x.norm(dim=-1), atol=1e-5)
    g = torch.randn_like(y)
    (y * g).sum().backward()
    xr = x.detach().clone().requires_grad_(True)
    x1, x2 = xr[..., : D // 2], xr[..., D // 2 :]
    (torch.cat([x1 * c - x2 * s, x1 * s + x2 * c], -1) * g).sum().backward()
    assert torch.allclose(x.grad, xr.grad, atol=1e-5)


def test_rope_relative_position_property():
    """q_m . k_n depends only on m - n."""
    D = 32
    cos, sin = rope_tables(64, D)
    q = torch.randn(1, 1, 1, D)
    k = torch.randn(1, 1, 1, D)

    def dot(m, n):
        qm = apply_rope(q, cos, sin, torch.tensor([[m]]))
        kn = apply_rope(k, cos, sin, torch.tensor([[n]]))
        return (qm * kn).sum().item()

    assert dot(5, 2) == pytest.approx(dot(13, 10), abs=1e-4)
    assert dot(5, 2) != pytest.approx(dot(5, 3), abs=1e-3)


def test_fp8_block_quant_roundtrip_error_is_bounded():
    if not hasattr(torch, "float8_e4m3fn"):
        pytest.skip("no fp8 dtype in this torch")
    a = torch.randn(64, 256) * 4
    b = torch.randn(128, 256)
    aq, asc = quantize_activations(a, block_k=128)
    bq, bsc = quantize_weights(b, block_n=128, block_k=128)
    a_hat = dequantize_activations(aq, asc)
    b_hat = dequantize_weights(bq, bsc)
    # e4m3 has 3 mantissa bits -> relative error ~ 2^-4 per element; measure in aggregate
    assert (a_hat - a).norm() / a.norm() < 0.08
    assert (b_hat - b).norm() / b.norm() < 0.08
    c = fp8_gemm(aq, asc, bq, bsc, out_dtype=torch.float32)
    ref = a @ b.t()
    assert (c - ref).norm() / ref.norm() < 0.1


def test_fp8_linear_matches_bf16_linear_approximately():
    if not hasattr(torch, "float8_e4m3fn"):
        pytest.skip("no fp8 dtype in this torch")
    lin = torch.nn.Linear(256, 128, bias=False)
    f8 = FP8Linear.from_linear(lin)
    x = torch.randn(8, 256)
    y_ref = lin(x)
    y = f8(x).float()
    assert (y - y_ref).norm() / y_ref.norm() < 0.1


@pytest.mark.skipif(not CUDA or not triton_enabled(), reason="Triton + CUDA required")
def test_triton_and_reference_paths_agree():
    """Run the same op with Triton on and off (KILN_DISABLE_TRITON) and compare bitwise-ish."""
    import importlib

    import kiln.kernels.dispatch as d

    N, V = 64, 8192
    logits = torch.randn(N, V, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    tgt = torch.randint(0, V, (N,), device="cuda")
    lp_t, ent_t = fused_logprob_entropy(logits, tgt)
    (lp_t.sum() + ent_t.sum()).backward()
    g_t = logits.grad.clone()
    logits.grad = None
    os.environ["KILN_DISABLE_TRITON"] = "1"
    try:
        importlib.reload(d)
        lp_r, ent_r = fused_logprob_entropy(logits, tgt)
        (lp_r.sum() + ent_r.sum()).backward()
    finally:
        os.environ.pop("KILN_DISABLE_TRITON")
        importlib.reload(d)
    assert torch.allclose(lp_t, lp_r, atol=1e-3, rtol=1e-3)
    assert torch.allclose(ent_t, ent_r, atol=1e-3, rtol=1e-3)
    assert torch.allclose(g_t.float(), logits.grad.float(), atol=1e-2, rtol=1e-2)
