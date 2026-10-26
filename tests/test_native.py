"""Native extensions: the C++ radix tree (differential vs. Python) and the CUDA kernels (parity, GPU only)."""

from __future__ import annotations

import random

import pytest
import torch

from kiln import native
from kiln.serving.router import BlockRadixTree as PyTree

cpp = native.cpp()
cuda = native.cuda()


# ------------------------------------------------------------------ C++ tree
@pytest.mark.skipif(cpp is None, reason="kiln._C not built (pip install --no-build-isolation -e .)")
def test_cpp_radix_tree_basic():
    t = cpp.BlockRadixTree(4, 3)
    a = list(range(12))
    assert t.insert(a) == 3 and t.match(a) == 12
    assert t.match(a[:8] + [99] * 4) == 8
    assert t.match([5, 6, 7, 8]) == 0
    t.insert([100, 101, 102, 103])
    assert t.size_blocks <= 3
    t.capacity = 1
    assert t.size_blocks == 1


@pytest.mark.skipif(cpp is None, reason="kiln._C not built")
def test_cpp_radix_tree_matches_python_reference():
    rng = random.Random(0)
    for trial in range(300):
        bs = rng.choice([1, 2, 4, 8])
        cap = rng.randint(1, 40)
        c, p = cpp.BlockRadixTree(bs, cap), PyTree(bs, cap)
        prompts = [[rng.randint(0, 5) for _ in range(rng.randint(0, 40))] for _ in range(30)]
        for q in prompts:
            if rng.random() < 0.5:
                q = rng.choice(prompts)
            assert c.insert(q) == p.insert(q), trial
            assert c.size_blocks == p.size_blocks <= cap, trial
            for r in prompts[:5]:
                assert c.match(r) == p.match(r), trial


@pytest.mark.skipif(cpp is None, reason="kiln._C not built")
def test_cpp_radix_tree_numpy_entry_points():
    np = pytest.importorskip("numpy")
    t = cpp.BlockRadixTree(4, 100)
    a = np.arange(16, dtype=np.int64)
    assert t.insert_array(a) == 4 and t.match_array(a) == 16
    assert t.match(list(range(16))) == 16
    assert t.insert_array(np.arange(8, dtype=np.int32)) == 0  # forcecast from int32


@pytest.mark.skipif(cpp is None, reason="kiln._C not built")
def test_router_uses_native_tree():
    from kiln.serving.router import EngineReplica

    r = EngineReplica("a", "http://a")
    assert type(r.tree).__module__ == "kiln._C"


# ------------------------------------------------------------ CUDA kernels
needs_cuda_ext = pytest.mark.skipif(cuda is None, reason="kiln._cuda not built or no GPU")


@needs_cuda_ext
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("vocab", [1000, 4097, 32768, 151936])
def test_cuda_fused_logprob_entropy_parity(dtype, vocab):
    N = 64
    logits = (torch.randn(N, vocab, device="cuda") * 3).to(dtype)
    tgt = torch.randint(0, vocab, (N,), device="cuda")
    logp, ent, lse = cuda.fused_logprob_entropy_fwd(logits, tgt)
    ref = torch.log_softmax(logits.float(), -1)
    lp_ref = ref.gather(-1, tgt[:, None]).squeeze(-1)
    ent_ref = -(ref.exp() * ref).sum(-1)
    tol = 1e-4 if dtype is torch.float32 else 2e-2
    assert torch.allclose(logp, lp_ref, atol=tol, rtol=tol)
    assert torch.allclose(ent, ent_ref, atol=tol, rtol=tol)
    g1, g2 = torch.randn_like(logp), torch.randn_like(ent)
    grad = cuda.fused_logprob_entropy_bwd(logits, tgt, lse, ent, g1, g2)
    x = logits.detach().float().requires_grad_(True)
    r = torch.log_softmax(x, -1)
    (r.gather(-1, tgt[:, None]).squeeze(-1) * g1 + (-(r.exp() * r).sum(-1)) * g2).sum().backward()
    assert torch.allclose(grad.float(), x.grad, atol=tol * 10, rtol=tol * 10)


@needs_cuda_ext
def test_cuda_fused_logprob_strided_rows():
    """Row stride != V (e.g. every other row of a larger buffer) must work without a copy."""
    V = 2048
    buf = torch.randn(16, V, device="cuda")
    view = buf[::2]  # stride(0) == 2*V, stride(1) == 1
    assert not view.is_contiguous()
    tgt = torch.randint(0, V, (8,), device="cuda")
    a = cuda.fused_logprob_entropy_fwd(view, tgt)[0]
    b = cuda.fused_logprob_entropy_fwd(view.contiguous(), tgt)[0]
    assert torch.allclose(a, b)


@needs_cuda_ext
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("hidden", [64, 1000, 4096, 8192])
def test_cuda_rmsnorm_parity(dtype, hidden):
    x = torch.randn(33, hidden, device="cuda", dtype=dtype)
    w = (1 + 0.1 * torch.randn(hidden, device="cuda")).to(dtype)
    y, rstd = cuda.rmsnorm_fwd(x, w, 1e-6)
    xf = x.float()
    y_ref = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + 1e-6) * w.float()
    tol = 1e-5 if dtype is torch.float32 else 2e-2
    assert torch.allclose(y.float(), y_ref, atol=tol, rtol=tol)
    dy = torch.randn_like(x)
    dx, dw = cuda.rmsnorm_bwd(x, w, dy, rstd)
    xr = xf.clone().requires_grad_(True)
    wr = w.float().clone().requires_grad_(True)
    (xr * torch.rsqrt(xr.pow(2).mean(-1, keepdim=True) + 1e-6) * wr * dy.float()).sum().backward()
    assert torch.allclose(dx.float(), xr.grad, atol=tol * 10, rtol=tol * 10)
    assert torch.allclose(dw.float(), wr.grad, atol=tol * 100, rtol=tol * 10)


@needs_cuda_ext
def test_autograd_dispatch_uses_cuda_backend(monkeypatch):
    from kiln.kernels import fused_logprob_entropy, rmsnorm

    monkeypatch.setenv("KILN_KERNEL_BACKEND", "cuda")
    assert native.backend() == "cuda"
    logits = torch.randn(8, 512, device="cuda", requires_grad=True)
    lp, ent = fused_logprob_entropy(logits, torch.randint(0, 512, (8,), device="cuda"))
    (lp.sum() + ent.sum()).backward()
    assert torch.isfinite(logits.grad).all()
    x = torch.randn(4, 256, device="cuda", requires_grad=True)
    rmsnorm(x, torch.ones(256, device="cuda")).sum().backward()
    assert torch.isfinite(x.grad).all()


def test_backend_env_override(monkeypatch):
    monkeypatch.setenv("KILN_KERNEL_BACKEND", "torch")
    assert native.backend() == "torch"
    s = native.summary()
    assert set(s) == {"cpp_ext", "cuda_ext", "backend"}


def test_backend_forcing_actually_disables_triton(monkeypatch):
    """KILN_KERNEL_BACKEND must pin the implementation, or kernel benchmarks compare a path with itself."""
    import kiln.kernels.dispatch as d

    monkeypatch.setattr(d, "_TRITON_IMPORTABLE", True)
    monkeypatch.setattr(d.torch.cuda, "is_available", lambda: True)
    for env, expect in (("auto", True), ("triton", True), ("torch", False), ("cuda", False)):
        monkeypatch.setenv("KILN_KERNEL_BACKEND", env)
        assert d.triton_enabled() is expect, env
    monkeypatch.setenv("KILN_KERNEL_BACKEND", "auto")
    monkeypatch.setenv("KILN_DISABLE_TRITON", "1")
    assert d.triton_enabled() is False
