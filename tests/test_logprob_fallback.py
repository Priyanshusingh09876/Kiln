"""The PyTorch fallback for fused log-prob + entropy is chunked over rows.

Added after the RTX 3080 Ti benchmark showed the unchunked fallback peaking at
16 GB (the whole card) at 4096 x 128k: it kept several fp32 [N, V] tensors alive.
"""

import pytest
import torch

import kiln.kernels.fused_logprob as F


@pytest.mark.parametrize("dtype,tol", [(torch.float32, 1e-5), (torch.bfloat16, 2e-2)])
@pytest.mark.parametrize("rows_per_chunk", [1, 3, 64])
def test_chunked_fallback_matches_float64(monkeypatch, dtype, tol, rows_per_chunk):
    V = 1000
    monkeypatch.setattr(F, "_REF_CHUNK_BYTES", 4 * V * rows_per_chunk)
    monkeypatch.setenv("KILN_KERNEL_BACKEND", "torch")
    g = torch.Generator().manual_seed(0)
    x = (torch.randn(10, V, generator=g) * 3).to(dtype).requires_grad_(True)
    t = torch.randint(0, V, (10,), generator=g)
    g1, g2 = torch.randn(10, generator=g), torch.randn(10, generator=g)
    lp, ent = F.fused_logprob_entropy(x, t)
    torch.autograd.backward([lp, ent], [g1, g2])

    xx = x.detach().double().requires_grad_(True)
    ls = torch.log_softmax(xx, -1)
    r_lp = ls.gather(-1, t[:, None]).squeeze(-1)
    r_ent = -(ls.exp() * ls).sum(-1)
    torch.autograd.backward([r_lp, r_ent], [g1.double(), g2.double()])

    assert x.grad.dtype == dtype  # gradient comes back in the input dtype, no fp32 copy
    assert (lp.double() - r_lp).abs().max() < 1e-5
    assert (ent.double() - r_ent).abs().max() < 1e-5
    assert (x.grad.double() - xx.grad).abs().max() < tol


def test_chunk_size_bounds_the_temporary():
    assert F._ref_rows(128256) * 128256 * 4 <= F._REF_CHUNK_BYTES
    assert F._ref_rows(10**9) == 1
