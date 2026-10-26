"""Run the Triton fused log-prob kernels under TRITON_INTERPRET=1 (CPU) and
compare with an fp64 autograd reference. Exits non-zero on mismatch.

Covers every kernel variant: the one-program-per-row forward, the split-row
forward + combine (including a slice that lies entirely past the vocabulary),
the v1 row-loop backward and the tiled backward, with fp32 and bf16 gradients."""

import os
import sys

import torch
import triton

from kiln.kernels import fused_logprob as F

torch.manual_seed(0)
worst = 0.0


def reference(x, t, glp, gh):
    xx = x.double().requires_grad_()
    ls = torch.log_softmax(xx, -1)
    r_lp = ls.gather(-1, t[:, None]).squeeze(-1)
    r_ent = -(ls.exp() * ls).sum(-1)
    (r_lp * glp.double() + r_ent * gh.double()).sum().backward()
    return r_lp.detach(), r_ent.detach(), xx.grad


def report(name, errs, tol):
    global worst
    errs = [float(e) for e in errs]
    bad = max(errs) > tol
    worst = max(worst, max(errs) / tol)
    print(f"{name:<44} {' '.join(f'{e:.2e}' for e in errs)}{'  FAIL' if bad else ''}")


for V in (100, 1024, 4097, 20000):  # 20000 > MAX_BLOCK_V: exercises the multi-block online softmax
    N = 4
    x = torch.randn(N, V) * 3
    t = torch.randint(0, V, (N,))
    glp, gh = torch.randn(N), torch.randn(N)
    r_lp, r_ent, r_grad = reference(x, t, glp, gh)
    B = min(triton.next_power_of_2(V), 8192)

    # v1 forward + v1 backward
    lp, ent, lse = torch.empty(N), torch.empty(N), torch.empty(N)
    F._fwd_kernel[(N,)](x, t, lp, ent, lse, V, x.stride(0), BLOCK_V=B)
    g = torch.empty_like(x)
    F._bwd_kernel[(N,)](x, t, lse, ent, glp, gh, g, V, x.stride(0), g.stride(0), BLOCK_V=B)
    report(
        f"V={V} row fwd / row bwd",
        [(lp - r_lp).abs().max(), (ent - r_ent).abs().max(), (g - r_grad).abs().max()],
        1e-4,
    )

    # split-row forward; S=4 with chunk 8192 leaves an all-padding slice when V=20000
    for S, chunk in ((2, F.MAX_BLOCK_V), (4, F.MAX_BLOCK_V), (3, 2048)):
        if (S - 1) * chunk >= V + chunk:
            continue
        parts = [torch.empty(N, S) for _ in range(4)]
        F._fwd_partial_kernel[(N, S)](x, t, *parts, V, x.stride(0), chunk, S, BLOCK_V=1024)
        if S * chunk < V:  # not covering the row: skip, the wrapper never does this
            continue
        plp, pent, plse = F._combine_partials(*parts)
        report(
            f"V={V} split fwd S={S} chunk={chunk}",
            [(plp - r_lp).abs().max(), (pent - r_ent).abs().max(), (plse - lse).abs().max()],
            1e-4,
        )

    # tiled backward, fp32 and bf16 gradient
    for dt, tol in ((torch.float32, 1e-4), (torch.bfloat16, 3e-2)):
        g = torch.empty(N, V, dtype=dt)
        Bt = 1024
        F._bwd_tile_kernel[(N, triton.cdiv(V, Bt))](
            x, t, lse, ent, glp, gh, g, V, x.stride(0), g.stride(0), BLOCK_V=Bt
        )
        report(
            f"V={V} tile bwd grad {str(dt).replace('torch.', '')}", [(g.double() - r_grad).abs().max()], tol
        )

# the Python wrappers used in training (fixed configs: the autotuner needs a GPU to time configs)
os.environ["KILN_TRITON_AUTOTUNE"] = "0"
for V, split in ((5000, "1"), (20000, "1"), (20000, "3"), (151936, "8")):
    os.environ["KILN_LOGPROB_SPLIT"] = split
    N = 3
    x = (torch.randn(N, V) * 3).to(torch.bfloat16)
    t = torch.randint(0, V, (N,))
    glp, gh = torch.randn(N), torch.randn(N)
    r_lp, r_ent, r_grad = reference(x.float(), t, glp, gh)
    lp, ent, lse = F._triton_forward(x, t)
    g = F._triton_backward(x, t, lse, ent, glp, gh)
    assert g.dtype == torch.bfloat16
    report(
        f"V={V} wrapper split={split} (bf16)",
        [(lp - r_lp).abs().max(), (ent - r_ent).abs().max(), (g.double() - r_grad).abs().max()],
        3e-2,
    )

sys.exit(0 if worst <= 1.0 else 1)
