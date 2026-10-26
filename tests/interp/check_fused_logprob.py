"""Run the Triton fused log-prob kernels under TRITON_INTERPRET=1 (CPU) and
compare with an fp64 autograd reference. Exits non-zero on mismatch."""

import sys

import torch
import triton

from kiln.kernels import fused_logprob as F

torch.manual_seed(0)
worst = 0.0
for V in (100, 1024, 4097, 20000):  # 20000 > BLOCK_V cap: exercises the multi-block online softmax
    N = 4
    x = torch.randn(N, V) * 3
    t = torch.randint(0, V, (N,))
    lp, ent, lse = torch.empty(N), torch.empty(N), torch.empty(N)
    B = min(triton.next_power_of_2(V), 8192)
    F._fwd_kernel[(N,)](x, t, lp, ent, lse, V, x.stride(0), BLOCK_V=B)
    glp, gh, g = torch.randn(N), torch.randn(N), torch.empty_like(x)
    F._bwd_kernel[(N,)](x, t, lse, ent, glp, gh, g, V, x.stride(0), g.stride(0), BLOCK_V=B)

    xx = x.double().requires_grad_()
    ls = torch.log_softmax(xx, -1)
    r_lp = ls.gather(-1, t[:, None]).squeeze(-1)
    r_ent = -(ls.exp() * ls).sum(-1)
    (r_lp * glp.double() + r_ent * gh.double()).sum().backward()
    errs = [(lp - r_lp).abs().max(), (ent - r_ent).abs().max(), (g - xx.grad).abs().max()]
    worst = max(worst, *(float(e) for e in errs))
    print(V, [f"{float(e):.2e}" for e in errs])
sys.exit(0 if worst < 1e-4 else 1)
