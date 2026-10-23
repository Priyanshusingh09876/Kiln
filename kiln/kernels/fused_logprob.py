"""Fused per-token log-probability + entropy over the vocabulary dimension.

The RL loss needs, for every generated token, (i) ``log p(y_t | x)`` for the
PPO/GRPO ratio and (ii) the full-distribution entropy ``H_t`` for the entropy
bonus. The naive PyTorch path does

    log_softmax(logits)              # read V, write V
    gather(..., targets)             # read V
    (p * log_p).sum(-1)              # read V twice

i.e. 4-5 passes over an ``[N, V]`` fp32 tensor that, at V=150k and N=16k
tokens, is ~10 GB and lives entirely in HBM. This kernel does a single
streaming pass with an online log-sum-exp that simultaneously accumulates
``sum(exp(x - m) * x)`` so entropy falls out of the same loop, and the
backward recomputes softmax on the fly from the saved LSE instead of
materialising it.

Semantics (reference implementation in :func:`_logprob_entropy_ref`)::

    lse_i   = logsumexp(logits_i)
    logp_i  = logits_i[target_i] - lse_i
    H_i     = lse_i - sum_j softmax(logits_i)_j * logits_i_j

Gradients (see docs/design/fused-kernels.md for the derivation)::

    d logp_i / d logits_ij = 1[j == target_i] - p_ij
    d H_i    / d logits_ij = -p_ij * (log p_ij + H_i)
"""

from __future__ import annotations

import torch

from kiln.kernels.dispatch import next_pow2, num_warps_for, use_triton

try:  # pragma: no cover
    import triton
    import triton.language as tl

    @triton.jit
    def _fwd_kernel(
        logits_ptr,
        targets_ptr,
        logp_ptr,
        ent_ptr,
        lse_ptr,
        V,
        stride_n,
        BLOCK_V: tl.constexpr,
    ):
        row = tl.program_id(0)
        base = logits_ptr + row * stride_n
        target = tl.load(targets_ptr + row)

        m = -float("inf")
        l = 0.0  # noqa: E741 - running sum exp(x - m)
        s = 0.0  # running sum exp(x - m) * x
        tgt_logit = 0.0
        for start in range(0, V, BLOCK_V):
            offs = start + tl.arange(0, BLOCK_V)
            mask = offs < V
            x = tl.load(base + offs, mask=mask, other=-float("inf")).to(tl.float32)
            blk_max = tl.max(x, axis=0)
            m_new = tl.maximum(m, blk_max)
            scale = tl.exp(m - m_new)
            e = tl.exp(x - m_new)
            e = tl.where(mask, e, 0.0)
            l = l * scale  # noqa: E741 - online-softmax denominator, matches the paper + tl.sum(e, axis=0)
            s = s * scale + tl.sum(e * tl.where(mask, x, 0.0), axis=0)
            m = m_new
            tgt_logit += tl.sum(tl.where(offs == target, x, 0.0), axis=0)

        lse = m + tl.log(l)
        logp = tgt_logit - lse
        ent = lse - s / l
        tl.store(logp_ptr + row, logp)
        tl.store(ent_ptr + row, ent)
        tl.store(lse_ptr + row, lse)

    @triton.jit
    def _bwd_kernel(
        logits_ptr,
        targets_ptr,
        lse_ptr,
        ent_ptr,
        g_logp_ptr,
        g_ent_ptr,
        grad_ptr,
        V,
        stride_n,
        stride_gn,
        BLOCK_V: tl.constexpr,
    ):
        row = tl.program_id(0)
        base = logits_ptr + row * stride_n
        gbase = grad_ptr + row * stride_gn
        target = tl.load(targets_ptr + row)
        lse = tl.load(lse_ptr + row)
        ent = tl.load(ent_ptr + row)
        g_lp = tl.load(g_logp_ptr + row)
        g_h = tl.load(g_ent_ptr + row)

        for start in range(0, V, BLOCK_V):
            offs = start + tl.arange(0, BLOCK_V)
            mask = offs < V
            x = tl.load(base + offs, mask=mask, other=-float("inf")).to(tl.float32)
            logp = x - lse
            p = tl.exp(logp)
            onehot = tl.where(offs == target, 1.0, 0.0)
            g = g_lp * (onehot - p) + g_h * (-p * (logp + ent))
            tl.store(gbase + offs, g, mask=mask)

    _HAS_TRITON = True
except Exception:  # pragma: no cover
    _HAS_TRITON = False


def _logprob_entropy_ref(logits: torch.Tensor, targets: torch.Tensor):
    x = logits.float()
    lse = torch.logsumexp(x, dim=-1)
    logp = x.gather(-1, targets.unsqueeze(-1)).squeeze(-1) - lse
    p = torch.exp(x - lse.unsqueeze(-1))
    ent = lse - (p * x).sum(-1)
    return logp, ent, lse


def _bwd_ref(logits, targets, lse, ent, g_logp, g_ent):
    x = logits.float()
    logp_all = x - lse.unsqueeze(-1)
    p = torch.exp(logp_all)
    onehot = torch.zeros_like(p).scatter_(-1, targets.unsqueeze(-1), 1.0)
    grad = g_logp.unsqueeze(-1) * (onehot - p) + g_ent.unsqueeze(-1) * (-p * (logp_all + ent.unsqueeze(-1)))
    return grad


class _FusedLogprobEntropy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits: torch.Tensor, targets: torch.Tensor):
        assert logits.dim() == 2, "flatten to [N, V] first"
        N, V = logits.shape
        if use_triton(logits, targets) and _HAS_TRITON:
            logits_c = logits if logits.stride(1) == 1 else logits.contiguous()
            logp = torch.empty(N, device=logits.device, dtype=torch.float32)
            ent = torch.empty_like(logp)
            lse = torch.empty_like(logp)
            BLOCK_V = min(next_pow2(V), 8192)
            _fwd_kernel[(N,)](
                logits_c,
                targets,
                logp,
                ent,
                lse,
                V,
                logits_c.stride(0),
                BLOCK_V=BLOCK_V,
                num_warps=num_warps_for(BLOCK_V),
            )
            ctx.save_for_backward(logits_c, targets, lse, ent)
            ctx.triton = True
        else:
            logp, ent, lse = _logprob_entropy_ref(logits, targets)
            ctx.save_for_backward(logits, targets, lse, ent)
            ctx.triton = False
        return logp, ent

    @staticmethod
    def backward(ctx, g_logp: torch.Tensor, g_ent: torch.Tensor):
        logits, targets, lse, ent = ctx.saved_tensors
        if g_logp is None:
            g_logp = torch.zeros_like(lse)
        if g_ent is None:
            g_ent = torch.zeros_like(lse)
        g_logp = g_logp.contiguous().float()
        g_ent = g_ent.contiguous().float()
        if ctx.triton:
            N, V = logits.shape
            grad = torch.empty(N, V, device=logits.device, dtype=torch.float32)
            BLOCK_V = min(next_pow2(V), 8192)
            _bwd_kernel[(N,)](
                logits,
                targets,
                lse,
                ent,
                g_logp,
                g_ent,
                grad,
                V,
                logits.stride(0),
                grad.stride(0),
                BLOCK_V=BLOCK_V,
                num_warps=num_warps_for(BLOCK_V),
            )
        else:
            grad = _bwd_ref(logits, targets, lse, ent, g_logp, g_ent)
        return grad.to(logits.dtype), None


def fused_logprob_entropy(logits: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(logp, entropy)`` of shape ``[*]`` for ``logits [*, V]`` and ``targets [*]``.

    Works for any leading shape; internally flattens to ``[N, V]``. Output is
    always fp32 regardless of logits dtype, which is what the loss math wants.
    """
    lead = logits.shape[:-1]
    V = logits.shape[-1]
    logp, ent = _FusedLogprobEntropy.apply(logits.reshape(-1, V), targets.reshape(-1).long())
    return logp.view(lead), ent.view(lead)


def sequence_logprobs(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    response_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Causal-LM convenience: logits ``[B, T, V]`` predict ``input_ids[:, 1:]``.

    Returns per-token ``logp`` and ``entropy`` of shape ``[B, T-1]`` with
    positions outside ``response_mask[:, 1:]`` zeroed.
    """
    shifted_logits = logits[:, :-1, :]
    targets = input_ids[:, 1:]
    mask = response_mask[:, 1:].to(torch.float32)
    logp, ent = fused_logprob_entropy(shifted_logits, targets)
    return logp * mask, ent * mask
