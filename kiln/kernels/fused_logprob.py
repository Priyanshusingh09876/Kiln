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

import os

import torch

from kiln import native
from kiln.kernels.dispatch import next_pow2, num_warps_for, use_triton
from kiln.native import use_cuda_ext

# Largest Triton block for the one-program-per-row forward.
MAX_BLOCK_V = 8192
# Split-row slices are multiples of this, so every split-kernel BLOCK_V (<= it) divides them.
SPLIT_GRAIN = 2048
# Fallback path: rows per chunk are chosen so one fp32 [rows, V] temporary is about this big.
_REF_CHUNK_BYTES = 256 * 2**20

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
        """One program per row, streaming the whole vocabulary with an online log-sum-exp."""
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
            l = l * scale + tl.sum(e, axis=0)  # noqa: E741 - online-softmax denominator
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
    def _fwd_partial_kernel(
        logits_ptr,
        targets_ptr,
        m_ptr,
        l_ptr,
        s_ptr,
        t_ptr,
        V,
        stride_n,
        CHUNK,
        NSPLIT,
        BLOCK_V: tl.constexpr,
    ):
        """Split-row forward: program (row, k) reduces vocabulary slice ``[k*CHUNK, (k+1)*CHUNK)``.

        With few rows (short RL batches, decode-time scoring) one program per row
        leaves most SMs idle; splitting each row gives every SM work. The partial
        ``(max, sum exp, sum exp*x, target logit)`` tuples are merged by
        :func:`_combine_partials`.
        """
        row = tl.program_id(0)
        k = tl.program_id(1)
        base = logits_ptr + row * stride_n
        target = tl.load(targets_ptr + row)
        lo = k * CHUNK

        m = -float("inf")
        l = 0.0  # noqa: E741
        s = 0.0
        tgt_logit = 0.0
        for start in range(0, CHUNK, BLOCK_V):
            offs = lo + start + tl.arange(0, BLOCK_V)
            mask = offs < V
            x = tl.load(base + offs, mask=mask, other=-float("inf")).to(tl.float32)
            m_new = tl.maximum(m, tl.max(x, axis=0))
            # a slice that is entirely padding keeps m = -inf; avoid exp(-inf - -inf) = nan
            m_safe = tl.where(m_new == -float("inf"), 0.0, m_new)
            scale = tl.exp(m - m_safe)
            e = tl.where(mask, tl.exp(x - m_safe), 0.0)
            l = l * scale + tl.sum(e, axis=0)  # noqa: E741
            s = s * scale + tl.sum(e * tl.where(mask, x, 0.0), axis=0)
            m = m_new
            tgt_logit += tl.sum(tl.where(offs == target, x, 0.0), axis=0)

        out = row * NSPLIT + k
        tl.store(m_ptr + out, m)
        tl.store(l_ptr + out, l)
        tl.store(s_ptr + out, s)
        tl.store(t_ptr + out, tgt_logit)

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
        """v1 backward: one program per row, looping over the vocabulary (kept for the tuning sweep)."""
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
            tl.store(gbase + offs, g.to(grad_ptr.dtype.element_ty), mask=mask)

    @triton.jit
    def _bwd_tile_kernel(
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
        """Backward on a 2-D grid ``(row, vocab tile)``: no loop, so every tile is an independent program.

        The gradient is stored directly in the logits dtype; v1 wrote fp32 and
        converted afterwards, which for bf16 costs an extra ``[N, V]`` fp32 buffer
        and two more full passes over memory.
        """
        row = tl.program_id(0)
        tile = tl.program_id(1)
        offs = tile * BLOCK_V + tl.arange(0, BLOCK_V)
        mask = offs < V
        target = tl.load(targets_ptr + row)
        lse = tl.load(lse_ptr + row)
        ent = tl.load(ent_ptr + row)
        g_lp = tl.load(g_logp_ptr + row)
        g_h = tl.load(g_ent_ptr + row)
        x = tl.load(logits_ptr + row * stride_n + offs, mask=mask, other=-float("inf")).to(tl.float32)
        logp = x - lse
        p = tl.exp(logp)
        onehot = tl.where(offs == target, 1.0, 0.0)
        g = g_lp * (onehot - p) + g_h * (-p * (logp + ent))
        tl.store(grad_ptr + row * stride_gn + offs, g.to(grad_ptr.dtype.element_ty), mask=mask)

    _ROW_CONFIGS = [
        triton.Config({"BLOCK_V": b}, num_warps=w) for b in (1024, 2048, 4096, 8192) for w in (4, 8, 16)
    ]
    _TILE_CONFIGS = [triton.Config({"BLOCK_V": b}, num_warps=w) for b in (1024, 2048, 4096) for w in (4, 8)]
    _fwd_tuned = triton.autotune(configs=_ROW_CONFIGS, key=["V"])(_fwd_kernel)
    _SPLIT_CONFIGS = [triton.Config({"BLOCK_V": b}, num_warps=w) for b in (1024, 2048) for w in (4, 8)]
    _fwd_partial_tuned = triton.autotune(configs=_SPLIT_CONFIGS, key=["V", "CHUNK"])(_fwd_partial_kernel)
    _bwd_tile_tuned = triton.autotune(configs=_TILE_CONFIGS, key=["V"])(_bwd_tile_kernel)

    _HAS_TRITON = True
except Exception:  # pragma: no cover
    _HAS_TRITON = False


def _cdiv(a: int, b: int) -> int:
    return -(-a // b)


def _autotune_enabled() -> bool:
    return os.environ.get("KILN_TRITON_AUTOTUNE", "1") != "0"


def _combine_partials(m, l, s, t):  # noqa: E741
    """Merge split-row partials ``[N, S]`` into ``(logp, ent, lse)`` ``[N]`` (fp32)."""
    M = m.max(dim=1).values
    w = torch.exp(m - M[:, None])
    w = torch.where(torch.isfinite(m), w, torch.zeros_like(w))  # empty slices contribute nothing
    L = (l * w).sum(1)
    S = (s * w).sum(1)
    lse = M + torch.log(L)
    return t.sum(1) - lse, lse - S / L, lse


def split_factor(n_rows: int, V: int, device: torch.device | None = None) -> int:
    """How many programs to split each row across so the GPU has ~4 programs per SM."""
    env = os.environ.get("KILN_LOGPROB_SPLIT")
    if env:
        return max(1, int(env))
    if device is None or not torch.cuda.is_available():
        return 1
    sms = torch.cuda.get_device_properties(device).multi_processor_count
    want = _cdiv(4 * sms, max(n_rows, 1))
    return max(1, min(want, _cdiv(V, SPLIT_GRAIN)))


def _triton_forward(logits_c: torch.Tensor, targets: torch.Tensor):
    N, V = logits_c.shape
    dev = logits_c.device
    S = split_factor(N, V, dev)
    tuned = _autotune_enabled()
    if S == 1:
        logp = torch.empty(N, device=dev, dtype=torch.float32)
        ent = torch.empty_like(logp)
        lse = torch.empty_like(logp)
        if tuned:
            _fwd_tuned[(N,)](logits_c, targets, logp, ent, lse, V, logits_c.stride(0))
        else:
            B = min(next_pow2(V), MAX_BLOCK_V)
            _fwd_kernel[(N,)](
                logits_c,
                targets,
                logp,
                ent,
                lse,
                V,
                logits_c.stride(0),
                BLOCK_V=B,
                num_warps=num_warps_for(B),
            )
        return logp, ent, lse
    chunk = _cdiv(_cdiv(V, S), SPLIT_GRAIN) * SPLIT_GRAIN  # a multiple of every split BLOCK_V
    S = _cdiv(V, chunk)
    parts = [torch.empty(N, S, device=dev, dtype=torch.float32) for _ in range(4)]
    if tuned:
        _fwd_partial_tuned[(N, S)](logits_c, targets, *parts, V, logits_c.stride(0), chunk, S)
    else:
        _fwd_partial_kernel[(N, S)](
            logits_c, targets, *parts, V, logits_c.stride(0), chunk, S, BLOCK_V=SPLIT_GRAIN, num_warps=8
        )
    return _combine_partials(*parts)


def _triton_backward(logits, targets, lse, ent, g_logp, g_ent):
    N, V = logits.shape
    grad = torch.empty(N, V, device=logits.device, dtype=logits.dtype)
    if _autotune_enabled():
        grid = lambda meta: (N, triton.cdiv(V, meta["BLOCK_V"]))  # noqa: E731
        _bwd_tile_tuned[grid](
            logits, targets, lse, ent, g_logp, g_ent, grad, V, logits.stride(0), grad.stride(0)
        )
    else:
        B = 4096
        _bwd_tile_kernel[(N, triton.cdiv(V, B))](
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
            BLOCK_V=B,
            num_warps=8,
        )
    return grad


def _ref_rows(V: int) -> int:
    return max(1, _REF_CHUNK_BYTES // (4 * max(V, 1)))


def _logprob_entropy_ref(logits: torch.Tensor, targets: torch.Tensor):
    """Reference forward, chunked over rows so at most one fp32 ``[rows, V]`` temporary is live."""
    N, V = logits.shape
    logp = torch.empty(N, device=logits.device, dtype=torch.float32)
    ent = torch.empty_like(logp)
    lse = torch.empty_like(logp)
    step = _ref_rows(V)
    for i in range(0, N, step):
        x = logits[i : i + step].float()
        c_lse = torch.logsumexp(x, dim=-1)
        logp[i : i + step] = x.gather(-1, targets[i : i + step].unsqueeze(-1)).squeeze(-1) - c_lse
        p = torch.exp(x - c_lse.unsqueeze(-1))
        ent[i : i + step] = c_lse - (p * x).sum(-1)
        lse[i : i + step] = c_lse
    return logp, ent, lse


def _bwd_ref(logits, targets, lse, ent, g_logp, g_ent):
    """Reference backward, chunked like the forward; writes the gradient in the logits dtype."""
    N, V = logits.shape
    grad = torch.empty_like(logits)
    step = _ref_rows(V)
    for i in range(0, N, step):
        sl = slice(i, i + step)
        logp_all = logits[sl].float() - lse[sl].unsqueeze(-1)
        p = torch.exp(logp_all)
        g = (logp_all + ent[sl].unsqueeze(-1)).mul_(p).mul_(-g_ent[sl].unsqueeze(-1))
        g.sub_(p.mul_(g_logp[sl].unsqueeze(-1)))
        g.scatter_add_(-1, targets[sl].unsqueeze(-1), g_logp[sl].unsqueeze(-1))
        grad[sl] = g
    return grad


class _FusedLogprobEntropy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits: torch.Tensor, targets: torch.Tensor):
        assert logits.dim() == 2, "flatten to [N, V] first"
        ctx.backend = "torch"
        if use_cuda_ext(logits, targets):
            logits_c = logits if logits.stride(1) == 1 else logits.contiguous()
            logp, ent, lse = native.cuda().fused_logprob_entropy_fwd(logits_c, targets)
            ctx.save_for_backward(logits_c, targets, lse, ent)
            ctx.backend = "cuda"
            return logp, ent
        if use_triton(logits, targets) and _HAS_TRITON:
            logits_c = logits if logits.stride(1) == 1 else logits.contiguous()
            logp, ent, lse = _triton_forward(logits_c, targets)
            ctx.save_for_backward(logits_c, targets, lse, ent)
            ctx.backend = "triton"
            return logp, ent
        with torch.no_grad():
            logp, ent, lse = _logprob_entropy_ref(logits, targets)
        ctx.save_for_backward(logits, targets, lse, ent)
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
        if ctx.backend == "cuda":
            grad = native.cuda().fused_logprob_entropy_bwd(logits, targets, lse, ent, g_logp, g_ent)
            return grad.to(logits.dtype), None
        if ctx.backend == "triton":
            return _triton_backward(logits, targets, lse, ent, g_logp, g_ent), None
        return _bwd_ref(logits, targets, lse, ent, g_logp, g_ent), None


def fused_logprob_entropy(logits: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(logp, entropy)`` of shape ``[*]`` for ``logits [*, V]`` and ``targets [*]``.

    Works for any leading shape; internally flattens to ``[N, V]``. Output is
    always fp32 regardless of logits dtype, which is what the loss math wants.
    The input gradient has the logits dtype.
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
