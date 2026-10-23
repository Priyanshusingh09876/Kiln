# ADR-02: Fused Triton kernels for the RL hot path

**Status:** accepted · **Module:** `kiln.kernels`

Every kernel ships with a pure-PyTorch reference implementation with
identical semantics. The reference runs on CPU, under `KILN_DISABLE_TRITON=1`,
and in the parity tests; the Triton path dispatches automatically on CUDA.

## Fused log-prob + entropy (`fused_logprob.py`)

The per-token quantity an RL step needs from the LM is just two scalars per
position — `log p(target)` and `H(p)` — but the naive route materialises
`log_softmax(logits)` (`[N, V]` fp32, e.g. 8k × 128k × 4 B = 4 GB per micro-batch)
twice (forward and backward). On a 128k-vocab model that intermediate is
larger than the activations of the whole transformer stack.

The kernel is a single pass per row with an online (streaming) softmax:

```
for each block of V:
    m_new = max(m, max(x)); scale = exp(m − m_new)
    l = l·scale + Σ exp(x − m_new)                 # normaliser
    s = s·scale + Σ exp(x − m_new)·x               # Σ p·x numerator
    tgt += x[target]
lse  = m + log l
logp = tgt − lse
ent  = lse − s / l
```

Backward is also one pass and never stores probabilities:
`∂logp/∂x = onehot − p`, `∂ent/∂x = −p·(x − lse + ent)`. Memory traffic is
therefore ~2 reads + 1 write of the logits instead of ~6, and no `[N, V]`
fp32 intermediate ever exists. Outputs are fp32 regardless of input dtype.

Block size is `next_pow2(V)` capped at 64k with `num_warps` scaled to block
size (`dispatch.num_warps_for`); rows larger than one block loop.

## RMSNorm (`rmsnorm.py`)

Forward: one read of `x`, fp32 accumulation of the mean square, one write.
Backward recomputes `rstd` from `x` rather than saving it, and reduces the
weight gradient per-row into a `[rows, H]` partial that is summed once
(avoids atomics). Matches `torch` to 1e-5 in fp32.

## RoPE (`rope.py`)

Rotate-half layout (HF Llama convention), in place on `[B, T, H, D]`
without a transpose, gathering `cos/sin` by arbitrary `position_ids` so it
works with packed sequences. Backward is the rotation by `−θ`, which is the
same kernel with the sign flipped.

## FP8 block-scaled GEMM (`fp8_gemm.py`)

DeepSeek-V3-style: activations quantised to e4m3 in `1×128` blocks,
weights in `128×128` blocks, each with its own fp32 scale. The Triton kernel
uses `tl.dot` on fp8 operands with fp32 accumulation and applies the two
scales per K-block, so quantisation error does not compound across K.
Requires sm90+ (Hopper) for the FP8 tensor-core path; elsewhere the
reference dequantises and runs a bf16 matmul. `FP8Linear` is for frozen
weights (reference policy, inference); training-time FP8 needs delayed
scaling which lives in the trainer, not the kernel.

## Why not Liger / torch.compile?

Liger's fused linear-cross-entropy is the right idea for *SFT* (it fuses the
LM-head matmul too). For RL we need entropy and the per-token log-probs for
several policies, and `torch.compile` does not fuse the two-pass softmax
reduction across the vocab dimension. Writing the kernel was less work than
bending either.

## Testing

`tests/test_kernels.py` checks, per kernel, forward and backward against the
naive formula at fp32 and bf16 for several shapes including non-power-of-two
vocabularies, extreme logits (no NaN), and on CUDA additionally pins the
Triton path against the reference path by flipping `KILN_DISABLE_TRITON`.
