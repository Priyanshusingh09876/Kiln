# Fused log-prob tuning sweep — `rtx3080-laptop`

Generated 2026-10-26 by `python scripts/tune_fused_logprob.py --tag rtx3080-laptop` on **NVIDIA GeForce RTX 3080 Ti Laptop GPU** (58 SMs, torch 2.14.1+cu126).
Kernel times are medians from `triton.testing.do_bench`. Measured copy bandwidth: **419 GB/s**; % is relative to it.
GB/s counts the minimum bytes each kernel must move (forward: read logits; backward: read logits, write gradient).

## Summary: v1 vs best configuration vs what training runs

| shape | v1 fwd + bwd | best fwd + best bwd | shipped (autograd, autotuned) | shipped % of copy BW | speedup over v1 |
|---|---|---|---|---|---|
| 4096×128256 bfloat16 | 18.04 ms | 7.45 ms | 8.76 ms (split=1) | 86% | 2.06× |
| 2048×151936 bfloat16 | 14.52 ms | 5.90 ms | 6.21 ms (split=1) | 72% | 2.34× |
| 256×128256 bfloat16 | 1.21 ms | 0.51 ms | 0.56 ms (split=1) | 84% | 2.16× |
| 64×151936 bfloat16 | 0.37 ms | 0.17 ms | 0.33 ms (split=4) | 42% | 1.13× |

*v1* is the forward with one program per row (`BLOCK_V=min(next_pow2(V), 8192)`, 16 warps) plus a backward
that writes fp32 and converts. *shipped* includes autograd overhead and the split-row merge.

## 4096×128256 bfloat16

**Forward** (fastest 8, and v1)

| kind | config | time | GB/s | % copy BW |
|---|---|---|---|---|
| row | BLOCK_V=8192 warps=8 | 2.444 ms | 430 | 103% |
| row | BLOCK_V=2048 warps=4 | 2.473 ms | 425 | 101% |
| split | S=2 BLOCK_V=2048 warps=4 | 2.612 ms | 402 | 96% |
| split | S=4 BLOCK_V=1024 warps=4 | 2.983 ms | 352 | 84% |
| row | BLOCK_V=8192 warps=4 | 2.984 ms | 352 | 84% |
| row | BLOCK_V=4096 warps=4 | 3.028 ms | 347 | 83% |
| row (v1) | BLOCK_V=8192 warps=16 | 3.062 ms | 343 | 82% |
| split | S=16 BLOCK_V=1024 warps=4 | 3.111 ms | 338 | 81% |

**Backward** (fastest 8, and v1)

| kind | config | time | GB/s | % copy BW |
|---|---|---|---|---|
| row, direct dtype | BLOCK_V=4096 warps=16 | 5.005 ms | 420 | 100% |
| row, direct dtype | BLOCK_V=8192 warps=8 | 5.011 ms | 419 | 100% |
| row, direct dtype | BLOCK_V=2048 warps=8 | 5.012 ms | 419 | 100% |
| row, direct dtype | BLOCK_V=4096 warps=8 | 5.014 ms | 419 | 100% |
| row, direct dtype | BLOCK_V=4096 warps=4 | 5.016 ms | 419 | 100% |
| row, direct dtype | BLOCK_V=2048 warps=4 | 5.023 ms | 418 | 100% |
| row, direct dtype | BLOCK_V=8192 warps=4 | 5.027 ms | 418 | 100% |
| tile, direct dtype | BLOCK_V=2048 warps=16 | 5.033 ms | 418 | 100% |
| v1 row + fp32 grad + convert | BLOCK_V=8192 warps=16 | 14.983 ms | 421 | 100% |

## 2048×151936 bfloat16

**Forward** (fastest 8, and v1)

| kind | config | time | GB/s | % copy BW |
|---|---|---|---|---|
| row | BLOCK_V=4096 warps=8 | 1.712 ms | 363 | 87% |
| row | BLOCK_V=4096 warps=4 | 1.718 ms | 362 | 86% |
| row | BLOCK_V=2048 warps=4 | 1.896 ms | 328 | 78% |
| row | BLOCK_V=8192 warps=8 | 2.019 ms | 308 | 74% |
| row (v1) | BLOCK_V=8192 warps=16 | 2.029 ms | 307 | 73% |
| row | BLOCK_V=8192 warps=4 | 2.046 ms | 304 | 73% |
| row | BLOCK_V=4096 warps=16 | 2.088 ms | 298 | 71% |
| split | S=4 BLOCK_V=1024 warps=4 | 2.182 ms | 285 | 68% |

**Backward** (fastest 8, and v1)

| kind | config | time | GB/s | % copy BW |
|---|---|---|---|---|
| row, direct dtype | BLOCK_V=8192 warps=16 | 4.187 ms | 297 | 71% |
| row, direct dtype | BLOCK_V=2048 warps=8 | 4.190 ms | 297 | 71% |
| row, direct dtype | BLOCK_V=8192 warps=8 | 4.191 ms | 297 | 71% |
| row, direct dtype | BLOCK_V=2048 warps=4 | 4.192 ms | 297 | 71% |
| row, direct dtype | BLOCK_V=4096 warps=16 | 4.192 ms | 297 | 71% |
| row, direct dtype | BLOCK_V=8192 warps=4 | 4.192 ms | 297 | 71% |
| row, direct dtype | BLOCK_V=1024 warps=4 | 4.193 ms | 297 | 71% |
| row, direct dtype | BLOCK_V=4096 warps=4 | 4.193 ms | 297 | 71% |
| v1 row + fp32 grad + convert | BLOCK_V=8192 warps=16 | 12.496 ms | 299 | 71% |

## 256×128256 bfloat16

**Forward** (fastest 8, and v1)

| kind | config | time | GB/s | % copy BW |
|---|---|---|---|---|
| row | BLOCK_V=2048 warps=4 | 0.193 ms | 341 | 81% |
| row | BLOCK_V=2048 warps=8 | 0.207 ms | 317 | 76% |
| row | BLOCK_V=4096 warps=4 | 0.228 ms | 288 | 69% |
| row | BLOCK_V=4096 warps=8 | 0.229 ms | 286 | 68% |
| row | BLOCK_V=1024 warps=4 | 0.233 ms | 281 | 67% |
| row | BLOCK_V=8192 warps=8 | 0.251 ms | 262 | 62% |
| row | BLOCK_V=4096 warps=16 | 0.252 ms | 261 | 62% |
| row (v1) | BLOCK_V=8192 warps=16 | 0.257 ms | 255 | 61% |

**Backward** (fastest 8, and v1)

| kind | config | time | GB/s | % copy BW |
|---|---|---|---|---|
| tile, direct dtype | BLOCK_V=1024 warps=16 | 0.315 ms | 416 | 99% |
| tile, direct dtype | BLOCK_V=1024 warps=4 | 0.319 ms | 411 | 98% |
| tile, direct dtype | BLOCK_V=1024 warps=8 | 0.319 ms | 411 | 98% |
| tile, direct dtype | BLOCK_V=2048 warps=8 | 0.319 ms | 411 | 98% |
| tile, direct dtype | BLOCK_V=2048 warps=16 | 0.319 ms | 411 | 98% |
| tile, direct dtype | BLOCK_V=4096 warps=4 | 0.319 ms | 411 | 98% |
| tile, direct dtype | BLOCK_V=4096 warps=16 | 0.321 ms | 410 | 98% |
| tile, direct dtype | BLOCK_V=8192 warps=4 | 0.321 ms | 410 | 98% |
| v1 row + fp32 grad + convert | BLOCK_V=8192 warps=16 | 0.954 ms | 413 | 98% |

## 64×151936 bfloat16

**Forward** (fastest 8, and v1)

| kind | config | time | GB/s | % copy BW |
|---|---|---|---|---|
| row | BLOCK_V=8192 warps=8 | 0.074 ms | 264 | 63% |
| row | BLOCK_V=4096 warps=4 | 0.078 ms | 250 | 60% |
| row | BLOCK_V=4096 warps=8 | 0.080 ms | 243 | 58% |
| row (v1) | BLOCK_V=8192 warps=16 | 0.083 ms | 234 | 56% |
| row | BLOCK_V=8192 warps=4 | 0.093 ms | 209 | 50% |
| row | BLOCK_V=2048 warps=4 | 0.098 ms | 198 | 47% |
| row | BLOCK_V=4096 warps=16 | 0.115 ms | 170 | 40% |
| row | BLOCK_V=2048 warps=8 | 0.118 ms | 165 | 39% |

**Backward** (fastest 8, and v1)

| kind | config | time | GB/s | % copy BW |
|---|---|---|---|---|
| tile, direct dtype | BLOCK_V=1024 warps=4 | 0.098 ms | 396 | 94% |
| tile, direct dtype | BLOCK_V=1024 warps=16 | 0.098 ms | 396 | 94% |
| tile, direct dtype | BLOCK_V=2048 warps=8 | 0.098 ms | 396 | 94% |
| tile, direct dtype | BLOCK_V=4096 warps=8 | 0.098 ms | 396 | 94% |
| tile, direct dtype | BLOCK_V=4096 warps=16 | 0.098 ms | 396 | 94% |
| tile, direct dtype | BLOCK_V=1024 warps=8 | 0.099 ms | 392 | 93% |
| tile, direct dtype | BLOCK_V=2048 warps=4 | 0.099 ms | 392 | 93% |
| tile, direct dtype | BLOCK_V=2048 warps=16 | 0.099 ms | 392 | 93% |
| v1 row + fp32 grad + convert | BLOCK_V=8192 warps=16 | 0.291 ms | 401 | 96% |

