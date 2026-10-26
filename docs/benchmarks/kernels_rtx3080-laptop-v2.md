# Kernel comparison — `rtx3080-laptop-v2`

Generated 2026-10-26 by `python scripts/bench_kernels.py --tag rtx3080-laptop-v2` on **NVIDIA GeForce RTX 3080 Ti Laptop GPU** (torch 2.14.1+cu126, CUDA 12.6).
Forward + backward, median of 20 iterations. Errors are max |Δ| against a float64 reference;
gradients are held to 10× the output tolerance. *naive* is the unfused PyTorch formula, *compile* is
`torch.compile` of it, *liger* is Liger Kernel; *torch* is Kiln's chunked fallback, *triton* and *cuda* are
Kiln's fused kernels. Raw data: `kernels_rtx3080-laptop-v2.json`.

**Bandwidth.** GB/s is the minimum bytes each operation must move divided by its time. This GPU's measured copy bandwidth is **418 GB/s** (spec sheet: 448 GB/s); the % column is relative to the measured copy, the practical ceiling for a memory-bound kernel.

Not run on this machine: **cuda** (kiln._cuda not built (needs nvcc at install time)).

*Liger log-prob rows compute cross-entropy only (no entropy) and overwrite the logits with the gradient,
so they do less work and need no gradient buffer; they are a reference point, not a like-for-like match.*

| kernel | shape | dtype | backend | time | vs naive | vs compile | GB/s | % of copy BW | peak extra memory | max error (outputs / grad) | ok |
|---|---|---|---|---|---|---|---|---|---|---|---|
| logprob+entropy | 4096×32000 | bfloat16 | naive | 38.93 ms | — | 0.22× | 20 | 5% | 2500 MB | 1.2e-06 / 2.8e-06 / 7.8e-03 | ✅ |
| logprob+entropy | 4096×32000 | bfloat16 | compile | 8.72 ms | 4.47× | — | 90 | 22% | 1250 MB | 1.1e-06 / 2.8e-06 / 7.8e-03 | ✅ |
| logprob+entropy | 4096×32000 | bfloat16 | liger | 3.91 ms | 9.96× | 2.23× | 201 | 48% | 0 MB | 6.2e-02 / 1.9e-03 | ❌ |
| logprob+entropy | 4096×32000 | bfloat16 | torch | 50.43 ms | 0.77× | 0.17× | 16 | 4% | 1506 MB | 1.4e-06 / 1.6e-05 / 7.8e-03 | ✅ |
| logprob+entropy | 4096×32000 | bfloat16 | triton | 2.15 ms | 18.07× | 4.05× | 365 | 87% | 250 MB | 1.4e-06 / 5.4e-06 / 7.8e-03 | ✅ |
| logprob+entropy | 4096×128256 | bfloat16 | naive | 153.56 ms | — | 0.27× | 21 | 5% | 10020 MB | 1.3e-06 / 3.9e-06 / 1.6e-02 | ✅ |
| logprob+entropy | 4096×128256 | bfloat16 | compile | 40.86 ms | 3.76× | — | 77 | 18% | 5010 MB | 1.2e-06 / 4.2e-06 / 1.6e-02 | ✅ |
| logprob+entropy | 4096×128256 | bfloat16 | liger | 14.66 ms | 10.48× | 2.79× | 215 | 51% | 0 MB | 6.2e-02 / 1.8e-03 | ❌ |
| logprob+entropy | 4096×128256 | bfloat16 | torch | 175.90 ms | 0.87× | 0.23× | 18 | 4% | 2282 MB | 1.2e-06 / 1.6e-05 / 1.6e-02 | ✅ |
| logprob+entropy | 4096×128256 | bfloat16 | triton | 10.32 ms | 14.88× | 3.96× | 306 | 73% | 1002 MB | 1.3e-06 / 4.9e-06 / 1.6e-02 | ✅ |
| logprob+entropy | 2048×151936 | bfloat16 | naive | 91.23 ms | — | 0.23× | 20 | 5% | 5940 MB | 1.2e-06 / 4.4e-06 / 7.8e-03 | ✅ |
| logprob+entropy | 2048×151936 | bfloat16 | compile | 20.84 ms | 4.38× | — | 90 | 21% | 2970 MB | 1.3e-06 / 4.8e-06 / 7.8e-03 | ✅ |
| logprob+entropy | 2048×151936 | bfloat16 | liger | 7.17 ms | 12.73× | 2.91× | 260 | 62% | 0 MB | 6.2e-02 / 1.3e-03 | ❌ |
| logprob+entropy | 2048×151936 | bfloat16 | torch | 102.56 ms | 0.89× | 0.20× | 18 | 4% | 1874 MB | 1.3e-06 / 1.6e-05 / 7.8e-03 | ✅ |
| logprob+entropy | 2048×151936 | bfloat16 | triton | 5.34 ms | 17.08× | 3.90× | 350 | 84% | 594 MB | 1.2e-06 / 3.9e-06 / 7.8e-03 | ✅ |
| logprob+entropy | 256×128256 | bfloat16 | naive | 11.44 ms | — | 0.36× | 17 | 4% | 630 MB | 1.3e-06 / 2.9e-06 / 7.7e-03 | ✅ |
| logprob+entropy | 256×128256 | bfloat16 | compile | 4.12 ms | 2.78× | — | 48 | 11% | 315 MB | 1.1e-06 / 3.6e-06 / 7.7e-03 | ✅ |
| logprob+entropy | 256×128256 | bfloat16 | liger | 1.48 ms | 7.75× | 2.79× | 133 | 32% | 0 MB | 6.2e-02 / 9.1e-04 | ❌ |
| logprob+entropy | 256×128256 | bfloat16 | torch | 12.96 ms | 0.88× | 0.32× | 15 | 4% | 441 MB | 1.1e-06 / 1.6e-05 / 7.7e-03 | ✅ |
| logprob+entropy | 256×128256 | bfloat16 | triton | 0.90 ms | 12.77× | 4.60× | 220 | 53% | 63 MB | 1.2e-06 / 4.5e-06 / 7.7e-03 | ✅ |
| logprob+entropy | 64×151936 | bfloat16 | naive | 3.52 ms | — | 0.32× | 17 | 4% | 190 MB | 1.1e-06 / 3.2e-06 / 4.5e-03 | ✅ |
| logprob+entropy | 64×151936 | bfloat16 | compile | 1.13 ms | 3.13× | — | 52 | 12% | 76 MB | 1.1e-06 / 2.7e-06 / 4.5e-03 | ✅ |
| logprob+entropy | 64×151936 | bfloat16 | liger | 1.13 ms | 3.12× | 1.00× | 52 | 12% | 0 MB | 6.0e-02 / 9.7e-04 | ❌ |
| logprob+entropy | 64×151936 | bfloat16 | torch | 3.46 ms | 1.02× | 0.33× | 17 | 4% | 133 MB | 1.1e-06 / 1.2e-05 / 4.5e-03 | ✅ |
| logprob+entropy | 64×151936 | bfloat16 | triton | 1.43 ms | 2.46× | 0.79× | 41 | 10% | 19 MB | 1.1e-06 / 2.3e-06 / 4.5e-03 | ✅ |
| rmsnorm | 16384×4096 | bfloat16 | naive | 23.79 ms | — | 0.12× | 28 | 7% | 1664 MB | 1.6e-02 / 1.6e-02 / 6.0e-05 | ✅ |
| rmsnorm | 16384×4096 | bfloat16 | compile | 2.83 ms | 8.42× | — | 237 | 57% | 264 MB | 1.6e-02 / 1.6e-02 / 6.0e-05 | ✅ |
| rmsnorm | 16384×4096 | bfloat16 | liger | 2.85 ms | 8.34× | 0.99× | 235 | 56% | 257 MB | 3.3e-02 / 1.6e-02 / 1.0e-04 | ❌ |
| rmsnorm | 16384×4096 | bfloat16 | torch | 26.24 ms | 0.91× | 0.11× | 26 | 6% | 1536 MB | 1.6e-02 / 1.6e-02 / 6.0e-05 | ✅ |
| rmsnorm | 16384×4096 | bfloat16 | triton | 4.29 ms | 5.54× | 0.66× | 156 | 37% | 556 MB | 1.6e-02 / 1.6e-02 / 6.0e-05 | ✅ |
| rmsnorm | 8192×8192 | bfloat16 | naive | 32.30 ms | — | 0.11× | 21 | 5% | 1664 MB | 1.6e-02 / 1.6e-02 / 1.2e-04 | ✅ |
| rmsnorm | 8192×8192 | bfloat16 | compile | 3.48 ms | 9.28× | — | 193 | 46% | 256 MB | 1.6e-02 / 1.6e-02 / 1.2e-04 | ✅ |
| rmsnorm | 8192×8192 | bfloat16 | liger | 22.61 ms | 1.43× | 0.15× | 30 | 7% | 258 MB | 3.3e-02 / 1.6e-02 / 1.5e-04 | ❌ |
| rmsnorm | 8192×8192 | bfloat16 | torch | 23.08 ms | 1.40× | 0.15× | 29 | 7% | 1536 MB | 1.6e-02 / 1.6e-02 / 1.2e-04 | ✅ |
| rmsnorm | 8192×8192 | bfloat16 | triton | 4.30 ms | 7.51× | 0.81× | 156 | 37% | 556 MB | 1.6e-02 / 1.6e-02 / 1.2e-04 | ✅ |
| rope | 4×4096×32×128 | bfloat16 | naive | 20.67 ms | — | 0.22× | 26 | 6% | 968 MB | 4.9e-07 / 3.1e-02 | ✅ |
| rope | 4×4096×32×128 | bfloat16 | compile | 4.45 ms | 4.65× | — | 121 | 29% | 640 MB | 4.7e-07 / 1.6e-02 | ✅ |
| rope | 4×4096×32×128 | bfloat16 | torch | 22.07 ms | 0.94× | 0.20× | 24 | 6% | 1032 MB | 1.6e-02 / 1.6e-02 | ✅ |
| rope | 4×4096×32×128 | bfloat16 | triton | 4.73 ms | 4.37× | 0.94× | 114 | 27% | 256 MB | 1.6e-02 / 1.6e-02 | ✅ |
| logprob+entropy | 4096×32000 | float32 | naive | 40.49 ms | — | 0.24× | 39 | 9% | 2500 MB | 2.0e-06 / 2.2e-06 / 2.1e-06 | ✅ |
| logprob+entropy | 4096×32000 | float32 | compile | 9.72 ms | 4.17× | — | 162 | 39% | 1000 MB | 2.0e-06 / 2.4e-06 / 9.7e-07 | ✅ |
| logprob+entropy | 4096×32000 | float32 | liger | 5.96 ms | 6.79× | 1.63× | 264 | 63% | 0 MB | 1.8e-06 / 2.5e-07 | ✅ |
| logprob+entropy | 4096×32000 | float32 | torch | 46.87 ms | 0.86× | 0.21× | 34 | 8% | 1512 MB | 1.8e-06 / 1.5e-05 / 7.9e-06 | ✅ |
| logprob+entropy | 4096×32000 | float32 | triton | 3.92 ms | 10.33× | 2.48× | 401 | 96% | 500 MB | 1.8e-06 / 4.0e-06 / 3.3e-06 | ✅ |
| logprob+entropy | 4096×128256 | float32 | naive | 168.83 ms | — | 0.41× | 37 | 9% | 10020 MB | 2.1e-06 / 3.0e-06 / 1.4e-06 | ✅ |
| logprob+entropy | 4096×128256 | float32 | compile | 70.01 ms | 2.41× | — | 90 | 22% | 4008 MB | 2.1e-06 / 2.8e-06 / 1.2e-06 | ✅ |
| logprob+entropy | 4096×128256 | float32 | liger | 16.13 ms | 10.47× | 4.34× | 391 | 93% | 0 MB | 2.1e-06 / 1.8e-07 | ✅ |
| logprob+entropy | 4096×128256 | float32 | torch | 170.58 ms | 0.99× | 0.41× | 37 | 9% | 3028 MB | 2.1e-06 / 1.5e-05 / 7.0e-06 | ✅ |
| logprob+entropy | 4096×128256 | float32 | triton | 21.60 ms | 7.82× | 3.24× | 292 | 70% | 2004 MB | 2.1e-06 / 6.9e-06 / 3.8e-06 | ✅ |
| logprob+entropy | 2048×151936 | float32 | naive | 82.98 ms | — | 0.42× | 45 | 11% | 5940 MB | 2.0e-06 / 3.1e-06 / 1.2e-06 | ✅ |
| logprob+entropy | 2048×151936 | float32 | compile | 34.59 ms | 2.40× | — | 108 | 26% | 2376 MB | 2.0e-06 / 2.7e-06 / 1.3e-06 | ✅ |
| logprob+entropy | 2048×151936 | float32 | liger | 9.80 ms | 8.46× | 3.53× | 381 | 91% | 0 MB | 2.0e-06 / 7.7e-08 | ✅ |
| logprob+entropy | 2048×151936 | float32 | torch | 95.05 ms | 0.87× | 0.36× | 39 | 9% | 2212 MB | 2.0e-06 / 1.2e-05 / 4.5e-06 | ✅ |
| logprob+entropy | 2048×151936 | float32 | triton | 9.22 ms | 9.00× | 3.75× | 405 | 97% | 1188 MB | 2.2e-06 / 9.3e-06 / 5.2e-06 | ✅ |
| logprob+entropy | 256×128256 | float32 | naive | 10.37 ms | — | 1.01× | 38 | 9% | 630 MB | 1.7e-06 / 2.7e-06 / 8.9e-07 | ✅ |
| logprob+entropy | 256×128256 | float32 | compile | 10.46 ms | 0.99× | — | 38 | 9% | 630 MB | 1.7e-06 / 2.7e-06 / 8.9e-07 | ✅ |
| logprob+entropy | 256×128256 | float32 | liger | 1.98 ms | 5.23× | 5.28× | 199 | 48% | 0 MB | 1.8e-06 / 4.9e-08 | ✅ |
| logprob+entropy | 256×128256 | float32 | torch | 11.99 ms | 0.86× | 0.87× | 33 | 8% | 504 MB | 1.8e-06 / 1.2e-05 / 1.2e-06 | ✅ |
| logprob+entropy | 256×128256 | float32 | triton | 1.18 ms | 8.79× | 8.87× | 334 | 80% | 126 MB | 1.8e-06 / 3.3e-06 / 1.1e-06 | ✅ |
| logprob+entropy | 64×151936 | float32 | naive | 2.74 ms | — | 1.01× | 43 | 10% | 190 MB | 1.5e-06 / 2.3e-06 / 1.3e-06 | ✅ |
| logprob+entropy | 64×151936 | float32 | compile | 2.78 ms | 0.99× | — | 42 | 10% | 190 MB | 1.5e-06 / 2.3e-06 / 1.3e-06 | ✅ |
| logprob+entropy | 64×151936 | float32 | liger | 1.43 ms | 1.91× | 1.94× | 81 | 19% | 0 MB | 1.7e-06 / 4.4e-08 | ✅ |
| logprob+entropy | 64×151936 | float32 | torch | 3.74 ms | 0.73× | 0.74× | 31 | 7% | 152 MB | 1.7e-06 / 1.1e-05 / 1.9e-06 | ✅ |
| logprob+entropy | 64×151936 | float32 | triton | 1.22 ms | 2.25× | 2.28× | 96 | 23% | 37 MB | 1.7e-06 / 2.0e-06 / 2.4e-06 | ✅ |
| rmsnorm | 16384×4096 | float32 | naive | 19.37 ms | — | 0.27× | 69 | 17% | 1536 MB | 8.8e-07 / 1.1e-06 / 4.5e-09 | ✅ |
| rmsnorm | 16384×4096 | float32 | compile | 5.22 ms | 3.71× | — | 257 | 62% | 520 MB | 9.6e-07 / 9.1e-07 / 4.9e-09 | ✅ |
| rmsnorm | 16384×4096 | float32 | liger | 5.84 ms | 3.32× | 0.89× | 230 | 55% | 513 MB | 9.6e-07 / 9.1e-07 / 9.7e-09 | ✅ |
| rmsnorm | 16384×4096 | float32 | torch | 19.47 ms | 0.99× | 0.27× | 69 | 16% | 1324 MB | 8.8e-07 / 1.1e-06 / 4.5e-09 | ✅ |
| rmsnorm | 16384×4096 | float32 | triton | 6.45 ms | 3.00× | 0.81× | 208 | 50% | 812 MB | 1.2e-06 / 1.2e-06 / 5.4e-09 | ✅ |
| rmsnorm | 8192×8192 | float32 | naive | 19.75 ms | — | 0.28× | 68 | 16% | 1536 MB | 1.0e-06 / 1.1e-06 / 7.2e-09 | ✅ |
| rmsnorm | 8192×8192 | float32 | compile | 5.46 ms | 3.62× | — | 246 | 59% | 512 MB | 1.0e-06 / 9.6e-07 / 9.9e-09 | ✅ |
| rmsnorm | 8192×8192 | float32 | liger | 5.78 ms | 3.42× | 0.94× | 232 | 56% | 514 MB | 1.0e-06 / 9.1e-07 / 1.0e-08 | ✅ |
| rmsnorm | 8192×8192 | float32 | torch | 23.36 ms | 0.85× | 0.23× | 57 | 14% | 1324 MB | 1.0e-06 / 1.1e-06 / 7.2e-09 | ✅ |
| rmsnorm | 8192×8192 | float32 | triton | 6.47 ms | 3.05× | 0.84× | 208 | 50% | 812 MB | 1.2e-06 / 1.2e-06 / 6.8e-09 | ✅ |
| rope | 4×4096×32×128 | float32 | naive | 18.35 ms | — | 0.18× | 59 | 14% | 896 MB | 4.7e-07 / 4.6e-07 | ✅ |
| rope | 4×4096×32×128 | float32 | compile | 3.31 ms | 5.55× | — | 325 | 78% | 512 MB | 4.7e-07 / 4.6e-07 | ✅ |
| rope | 4×4096×32×128 | float32 | torch | 16.68 ms | 1.10× | 0.20× | 64 | 15% | 776 MB | 4.7e-07 / 4.6e-07 | ✅ |
| rope | 4×4096×32×128 | float32 | triton | 4.41 ms | 4.16× | 0.75× | 243 | 58% | 512 MB | 4.7e-07 / 4.6e-07 | ✅ |
