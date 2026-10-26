# Kernel comparison — `rtx3080-laptop`

Generated 2026-10-26 by `python scripts/bench_kernels.py --tag rtx3080-laptop` on **NVIDIA GeForce RTX 3080 Ti Laptop GPU** (torch 2.14.1+cu126, CUDA 12.6).
Forward + backward, median of 20 iterations. Errors are max |Δ| against a float64 reference;
gradients are held to 10× the output tolerance. *naive* is the unfused PyTorch formula; *torch* is Kiln's
reference path; *triton* and *cuda* are the fused kernels. Raw data: `kernels_rtx3080-laptop.json`.

Not run on this machine: **cuda** (kiln._cuda not built (needs nvcc at install time)).

| kernel | shape | dtype | backend | time | peak extra memory | max error (outputs / grad) | within tolerance |
|---|---|---|---|---|---|---|---|
| logprob+entropy | 4096×32000 | bfloat16 | naive | 38.46 ms | 2500 MB | 1.2e-06 / 2.8e-06 / 7.8e-03 | ✅ |
| logprob+entropy | 4096×32000 | bfloat16 | torch | 50.57 ms | 4000 MB | 1.4e-06 / 1.6e-05 / 7.8e-03 | ✅ |
| logprob+entropy | 4096×32000 | bfloat16 | triton | 6.28 ms | 750 MB | 1.4e-06 / 6.7e-06 / 7.8e-03 | ✅ |
| logprob+entropy | 4096×128256 | bfloat16 | naive | 153.62 ms | 10020 MB | 1.3e-06 / 3.9e-06 / 1.6e-02 | ✅ |
| logprob+entropy | 4096×128256 | bfloat16 | torch | 1416.00 ms | 16032 MB | 1.2e-06 / 1.6e-05 / 1.6e-02 | ✅ |
| logprob+entropy | 4096×128256 | bfloat16 | triton | 20.77 ms | 3006 MB | 1.3e-06 / 3.9e-06 / 1.6e-02 | ✅ |
| logprob+entropy | 2048×151936 | bfloat16 | naive | 91.23 ms | 5940 MB | 1.2e-06 / 4.4e-06 / 7.8e-03 | ✅ |
| logprob+entropy | 2048×151936 | bfloat16 | torch | 119.54 ms | 9504 MB | 1.3e-06 / 1.6e-05 / 7.8e-03 | ✅ |
| logprob+entropy | 2048×151936 | bfloat16 | triton | 15.19 ms | 1783 MB | 1.2e-06 / 3.9e-06 / 7.8e-03 | ✅ |
| rmsnorm | 16384×4096 | bfloat16 | naive | 32.47 ms | 1536 MB | 1.6e-02 / 1.6e-02 / 6.0e-05 | ✅ |
| rmsnorm | 16384×4096 | bfloat16 | torch | 30.96 ms | 1408 MB | 1.6e-02 / 1.6e-02 / 6.0e-05 | ✅ |
| rmsnorm | 16384×4096 | bfloat16 | triton | 4.61 ms | 556 MB | 1.6e-02 / 1.6e-02 / 6.0e-05 | ✅ |
| rmsnorm | 8192×8192 | bfloat16 | naive | 26.90 ms | 1536 MB | 1.6e-02 / 1.6e-02 / 1.2e-04 | ✅ |
| rmsnorm | 8192×8192 | bfloat16 | torch | 30.94 ms | 1408 MB | 1.6e-02 / 1.6e-02 / 1.2e-04 | ✅ |
| rmsnorm | 8192×8192 | bfloat16 | triton | 4.52 ms | 556 MB | 1.6e-02 / 1.6e-02 / 1.2e-04 | ✅ |
| rope | 4×4096×32×128 | bfloat16 | naive | 19.02 ms | 520 MB | 4.9e-07 / 3.1e-02 | ✅ |
| rope | 4×4096×32×128 | bfloat16 | torch | 22.01 ms | 904 MB | 1.6e-02 / 1.6e-02 | ✅ |
| rope | 4×4096×32×128 | bfloat16 | triton | 5.52 ms | 256 MB | 1.6e-02 / 1.6e-02 | ✅ |
| logprob+entropy | 4096×32000 | float32 | naive | 40.51 ms | 2500 MB | 2.0e-06 / 2.2e-06 / 2.1e-06 | ✅ |
| logprob+entropy | 4096×32000 | float32 | torch | 52.40 ms | 3500 MB | 1.8e-06 / 1.5e-05 / 7.9e-06 | ✅ |
| logprob+entropy | 4096×32000 | float32 | triton | 5.50 ms | 500 MB | 1.8e-06 / 6.0e-06 / 2.3e-06 | ✅ |
| logprob+entropy | 4096×128256 | float32 | naive | 169.22 ms | 10020 MB | 2.1e-06 / 3.0e-06 / 1.4e-06 | ✅ |
| logprob+entropy | 4096×128256 | float32 | torch | 1194.53 ms | 14028 MB | 2.1e-06 / 1.5e-05 / 7.0e-06 | ✅ |
| logprob+entropy | 4096×128256 | float32 | triton | 17.93 ms | 2004 MB | 2.1e-06 / 3.8e-06 / 4.3e-06 | ✅ |
| logprob+entropy | 2048×151936 | float32 | naive | 96.15 ms | 5940 MB | 2.0e-06 / 3.1e-06 / 1.2e-06 | ✅ |
| logprob+entropy | 2048×151936 | float32 | torch | 118.27 ms | 8316 MB | 2.0e-06 / 1.2e-05 / 4.5e-06 | ✅ |
| logprob+entropy | 2048×151936 | float32 | triton | 10.49 ms | 1188 MB | 2.0e-06 / 4.8e-06 / 2.7e-06 | ✅ |
| rmsnorm | 16384×4096 | float32 | naive | 26.06 ms | 1280 MB | 8.8e-07 / 1.1e-06 / 4.5e-09 | ✅ |
| rmsnorm | 16384×4096 | float32 | torch | 18.75 ms | 1068 MB | 8.8e-07 / 1.1e-06 / 4.5e-09 | ✅ |
| rmsnorm | 16384×4096 | float32 | triton | 6.91 ms | 812 MB | 1.2e-06 / 1.2e-06 / 5.4e-09 | ✅ |
| rmsnorm | 8192×8192 | float32 | naive | 21.64 ms | 1280 MB | 1.0e-06 / 1.1e-06 / 7.2e-09 | ✅ |
| rmsnorm | 8192×8192 | float32 | torch | 16.50 ms | 1068 MB | 1.0e-06 / 1.1e-06 / 7.2e-09 | ✅ |
| rmsnorm | 8192×8192 | float32 | triton | 8.33 ms | 812 MB | 1.2e-06 / 1.2e-06 / 6.8e-09 | ✅ |
| rope | 4×4096×32×128 | float32 | naive | 21.05 ms | 640 MB | 4.7e-07 / 4.6e-07 | ✅ |
| rope | 4×4096×32×128 | float32 | torch | 13.42 ms | 520 MB | 4.7e-07 / 4.6e-07 | ✅ |
| rope | 4×4096×32×128 | float32 | triton | 6.55 ms | 512 MB | 4.7e-07 / 4.6e-07 | ✅ |
