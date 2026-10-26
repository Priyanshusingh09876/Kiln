# Kernel comparison — `cpu`

Generated 2026-10-26 by `python scripts/bench_kernels.py --tag cpu` on **CPU (x86_64)** (torch 2.14.1+cu130, CUDA 13.0).
Forward + backward, median of 5 iterations. Errors are max |Δ| against a float64 reference;
gradients are held to 10× the output tolerance. *naive* is the unfused PyTorch formula; *torch* is Kiln's
reference path; *triton* and *cuda* are the fused kernels. Raw data: `kernels_cpu.json`.

Not run on this machine: **triton** (no CUDA device); **cuda** (no CUDA device).

| kernel | shape | dtype | backend | time | peak extra memory | max error (outputs / grad) | within tolerance |
|---|---|---|---|---|---|---|---|
| logprob+entropy | 256×4096 | float32 | naive | 9.84 ms | n/a | 1.6e-06 / 1.5e-06 / 1.5e-06 | ✅ |
| logprob+entropy | 256×4096 | float32 | torch | 4.58 ms | n/a | 1.3e-06 / 6.9e-06 / 5.5e-06 | ✅ |
| logprob+entropy | 256×32000 | float32 | naive | 65.52 ms | n/a | 3.7e-06 / 7.5e-06 / 8.1e-06 | ✅ |
| logprob+entropy | 256×32000 | float32 | torch | 81.92 ms | n/a | 1.5e-06 / 1.4e-05 / 6.2e-06 | ✅ |
| rmsnorm | 2048×1024 | float32 | naive | 8.86 ms | n/a | 9.3e-07 / 1.2e-06 / 1.1e-08 | ✅ |
| rmsnorm | 2048×1024 | float32 | torch | 6.41 ms | n/a | 9.3e-07 / 8.7e-07 / 1.1e-08 | ✅ |
| rope | 2×256×8×64 | float32 | naive | 1.29 ms | n/a | 4.2e-07 / 3.6e-07 | ✅ |
| rope | 2×256×8×64 | float32 | torch | 1.24 ms | n/a | 4.2e-07 / 3.6e-07 | ✅ |

On CPU only the reference paths run, so this table establishes **correctness** of the reference
implementations and the harness itself; timings are CPU timings and say nothing about the GPU kernels.
Run the same command on a CUDA machine to fill in Triton and CUDA C++ rows, memory, and speed.
