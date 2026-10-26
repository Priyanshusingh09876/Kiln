# csrc — native extensions

| file | builds into | needs | what |
|---|---|---|---|
| `radix_tree.cpp` | `kiln._C` | C++17 compiler | Block-granular prefix tree with leaf-LRU eviction for the cache-aware router. Zero-copy NumPy entry points. Differential-tested against the Python reference (`tests/test_native.py`). |
| `fused_logprob_entropy.cu` | `kiln._cuda` | nvcc, sm80+ | One-block-per-row online-softmax log-prob + entropy, forward and single-pass backward. fp32/fp16/bf16 in, fp32 out, honours row stride. |
| `rmsnorm.cu` | `kiln._cuda` | nvcc, sm80+ | RMSNorm forward/backward with fp32 accumulation; per-row `dw` partials reduced once (no atomics). |
| `include/kiln_cuda.cuh` | — | — | Warp/block reductions, online-softmax state merge, dtype converters. |
| `bindings.cpp` | `kiln._cuda` | — | pybind11 module. |
| `host_stub/cuda_stub.h` | — | — | Shim so the `.cu` files type-check under g++ on CPU CI (`scripts/check_cuda_host_compile.sh`). |

Build: `pip install --no-build-isolation -e .` — builds `_C` always and `_cuda` when `nvcc` is on `PATH` / `CUDA_HOME` is set. `KILN_SKIP_NATIVE=1` skips both. Kernel backend selection at runtime: `KILN_KERNEL_BACKEND=auto|cuda|triton|torch` (`auto` = CUDA ext → Triton → PyTorch).

Why both CUDA C++ *and* Triton versions of the same two kernels? Triton is the fast way to iterate and is what the rest of the kernel set (RoPE, FP8 GEMM) uses. The CUDA versions exist for environments where Triton's JIT is unwelcome (inference images, deterministic builds), as the reference for register-level tuning (`-lineinfo` is on for Nsight), and because `KILN_KERNEL_BACKEND` lets the parity tests pin all three paths against each other.

Testing status: the C++ extension is built and tested on every CI run. The CUDA sources are type-checked against real libtorch + CUDA headers on every CI run (`cuda-host-check` job). They have **not yet been compiled by nvcc or executed**; `tests/test_native.py` contains the GPU parity tests that will do that (tier 1 in `docs/VALIDATION.md`).
