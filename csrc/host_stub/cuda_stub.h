// Host-compile shim so csrc/*.cu can be *type-checked* by g++/clang++ on
// machines without nvcc (CPU CI). The real CUDA runtime / fp16 / bf16 headers
// are used for types (they are host-includable); only the device-side
// builtins that exist solely under __CUDACC__ are stubbed here. This is NOT an
// emulator — nothing here computes correct numerics — it exists so a missing
// include, wrong template argument or signature mismatch in a kernel fails on
// every PR instead of only on the GPU runner. Used by
// scripts/check_cuda_host_compile.sh.
#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <cmath>

#ifndef __CUDACC__
#ifndef __launch_bounds__
#define __launch_bounds__(...)
#endif
static dim3 threadIdx, blockIdx, blockDim, gridDim;
inline void __syncthreads() {}
inline float __shfl_xor_sync(unsigned, float v, int) { return v; }
inline float __expf(float x) { return std::exp(x); }
inline float __logf(float x) { return std::log(x); }
inline float rsqrtf(float x) { return 1.0f / std::sqrt(x); }
// kernel<<<grid, block, smem, stream>>>(args) is rewritten by the check script
// into kernel_launch(kernel<...>, grid, block, smem, stream, args).
template <typename F, typename... Args>
inline void kernel_launch(F f, long long, int, int, cudaStream_t, Args... args) {
  f(args...);
}
#endif
