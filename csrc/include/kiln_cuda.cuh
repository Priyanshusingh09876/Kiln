// Shared device helpers for Kiln's CUDA kernels.
//
// Conventions: one thread block per row, fp32 accumulation regardless of the
// input dtype, warp-shuffle reductions, and no dynamic shared memory so the
// kernels can be launched without occupancy calculators.
#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cfloat>
#include <cstdint>

namespace kiln {

constexpr int kWarpSize = 32;
constexpr int kMaxThreads = 1024;

// ---------------------------------------------------------------- dtype load
template <typename T>
__device__ __forceinline__ float to_float(T v);
template <>
__device__ __forceinline__ float to_float<float>(float v) { return v; }
template <>
__device__ __forceinline__ float to_float<__half>(__half v) { return __half2float(v); }
template <>
__device__ __forceinline__ float to_float<__nv_bfloat16>(__nv_bfloat16 v) { return __bfloat162float(v); }

template <typename T>
__device__ __forceinline__ T from_float(float v);
template <>
__device__ __forceinline__ float from_float<float>(float v) { return v; }
template <>
__device__ __forceinline__ __half from_float<__half>(float v) { return __float2half(v); }
template <>
__device__ __forceinline__ __nv_bfloat16 from_float<__nv_bfloat16>(float v) { return __float2bfloat16(v); }

// ------------------------------------------------------------- reductions
struct SumOp {
  __device__ __forceinline__ float operator()(float a, float b) const { return a + b; }
  static __device__ __forceinline__ float identity() { return 0.f; }
};
struct MaxOp {
  __device__ __forceinline__ float operator()(float a, float b) const { return fmaxf(a, b); }
  static __device__ __forceinline__ float identity() { return -FLT_MAX; }
};

template <typename Op>
__device__ __forceinline__ float warp_reduce(float v, Op op) {
#pragma unroll
  for (int off = kWarpSize / 2; off > 0; off >>= 1) {
    v = op(v, __shfl_xor_sync(0xffffffffu, v, off));
  }
  return v;
}

// Block-wide reduction; `smem` must hold at least blockDim.x / 32 floats.
// Every thread receives the reduced value.
template <typename Op>
__device__ __forceinline__ float block_reduce(float v, Op op, float* smem) {
  const int lane = threadIdx.x & (kWarpSize - 1);
  const int wid = threadIdx.x >> 5;
  const int nwarps = (blockDim.x + kWarpSize - 1) / kWarpSize;
  v = warp_reduce(v, op);
  __syncthreads();  // protect smem reuse across consecutive reductions
  if (lane == 0) smem[wid] = v;
  __syncthreads();
  v = (threadIdx.x < nwarps) ? smem[threadIdx.x] : Op::identity();
  if (wid == 0) v = warp_reduce(v, op);
  if (threadIdx.x == 0) smem[0] = v;
  __syncthreads();
  return smem[0];
}

// Online-softmax state: running max m, sum l = Σ exp(x−m), s = Σ exp(x−m)·x.
struct SoftmaxState {
  float m, l, s;
  __device__ __forceinline__ SoftmaxState() : m(-FLT_MAX), l(0.f), s(0.f) {}
  __device__ __forceinline__ void push(float x) {
    const float m_new = fmaxf(m, x);
    const float scale = __expf(m - m_new);
    const float e = __expf(x - m_new);
    l = l * scale + e;
    s = s * scale + e * x;
    m = m_new;
  }
  __device__ __forceinline__ void merge(const SoftmaxState& o) {
    const float m_new = fmaxf(m, o.m);
    const float a = __expf(m - m_new), b = __expf(o.m - m_new);
    l = l * a + o.l * b;
    s = s * a + o.s * b;
    m = m_new;
  }
};

__device__ __forceinline__ SoftmaxState warp_merge(SoftmaxState st) {
#pragma unroll
  for (int off = kWarpSize / 2; off > 0; off >>= 1) {
    SoftmaxState o;
    o.m = __shfl_xor_sync(0xffffffffu, st.m, off);
    o.l = __shfl_xor_sync(0xffffffffu, st.l, off);
    o.s = __shfl_xor_sync(0xffffffffu, st.s, off);
    st.merge(o);
  }
  return st;
}

// Block-wide merge of online-softmax states; `smem` holds >= 3 * (blockDim.x/32) floats.
__device__ __forceinline__ SoftmaxState block_merge(SoftmaxState st, float* smem) {
  const int lane = threadIdx.x & (kWarpSize - 1);
  const int wid = threadIdx.x >> 5;
  const int nwarps = (blockDim.x + kWarpSize - 1) / kWarpSize;
  st = warp_merge(st);
  __syncthreads();
  if (lane == 0) {
    smem[3 * wid] = st.m;
    smem[3 * wid + 1] = st.l;
    smem[3 * wid + 2] = st.s;
  }
  __syncthreads();
  SoftmaxState out;
  if (threadIdx.x < nwarps) {
    out.m = smem[3 * threadIdx.x];
    out.l = smem[3 * threadIdx.x + 1];
    out.s = smem[3 * threadIdx.x + 2];
  }
  if (wid == 0) out = warp_merge(out);
  if (threadIdx.x == 0) {
    smem[0] = out.m;
    smem[1] = out.l;
    smem[2] = out.s;
  }
  __syncthreads();
  out.m = smem[0];
  out.l = smem[1];
  out.s = smem[2];
  return out;
}

inline int threads_for(int64_t n) {
  if (n >= 4096) return 1024;
  if (n >= 2048) return 512;
  if (n >= 1024) return 256;
  return 128;
}

}  // namespace kiln
