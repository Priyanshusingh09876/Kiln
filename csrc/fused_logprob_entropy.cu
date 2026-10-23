// Fused token log-prob + entropy, forward and backward, one block per row.
//
// Forward (single pass over the vocabulary, online softmax):
//   m = max x, l = Σ exp(x−m), s = Σ exp(x−m)·x
//   lse = m + log l ; logp = x[t] − lse ; ent = lse − s / l
// Backward (single pass, never materialises p):
//   d logp / dx = onehot(t) − p
//   d ent  / dx = −p · (x − lse + ent)
// Outputs are fp32 regardless of logits dtype; gradient is returned in the
// logits dtype. Row stride is honoured so a sliced [B, T-1, V] view works
// without a copy.

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>

#include "include/kiln_cuda.cuh"

namespace kiln {

template <typename T, int THREADS>
__global__ void __launch_bounds__(THREADS) logprob_entropy_fwd_kernel(
    const T* __restrict__ logits, const int64_t* __restrict__ targets, float* __restrict__ logp,
    float* __restrict__ ent, float* __restrict__ lse_out, int64_t V, int64_t row_stride) {
  __shared__ float smem[3 * (THREADS / kWarpSize)];
  const int64_t row = blockIdx.x;
  const T* x = logits + row * row_stride;
  const int64_t t = targets[row];

  SoftmaxState st;
  float tgt = 0.f;
  for (int64_t i = threadIdx.x; i < V; i += THREADS) {
    const float v = to_float<T>(x[i]);
    st.push(v);
    if (i == t) tgt = v;
  }
  st = block_merge(st, smem);
  // only one thread saw the target; sum broadcasts it
  tgt = block_reduce(tgt, SumOp(), smem);

  if (threadIdx.x == 0) {
    const float lse = st.m + __logf(st.l);
    logp[row] = tgt - lse;
    ent[row] = lse - st.s / st.l;
    lse_out[row] = lse;
  }
}

template <typename T, int THREADS>
__global__ void __launch_bounds__(THREADS) logprob_entropy_bwd_kernel(
    const T* __restrict__ logits, const int64_t* __restrict__ targets, const float* __restrict__ lse,
    const float* __restrict__ ent, const float* __restrict__ g_logp, const float* __restrict__ g_ent,
    T* __restrict__ grad, int64_t V, int64_t row_stride, int64_t grad_stride) {
  const int64_t row = blockIdx.x;
  const T* x = logits + row * row_stride;
  T* g = grad + row * grad_stride;
  const int64_t t = targets[row];
  const float l = lse[row], e = ent[row], gl = g_logp[row], ge = g_ent[row];

  for (int64_t i = threadIdx.x; i < V; i += THREADS) {
    const float xi = to_float<T>(x[i]);
    const float lp = xi - l;          // log p_i
    const float p = __expf(lp);
    float d = -gl * p - ge * p * (lp + e);
    if (i == t) d += gl;
    g[i] = from_float<T>(d);
  }
}

#define DISPATCH_THREADS(THREADS_VAR, ...)              \
  switch (THREADS_VAR) {                                 \
    case 1024: { constexpr int THREADS = 1024; __VA_ARGS__; break; } \
    case 512:  { constexpr int THREADS = 512;  __VA_ARGS__; break; } \
    case 256:  { constexpr int THREADS = 256;  __VA_ARGS__; break; } \
    default:   { constexpr int THREADS = 128;  __VA_ARGS__; break; } \
  }

template <typename T>
void fwd_launch(const torch::Tensor& logits, const torch::Tensor& targets, torch::Tensor& logp,
                torch::Tensor& ent, torch::Tensor& lse, cudaStream_t stream) {
  const int64_t N = logits.size(0), V = logits.size(1);
  const int threads = threads_for(V);
  DISPATCH_THREADS(threads, (logprob_entropy_fwd_kernel<T, THREADS><<<N, THREADS, 0, stream>>>(
      reinterpret_cast<const T*>(logits.data_ptr()), targets.data_ptr<int64_t>(), logp.data_ptr<float>(),
      ent.data_ptr<float>(), lse.data_ptr<float>(), V, logits.stride(0))));
}

template <typename T>
void bwd_launch(const torch::Tensor& logits, const torch::Tensor& targets, const torch::Tensor& lse,
                const torch::Tensor& ent, const torch::Tensor& g_logp, const torch::Tensor& g_ent,
                torch::Tensor& grad, cudaStream_t stream) {
  const int64_t N = logits.size(0), V = logits.size(1);
  const int threads = threads_for(V);
  DISPATCH_THREADS(threads, (logprob_entropy_bwd_kernel<T, THREADS><<<N, THREADS, 0, stream>>>(
      reinterpret_cast<const T*>(logits.data_ptr()), targets.data_ptr<int64_t>(), lse.data_ptr<float>(),
      ent.data_ptr<float>(), g_logp.data_ptr<float>(), g_ent.data_ptr<float>(),
      reinterpret_cast<T*>(grad.data_ptr()), V, logits.stride(0), grad.stride(0))));
}

}  // namespace kiln

std::vector<torch::Tensor> fused_logprob_entropy_fwd(torch::Tensor logits, torch::Tensor targets) {
  TORCH_CHECK(logits.is_cuda() && targets.is_cuda(), "inputs must be CUDA tensors");
  TORCH_CHECK(logits.dim() == 2, "logits must be [N, V]");
  TORCH_CHECK(logits.stride(1) == 1, "logits must be contiguous along V");
  TORCH_CHECK(targets.dtype() == torch::kInt64, "targets must be int64");
  const c10::cuda::CUDAGuard guard(logits.device());
  auto stream = c10::cuda::getCurrentCUDAStream().stream();
  auto opts = logits.options().dtype(torch::kFloat32);
  auto logp = torch::empty({logits.size(0)}, opts);
  auto ent = torch::empty_like(logp);
  auto lse = torch::empty_like(logp);
  auto tg = targets.contiguous();
  switch (logits.scalar_type()) {
    case torch::kFloat32: kiln::fwd_launch<float>(logits, tg, logp, ent, lse, stream); break;
    case torch::kFloat16: kiln::fwd_launch<__half>(logits, tg, logp, ent, lse, stream); break;
    case torch::kBFloat16: kiln::fwd_launch<__nv_bfloat16>(logits, tg, logp, ent, lse, stream); break;
    default: TORCH_CHECK(false, "unsupported dtype");
  }
  return {logp, ent, lse};
}

torch::Tensor fused_logprob_entropy_bwd(torch::Tensor logits, torch::Tensor targets, torch::Tensor lse,
                                        torch::Tensor ent, torch::Tensor g_logp, torch::Tensor g_ent) {
  TORCH_CHECK(logits.is_cuda(), "logits must be CUDA");
  const c10::cuda::CUDAGuard guard(logits.device());
  auto stream = c10::cuda::getCurrentCUDAStream().stream();
  auto grad = torch::empty({logits.size(0), logits.size(1)}, logits.options());
  auto tg = targets.contiguous();
  auto gl = g_logp.contiguous().to(torch::kFloat32), ge = g_ent.contiguous().to(torch::kFloat32);
  switch (logits.scalar_type()) {
    case torch::kFloat32: kiln::bwd_launch<float>(logits, tg, lse, ent, gl, ge, grad, stream); break;
    case torch::kFloat16: kiln::bwd_launch<__half>(logits, tg, lse, ent, gl, ge, grad, stream); break;
    case torch::kBFloat16: kiln::bwd_launch<__nv_bfloat16>(logits, tg, lse, ent, gl, ge, grad, stream); break;
    default: TORCH_CHECK(false, "unsupported dtype");
  }
  return grad;
}
