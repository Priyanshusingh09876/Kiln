// RMSNorm forward + backward, one block per row, fp32 accumulation.
//
//   rstd = 1 / sqrt(mean(x²) + eps) ; y = x · rstd · w
//   dx   = rstd · (w·dy − x̂ · mean(w·dy·x̂))      with x̂ = x · rstd
//   dw   = Σ_rows dy · x̂   (per-row partials, reduced once on the host side: no atomics)

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>

#include "include/kiln_cuda.cuh"

namespace kiln {

template <typename T, int THREADS>
__global__ void __launch_bounds__(THREADS) rmsnorm_fwd_kernel(
    const T* __restrict__ x, const T* __restrict__ w, T* __restrict__ y, float* __restrict__ rstd_out,
    int64_t N, int64_t row_stride, float eps) {
  __shared__ float smem[THREADS / kWarpSize];
  const int64_t row = blockIdx.x;
  const T* xr = x + row * row_stride;
  T* yr = y + row * row_stride;

  float ss = 0.f;
  for (int64_t i = threadIdx.x; i < N; i += THREADS) {
    const float v = to_float<T>(xr[i]);
    ss += v * v;
  }
  ss = block_reduce(ss, SumOp(), smem);
  const float rstd = rsqrtf(ss / static_cast<float>(N) + eps);
  if (threadIdx.x == 0) rstd_out[row] = rstd;
  for (int64_t i = threadIdx.x; i < N; i += THREADS) {
    yr[i] = from_float<T>(to_float<T>(xr[i]) * rstd * to_float<T>(w[i]));
  }
}

template <typename T, int THREADS>
__global__ void __launch_bounds__(THREADS) rmsnorm_bwd_kernel(
    const T* __restrict__ x, const T* __restrict__ w, const T* __restrict__ dy, const float* __restrict__ rstd,
    T* __restrict__ dx, float* __restrict__ dw_partial, int64_t N, int64_t row_stride) {
  __shared__ float smem[THREADS / kWarpSize];
  const int64_t row = blockIdx.x;
  const T* xr = x + row * row_stride;
  const T* dyr = dy + row * row_stride;
  T* dxr = dx + row * row_stride;
  float* dwr = dw_partial + row * N;
  const float r = rstd[row];

  float c = 0.f;  // Σ (w·dy·x̂)
  for (int64_t i = threadIdx.x; i < N; i += THREADS) {
    const float xhat = to_float<T>(xr[i]) * r;
    const float wdy = to_float<T>(w[i]) * to_float<T>(dyr[i]);
    c += wdy * xhat;
    dwr[i] = to_float<T>(dyr[i]) * xhat;
  }
  c = block_reduce(c, SumOp(), smem) / static_cast<float>(N);
  for (int64_t i = threadIdx.x; i < N; i += THREADS) {
    const float xhat = to_float<T>(xr[i]) * r;
    const float wdy = to_float<T>(w[i]) * to_float<T>(dyr[i]);
    dxr[i] = from_float<T>(r * (wdy - xhat * c));
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
void fwd_launch(const torch::Tensor& x, const torch::Tensor& w, torch::Tensor& y, torch::Tensor& rstd, float eps,
                cudaStream_t stream) {
  const int64_t R = x.size(0), N = x.size(1);
  const int threads = threads_for(N);
  DISPATCH_THREADS(threads, (rmsnorm_fwd_kernel<T, THREADS><<<R, THREADS, 0, stream>>>(
      reinterpret_cast<const T*>(x.data_ptr()), reinterpret_cast<const T*>(w.data_ptr()),
      reinterpret_cast<T*>(y.data_ptr()), rstd.data_ptr<float>(), N, x.stride(0), eps)));
}

template <typename T>
void bwd_launch(const torch::Tensor& x, const torch::Tensor& w, const torch::Tensor& dy, const torch::Tensor& rstd,
                torch::Tensor& dx, torch::Tensor& dw_partial, cudaStream_t stream) {
  const int64_t R = x.size(0), N = x.size(1);
  const int threads = threads_for(N);
  DISPATCH_THREADS(threads, (rmsnorm_bwd_kernel<T, THREADS><<<R, THREADS, 0, stream>>>(
      reinterpret_cast<const T*>(x.data_ptr()), reinterpret_cast<const T*>(w.data_ptr()),
      reinterpret_cast<const T*>(dy.data_ptr()), rstd.data_ptr<float>(), reinterpret_cast<T*>(dx.data_ptr()),
      dw_partial.data_ptr<float>(), N, x.stride(0))));
}

}  // namespace kiln

std::vector<torch::Tensor> rmsnorm_fwd(torch::Tensor x, torch::Tensor w, double eps) {
  TORCH_CHECK(x.is_cuda() && w.is_cuda(), "inputs must be CUDA tensors");
  TORCH_CHECK(x.dim() == 2 && x.is_contiguous(), "x must be contiguous [R, N]");
  TORCH_CHECK(w.dtype() == x.dtype() && w.numel() == x.size(1), "weight dtype/size mismatch");
  const c10::cuda::CUDAGuard guard(x.device());
  auto stream = c10::cuda::getCurrentCUDAStream().stream();
  auto y = torch::empty_like(x);
  auto rstd = torch::empty({x.size(0)}, x.options().dtype(torch::kFloat32));
  auto wc = w.contiguous();
  switch (x.scalar_type()) {
    case torch::kFloat32: kiln::fwd_launch<float>(x, wc, y, rstd, (float)eps, stream); break;
    case torch::kFloat16: kiln::fwd_launch<__half>(x, wc, y, rstd, (float)eps, stream); break;
    case torch::kBFloat16: kiln::fwd_launch<__nv_bfloat16>(x, wc, y, rstd, (float)eps, stream); break;
    default: TORCH_CHECK(false, "unsupported dtype");
  }
  return {y, rstd};
}

std::vector<torch::Tensor> rmsnorm_bwd(torch::Tensor x, torch::Tensor w, torch::Tensor dy, torch::Tensor rstd) {
  TORCH_CHECK(x.is_cuda(), "x must be CUDA");
  const c10::cuda::CUDAGuard guard(x.device());
  auto stream = c10::cuda::getCurrentCUDAStream().stream();
  auto dyc = dy.contiguous();
  auto wc = w.contiguous();
  auto dx = torch::empty_like(x);
  auto dw_partial = torch::empty({x.size(0), x.size(1)}, x.options().dtype(torch::kFloat32));
  switch (x.scalar_type()) {
    case torch::kFloat32: kiln::bwd_launch<float>(x, wc, dyc, rstd, dx, dw_partial, stream); break;
    case torch::kFloat16: kiln::bwd_launch<__half>(x, wc, dyc, rstd, dx, dw_partial, stream); break;
    case torch::kBFloat16: kiln::bwd_launch<__nv_bfloat16>(x, wc, dyc, rstd, dx, dw_partial, stream); break;
    default: TORCH_CHECK(false, "unsupported dtype");
  }
  return {dx, dw_partial.sum(0).to(w.dtype())};
}
