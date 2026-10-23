// Python bindings for the CUDA kernels (built only when nvcc is available).
#include <torch/extension.h>

std::vector<torch::Tensor> fused_logprob_entropy_fwd(torch::Tensor logits, torch::Tensor targets);
torch::Tensor fused_logprob_entropy_bwd(torch::Tensor logits, torch::Tensor targets, torch::Tensor lse,
                                        torch::Tensor ent, torch::Tensor g_logp, torch::Tensor g_ent);
std::vector<torch::Tensor> rmsnorm_fwd(torch::Tensor x, torch::Tensor w, double eps);
std::vector<torch::Tensor> rmsnorm_bwd(torch::Tensor x, torch::Tensor w, torch::Tensor dy, torch::Tensor rstd);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "kiln CUDA kernels: fused log-prob/entropy, RMSNorm";
  m.def("fused_logprob_entropy_fwd", &fused_logprob_entropy_fwd, "online-softmax log-prob + entropy forward");
  m.def("fused_logprob_entropy_bwd", &fused_logprob_entropy_bwd, "single-pass backward");
  m.def("rmsnorm_fwd", &rmsnorm_fwd, "RMSNorm forward");
  m.def("rmsnorm_bwd", &rmsnorm_bwd, "RMSNorm backward (dx, dw)");
}
