// Dev-mode pybind bindings for the KDA forward intra-chunk kernels (intra.cu).
// At integration time these move into csrc/train/binding.cpp / train_ops.h.

#include <torch/extension.h>

#include <optional>

void chunk_kda_fwd_intra_sub_chunk(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor g,
    torch::Tensor beta,
    torch::Tensor Aqk,
    torch::Tensor Akkd,
    double scale,
    int64_t chunk_size,
    std::optional<torch::Tensor> cu_seqlens,
    std::optional<torch::Tensor> chunk_indices
);

void chunk_kda_fwd_intra_token_parallel(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor g,
    torch::Tensor beta,
    torch::Tensor Aqk,
    torch::Tensor Akkd,
    double scale,
    int64_t chunk_size,
    std::optional<torch::Tensor> cu_seqlens
);

void chunk_kda_fwd_inter_solve_fused(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor g,
    torch::Tensor beta,
    torch::Tensor Aqk,
    torch::Tensor Akkd,
    torch::Tensor Akk,
    double scale,
    int64_t chunk_size,
    bool safe_gate,
    std::optional<torch::Tensor> cu_seqlens,
    std::optional<torch::Tensor> chunk_indices
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("chunk_kda_fwd_intra_sub_chunk", &chunk_kda_fwd_intra_sub_chunk,
          "KDA intra sub-chunk diagonal blocks (safe_gate path)");
    m.def("chunk_kda_fwd_intra_token_parallel", &chunk_kda_fwd_intra_token_parallel,
          "KDA intra token-parallel diagonal blocks (non safe_gate path)");
    m.def("chunk_kda_fwd_inter_solve_fused", &chunk_kda_fwd_inter_solve_fused,
          "KDA intra off-diagonal blocks + merged tril solve");
}
