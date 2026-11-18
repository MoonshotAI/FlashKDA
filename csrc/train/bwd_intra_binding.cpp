#include <torch/extension.h>

// fla/ops/kda/chunk_intra.py::chunk_kda_bwd_intra.
// dq/dk/db/dg are the upstream (fused kernel) gradients; the kernel accumulates the
// intra-chunk parts and the host wrapper reduces db2 over the NK dim and adds db,
// exactly like the Triton host. Returns (dq2, dk2, db_out, dg2), all fp32.
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> chunk_kda_bwd_intra(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor g,
    torch::Tensor beta,
    torch::Tensor dAqk,
    torch::Tensor dAkk,
    torch::Tensor dq,
    torch::Tensor dk,
    torch::Tensor db,
    torch::Tensor dg,
    bool safe_gate,
    std::optional<torch::Tensor> cu_seqlens,
    std::optional<torch::Tensor> chunk_indices,
    int64_t chunk_size
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("chunk_kda_bwd_intra", &chunk_kda_bwd_intra, "KDA backward intra-chunk dq/dk/db/dg (CUDA)",
        py::arg("q"), py::arg("k"), py::arg("g"), py::arg("beta"),
        py::arg("dAqk"), py::arg("dAkk"),
        py::arg("dq"), py::arg("dk"), py::arg("db"), py::arg("dg"),
        py::arg("safe_gate"), py::arg("cu_seqlens") = py::none(),
        py::arg("chunk_indices") = py::none(), py::arg("chunk_size") = 64);
}
