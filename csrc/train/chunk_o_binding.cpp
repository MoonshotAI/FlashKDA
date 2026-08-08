#include <torch/extension.h>

torch::Tensor chunk_gla_fwd_o_gk(
    torch::Tensor q,
    torch::Tensor v,
    torch::Tensor g,
    torch::Tensor A,
    torch::Tensor h,
    double scale,
    bool state_v_first,
    std::optional<torch::Tensor> cu_seqlens,
    int64_t chunk_size,
    std::optional<torch::Tensor> chunk_indices
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("chunk_gla_fwd_o_gk", &chunk_gla_fwd_o_gk, "Chunked GLA/KDA forward output (CUDA)",
        py::arg("q"), py::arg("v"), py::arg("g"), py::arg("A"), py::arg("h"),
        py::arg("scale"),
        py::arg("state_v_first") = false,
        py::arg("cu_seqlens") = py::none(),
        py::arg("chunk_size") = 64,
        py::arg("chunk_indices") = py::none());
}
