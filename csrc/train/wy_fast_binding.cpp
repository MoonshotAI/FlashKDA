#include <torch/extension.h>

#include <vector>

std::vector<torch::Tensor> recompute_w_u_fwd(
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor beta,
    torch::Tensor A,
    torch::Tensor gk,
    std::optional<torch::Tensor> q,
    std::optional<torch::Tensor> cu_seqlens,
    std::optional<torch::Tensor> chunk_indices
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("recompute_w_u_fwd", &recompute_w_u_fwd, "Recompute w/u/qg/kg for KDA forward (CUDA)",
        py::arg("k"), py::arg("v"), py::arg("beta"), py::arg("A"), py::arg("gk"),
        py::arg("q") = py::none(),
        py::arg("cu_seqlens") = py::none(),
        py::arg("chunk_indices") = py::none());
}
