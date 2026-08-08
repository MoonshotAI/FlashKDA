#include <torch/extension.h>

#include <optional>
#include <vector>

// Defined in bwd_wy_dqkg.cu. fla/ops/kda/chunk_bwd.py::chunk_kda_bwd_wy_dqkg_fused.
// Returns (dq, dk, dv2, db, dg, dAkk).
std::vector<torch::Tensor> chunk_kda_bwd_wy_dqkg_fused(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor v_new,
    torch::Tensor g,
    torch::Tensor beta,
    torch::Tensor A,
    torch::Tensor h,
    torch::Tensor do_,
    torch::Tensor dh,
    torch::Tensor dv,
    double scale,
    bool state_v_first,
    std::optional<torch::Tensor> cu_seqlens,
    std::optional<torch::Tensor> chunk_indices,
    int64_t chunk_size
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("chunk_kda_bwd_wy_dqkg_fused", &chunk_kda_bwd_wy_dqkg_fused,
        "KDA fused backward dq/dk/dv2/dg/db/dAkk (CUDA)",
        py::arg("q"), py::arg("k"), py::arg("v"), py::arg("v_new"),
        py::arg("g"), py::arg("beta"), py::arg("A"), py::arg("h"),
        py::arg("do"), py::arg("dh"), py::arg("dv"),
        py::arg("scale"), py::arg("state_v_first"),
        py::arg("cu_seqlens"), py::arg("chunk_indices"), py::arg("chunk_size"));
}
