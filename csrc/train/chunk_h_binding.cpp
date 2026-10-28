#include <torch/extension.h>

std::tuple<torch::Tensor, torch::Tensor, std::optional<torch::Tensor>>
chunk_gated_delta_rule_fwd_h(
    torch::Tensor kg,
    torch::Tensor w,
    torch::Tensor u,
    torch::Tensor gk,
    std::optional<torch::Tensor> initial_state,
    bool output_final_state,
    int64_t chunk_size,
    bool state_v_first,
    std::optional<torch::Tensor> cu_seqlens,
    std::optional<torch::Tensor> chunk_offsets,
    int64_t nt_total
);

std::tuple<torch::Tensor, std::optional<torch::Tensor>, torch::Tensor>
chunk_gated_delta_rule_bwd_dhu(
    torch::Tensor qg,
    torch::Tensor kg,
    torch::Tensor w,
    torch::Tensor gk,
    torch::Tensor do_,
    torch::Tensor dv,
    std::optional<torch::Tensor> h0,
    std::optional<torch::Tensor> dht,
    double scale,
    int64_t chunk_size,
    bool state_v_first,
    std::optional<torch::Tensor> cu_seqlens,
    std::optional<torch::Tensor> chunk_offsets,
    int64_t nt_total
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("chunk_gated_delta_rule_fwd_h", &chunk_gated_delta_rule_fwd_h,
        "KDA chunked state forward h (CUDA)",
        py::arg("kg"), py::arg("w"), py::arg("u"), py::arg("gk"),
        py::arg("initial_state") = py::none(),
        py::arg("output_final_state") = false,
        py::arg("chunk_size") = 64,
        py::arg("state_v_first") = false,
        py::arg("cu_seqlens") = py::none(),
        py::arg("chunk_offsets") = py::none(),
        py::arg("nt_total") = 0);
    m.def("chunk_gated_delta_rule_bwd_dhu", &chunk_gated_delta_rule_bwd_dhu,
        "KDA chunked state backward dhu (CUDA)",
        py::arg("qg"), py::arg("kg"), py::arg("w"), py::arg("gk"),
        py::arg("do_"), py::arg("dv"),
        py::arg("h0") = py::none(),
        py::arg("dht") = py::none(),
        py::arg("scale") = 1.0,
        py::arg("chunk_size") = 64,
        py::arg("state_v_first") = false,
        py::arg("cu_seqlens") = py::none(),
        py::arg("chunk_offsets") = py::none(),
        py::arg("nt_total") = 0);
}
