#include <torch/extension.h>

// fla/ops/kda/chunk_bwd.py::chunk_kda_bwd_dAv.
// q, k are accepted for signature parity with the Triton host (the kernel does not read them).
// v is v_new, A is Aqk ([B,T,HV,64], same dtype as do). Returns (dA fp32 [B,T,HV,64], dv like do).
std::tuple<torch::Tensor, torch::Tensor> chunk_kda_bwd_dAv(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor do_,
    torch::Tensor A,
    double scale,
    std::optional<torch::Tensor> cu_seqlens,
    std::optional<torch::Tensor> chunk_indices,
    int64_t chunk_size
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("chunk_kda_bwd_dAv", &chunk_kda_bwd_dAv, "KDA backward dAqk + intra dv (CUDA)",
        py::arg("q"), py::arg("k"), py::arg("v"), py::arg("do_"), py::arg("A"),
        py::arg("scale"), py::arg("cu_seqlens") = py::none(),
        py::arg("chunk_indices") = py::none(), py::arg("chunk_size") = 64);
}
