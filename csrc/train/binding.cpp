#include <torch/extension.h>

#include "train_ops.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("kda_gate_chunk_cumsum", &kda_gate_chunk_cumsum, "KDA gate activation + chunk cumsum (CUDA)",
        py::arg("g"), py::arg("A_log"), py::arg("dt_bias"),
        py::arg("out"), py::arg("scale"), py::arg("has_scale"),
        py::arg("lower_bound"), py::arg("use_lower_bound"),
        py::arg("chunk_size"), py::arg("cu_seqlens"), py::arg("chunk_indices"));
    m.def("chunk_local_cumsum", &chunk_local_cumsum, "Chunk-local (reverse) cumsum (CUDA)",
        py::arg("g"), py::arg("out"), py::arg("scale"), py::arg("has_scale"),
        py::arg("reverse"), py::arg("chunk_size"),
        py::arg("cu_seqlens"), py::arg("chunk_indices"));
    m.def("kda_gate_bwd", &kda_gate_bwd, "KDA gate backward (CUDA)",
        py::arg("g"), py::arg("A_log"), py::arg("dt_bias"),
        py::arg("dyg"), py::arg("dg"), py::arg("dA_partial"),
        py::arg("lower_bound"), py::arg("use_lower_bound"));

    m.def("chunk_kda_fwd_intra_sub_chunk", &chunk_kda_fwd_intra_sub_chunk,
        "KDA intra sub-chunk diagonal blocks (safe_gate path)",
        py::arg("q"), py::arg("k"), py::arg("g"), py::arg("beta"),
        py::arg("Aqk"), py::arg("Akkd"), py::arg("scale"), py::arg("chunk_size"),
        py::arg("cu_seqlens") = py::none(), py::arg("chunk_indices") = py::none());
    m.def("chunk_kda_fwd_intra_token_parallel", &chunk_kda_fwd_intra_token_parallel,
        "KDA intra token-parallel diagonal blocks (non safe_gate path)",
        py::arg("q"), py::arg("k"), py::arg("g"), py::arg("beta"),
        py::arg("Aqk"), py::arg("Akkd"), py::arg("scale"), py::arg("chunk_size"),
        py::arg("cu_seqlens") = py::none());
    m.def("chunk_kda_fwd_inter_solve_fused", &chunk_kda_fwd_inter_solve_fused,
        "KDA intra off-diagonal blocks + merged tril solve",
        py::arg("q"), py::arg("k"), py::arg("g"), py::arg("beta"),
        py::arg("Aqk"), py::arg("Akkd"), py::arg("Akk"),
        py::arg("scale"), py::arg("chunk_size"), py::arg("safe_gate"),
        py::arg("cu_seqlens") = py::none(), py::arg("chunk_indices") = py::none());

    m.def("recompute_w_u_fwd", &recompute_w_u_fwd, "Recompute w/u/qg/kg for KDA forward (CUDA)",
        py::arg("k"), py::arg("v"), py::arg("beta"), py::arg("A"), py::arg("gk"),
        py::arg("q") = py::none(),
        py::arg("cu_seqlens") = py::none(),
        py::arg("chunk_indices") = py::none());

    m.def("chunk_gla_fwd_o_gk", &chunk_gla_fwd_o_gk, "Chunked GLA/KDA forward output (CUDA)",
        py::arg("q"), py::arg("v"), py::arg("g"), py::arg("A"), py::arg("h"),
        py::arg("scale"),
        py::arg("state_v_first") = false,
        py::arg("cu_seqlens") = py::none(),
        py::arg("chunk_size") = 64,
        py::arg("chunk_indices") = py::none());

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

    m.def("chunk_kda_bwd_dAv", &chunk_kda_bwd_dAv, "KDA backward dAqk + intra dv (CUDA)",
        py::arg("q"), py::arg("k"), py::arg("v"), py::arg("do_"), py::arg("A"),
        py::arg("scale"), py::arg("cu_seqlens") = py::none(),
        py::arg("chunk_indices") = py::none(), py::arg("chunk_size") = 64);
    m.def("chunk_kda_bwd_wy_dqkg_fused", &chunk_kda_bwd_wy_dqkg_fused,
        "KDA fused backward dq/dk/dv2/dg/db/dAkk (CUDA)",
        py::arg("q"), py::arg("k"), py::arg("v"), py::arg("v_new"),
        py::arg("g"), py::arg("beta"), py::arg("A"), py::arg("h"),
        py::arg("do"), py::arg("dh"), py::arg("dv"),
        py::arg("scale"), py::arg("state_v_first"),
        py::arg("cu_seqlens"), py::arg("chunk_indices"), py::arg("chunk_size"));
    m.def("chunk_kda_bwd_intra", &chunk_kda_bwd_intra, "KDA backward intra-chunk dq/dk/db/dg (CUDA)",
        py::arg("q"), py::arg("k"), py::arg("g"), py::arg("beta"),
        py::arg("dAqk"), py::arg("dAkk"),
        py::arg("dq"), py::arg("dk"), py::arg("db"), py::arg("dg"),
        py::arg("safe_gate"), py::arg("cu_seqlens") = py::none(),
        py::arg("chunk_indices") = py::none(), py::arg("chunk_size") = 64);
}
