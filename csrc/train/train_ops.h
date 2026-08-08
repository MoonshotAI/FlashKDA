#pragma once

#include <optional>
#include <tuple>
#include <vector>
#include <torch/extension.h>

// KDA training-path operators, replicating fla/ops/kda Triton semantics.
// All varlen entry points take FLA-style `chunk_indices` (int64 [NT, 2] pairs
// of (sequence index, chunk index within sequence)) plus int64 cu_seqlens.

// fla/ops/kda/gate.py::kda_gate_chunk_cumsum.
// g: [B, T, H, S] (bf16/fp16/fp32), A_log: [H] fp32, dt_bias: [H*S] fp32 or nullopt.
// out: fp32 [B, T, H, S]. Gate activation + chunk-local inclusive cumsum along T.
void kda_gate_chunk_cumsum(
    torch::Tensor g,
    torch::Tensor A_log,
    std::optional<torch::Tensor> dt_bias,
    torch::Tensor out,
    double scale,
    bool has_scale,
    double lower_bound,
    bool use_lower_bound,
    int64_t chunk_size,
    std::optional<torch::Tensor> cu_seqlens,
    std::optional<torch::Tensor> chunk_indices
);

// fla/ops/utils/cumsum.py::chunk_local_cumsum (vector variant), no gate activation.
// g: [B, T, H, S], out: fp32 same shape. reverse=True computes suffix sums within each chunk.
void chunk_local_cumsum(
    torch::Tensor g,
    torch::Tensor out,
    double scale,
    bool has_scale,
    bool reverse,
    int64_t chunk_size,
    std::optional<torch::Tensor> cu_seqlens,
    std::optional<torch::Tensor> chunk_indices
);

// fla/ops/kda/gate.py::kda_gate_bwd.
// g: [B, T, H, D] raw (pre-bias) input, dyg: fp32 same shape (log2-domain grad).
// dg: fp32 same shape (output), dA_partial: fp32 [cdiv(B*T, 32), H] (output, sum over dim 0 by caller).
void kda_gate_bwd(
    torch::Tensor g,
    torch::Tensor A_log,
    std::optional<torch::Tensor> dt_bias,
    torch::Tensor dyg,
    torch::Tensor dg,
    torch::Tensor dA_partial,
    double lower_bound,
    bool use_lower_bound
);

// fla/ops/kda/chunk_intra.py forward kernels (intra.cu).
// Aqk: [B,T,HV,BT] same dtype as k; Akkd: [B,T,HV,16] fp32 diagonal blocks.
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

// Akk: [B,T,HV,BT] same dtype as k, zero-initialized by the caller.
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

// fla/ops/kda/chunk_intra.py::recompute_w_u_fwd (wy_fast.cu).
// Returns [w, u, qg (None when q is nullopt), kg].
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

// fla/ops/gla (gk variant) forward output (chunk_o.cu).
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

// fla/ops/common/chunk_delta_h.py (chunk_h.cu). Varlen via chunk_offsets + nt_total.
// Returns (h, v_new, final_state or nullopt).
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

// Returns (dh, dh0 or nullopt, dv).
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

// fla/ops/kda/chunk_bwd.py::chunk_kda_bwd_dAv (bwd_dav.cu).
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

// fla/ops/kda/chunk_intra.py::chunk_kda_bwd_intra (bwd_intra.cu).
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

// fla/ops/kda/chunk_bwd.py::chunk_kda_bwd_wy_dqkg_fused (bwd_wy_dqkg.cu).
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
