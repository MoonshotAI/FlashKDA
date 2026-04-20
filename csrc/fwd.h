#pragma once
#include <cuda_runtime.h>

#include <cutlass/bfloat16.h>

template <int D, bool HasStateIn = true, bool HasStateOut = true, bool StateFP32 = false, bool IsVarlen = true>
void launch_fwd(
    cutlass::bfloat16_t const* q_ptr,
    cutlass::bfloat16_t const* k_ptr,
    cutlass::bfloat16_t const* v_ptr,
    cutlass::bfloat16_t const* g_bf16_ptr,
    cutlass::bfloat16_t const* beta_ptr,
    void const* initial_state_ptr,
    float scale,
    void* final_state_ptr,
    cutlass::bfloat16_t* out_ptr,
    void* workspace_ptr,
    int total_tiles,
    int T_total,
    int H,
    int N,
    int64_t const* cu_seqlens_ptr,
    float const* A_log_ptr,
    float const* dt_bias_ptr,
    float gate_scale,
    cudaStream_t stream
);
