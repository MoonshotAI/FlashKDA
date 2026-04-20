#include "fwd.h"
#include "fwd_kernel1.cuh"
#include "fwd_kernel2.cuh"

// ==================== launch_fwd ====================
template <int D, bool HasStateIn, bool HasStateOut, bool StateFP32, bool IsVarlen>
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
) {
    using BF16 = cutlass::bfloat16_t;
    constexpr int kInputStages = 3;
    constexpr int kOutputStages = 2;
    constexpr int CHUNK = 16;

    using K1L = K1Layouts<D, CHUNK>;
    using K2L = K2Layouts<D, CHUNK>;
    using WS = WorkspaceSizes<CHUNK, D>;

    // TMA layouts for Kernel 1
    using TMAQKLayout = typename K1L::TMAQKLayout;
    using TMAGLayout = typename K1L::TMAGLayout;
    using TMABetaSmemLayout = typename K1L::TMABetaSmemLayout;
    using TMAVOLayout = typename K1L::TMAVOLayout;
    using TMALMLayout = typename K1L::TMALMLayout;
    using TMAGTotalSmemLayout = typename K1L::TMAGTotalSmemLayout;

    // TMA layouts for Kernel 2
    using TMAStateSmemLayout = typename K2L::TMAStateSmemLayout;
    using TMAFP32StateSmemLayout = typename K2L::TMAFP32StateSmemLayout;

    // --- gmem layouts for original tensors
    auto gmem_layout = make_layout(make_shape(H, T_total, D), make_stride(D, D * H, 1));
    // 1D beta layout: [H*T] contiguous
    auto beta_gmem_layout = make_layout(make_shape(H * T_total));
    auto state_gmem_layout = make_layout(make_shape(N * H, D, D), LayoutRight{});

    Tensor m_q   = make_tensor(make_gmem_ptr(q_ptr), gmem_layout);
    Tensor m_k   = make_tensor(make_gmem_ptr(k_ptr), gmem_layout);
    Tensor m_v   = make_tensor(make_gmem_ptr(v_ptr), gmem_layout);
    Tensor m_out = make_tensor(make_gmem_ptr(out_ptr), gmem_layout);
    Tensor m_beta = make_tensor(make_gmem_ptr<BF16>(beta_ptr), beta_gmem_layout);

    // --- Workspace gmem layouts (separated arrays)
    int64_t n_ht = int64_t(H) * total_tiles;
    char* ws = reinterpret_cast<char*>(workspace_ptr);
    BF16*  ws_kd  = reinterpret_cast<BF16*>(ws);
    BF16*  ws_qd  = reinterpret_cast<BF16*>(ws + n_ht * WS::kKDecayed);
    BF16*  ws_kr  = reinterpret_cast<BF16*>(ws + n_ht * (WS::kKDecayed + WS::kQDecayed));
    float* ws_gt  = reinterpret_cast<float*>(ws + n_ht * (WS::kKDecayed + WS::kQDecayed + WS::kKRestored));
    BF16*  ws_inv = reinterpret_cast<BF16*>(ws + n_ht * (WS::kKDecayed + WS::kQDecayed + WS::kKRestored + WS::kGTotal));
    BF16*  ws_mqk = reinterpret_cast<BF16*>(ws + n_ht * (WS::kKDecayed + WS::kQDecayed + WS::kKRestored + WS::kGTotal + WS::kINV));

    auto ws_kd_gmem_layout = make_layout(make_shape(int(n_ht), CHUNK, D), LayoutRight{});
    auto ws_qd_gmem_layout = ws_kd_gmem_layout;
    auto ws_kr_gmem_layout = ws_kd_gmem_layout;
    auto ws_gt_gmem_layout = make_layout(make_shape(int(n_ht), D), LayoutRight{});
    auto ws_lm_gmem_layout = make_layout(make_shape(int(n_ht), CHUNK, CHUNK), LayoutRight{});

    Tensor m_ws_kd  = make_tensor(make_gmem_ptr(ws_kd), ws_kd_gmem_layout);
    Tensor m_ws_qd  = make_tensor(make_gmem_ptr(ws_qd), ws_qd_gmem_layout);
    Tensor m_ws_kr  = make_tensor(make_gmem_ptr(ws_kr), ws_kr_gmem_layout);
    Tensor m_ws_gt  = make_tensor(make_gmem_ptr(ws_gt), ws_gt_gmem_layout);
    Tensor m_ws_inv = make_tensor(make_gmem_ptr(ws_inv), ws_lm_gmem_layout);
    Tensor m_ws_mqk = make_tensor(make_gmem_ptr(ws_mqk), ws_lm_gmem_layout);

    // --- TMA descriptors for Kernel 1 (loads: q,k,beta; stores: workspace)
    auto tma_load_q    = make_tma_copy(SM90_TMA_LOAD{}, m_q, TMAQKLayout{});
    auto tma_load_k    = make_tma_copy(SM90_TMA_LOAD{}, m_k, TMAQKLayout{});
    auto tma_load_beta = make_tma_copy(SM90_TMA_LOAD{}, m_beta, TMABetaSmemLayout{});

    Tensor m_g = make_tensor(make_gmem_ptr(g_bf16_ptr), gmem_layout);
    auto tma_load_g = make_tma_copy(SM90_TMA_LOAD{}, m_g, TMAQKLayout{});

    auto dt_bias_gmem_layout = make_layout(make_shape(H, D), LayoutRight{});
    Tensor m_dt_bias = make_tensor(make_gmem_ptr(dt_bias_ptr), dt_bias_gmem_layout);
    auto tma_load_dt_bias = make_tma_copy(SM90_TMA_LOAD{}, m_dt_bias, TMAGTotalSmemLayout{});

    auto tma_store_ws_kd  = make_tma_copy(SM90_TMA_STORE{}, m_ws_kd, TMAVOLayout{});
    auto tma_store_ws_qd  = make_tma_copy(SM90_TMA_STORE{}, m_ws_qd, TMAVOLayout{});
    auto tma_store_ws_kr  = make_tma_copy(SM90_TMA_STORE{}, m_ws_kr, TMAVOLayout{});
    auto tma_store_ws_gt  = make_tma_copy(SM90_TMA_STORE{}, m_ws_gt, TMAGTotalSmemLayout{});
    auto tma_store_ws_inv = make_tma_copy(SM90_TMA_STORE{}, m_ws_inv, TMALMLayout{});
    auto tma_store_ws_mqk = make_tma_copy(SM90_TMA_STORE{}, m_ws_mqk, TMALMLayout{});

    // --- TMA descriptors for Kernel 2 (loads: v,beta,workspace; load/store: state,out)
    auto tma_load_v     = make_tma_copy(SM90_TMA_LOAD{}, m_v, TMAVOLayout{});
    auto tma_load_beta2 = make_tma_copy(SM90_TMA_LOAD{}, m_beta, TMABetaSmemLayout{});

    auto tma_load_ws_kd  = make_tma_copy(SM90_TMA_LOAD{}, m_ws_kd, TMAVOLayout{});
    auto tma_load_ws_qd  = make_tma_copy(SM90_TMA_LOAD{}, m_ws_qd, TMAVOLayout{});
    auto tma_load_ws_kr  = make_tma_copy(SM90_TMA_LOAD{}, m_ws_kr, TMAVOLayout{});
    auto tma_load_ws_gt  = make_tma_copy(SM90_TMA_LOAD{}, m_ws_gt, TMAGTotalSmemLayout{});
    auto tma_load_ws_inv = make_tma_copy(SM90_TMA_LOAD{}, m_ws_inv, TMALMLayout{});
    auto tma_load_ws_mqk = make_tma_copy(SM90_TMA_LOAD{}, m_ws_mqk, TMALMLayout{});

    auto tma_store_out = make_tma_copy(SM90_TMA_STORE{}, m_out, TMAVOLayout{});

    // --- State TMA descriptors (conditional on HasStateIn/HasStateOut and StateFP32)
    auto make_state_tma = [&]() {
        if constexpr (StateFP32) {
            // FP32 state TMA descriptors
            auto m_initial_fp32 = make_tensor(
                make_gmem_ptr(static_cast<float const*>(initial_state_ptr)), state_gmem_layout);
            auto m_final_fp32 = make_tensor(
                make_gmem_ptr(static_cast<float*>(final_state_ptr)), state_gmem_layout);
            auto tma_load = make_tma_copy(SM90_TMA_LOAD{}, m_initial_fp32, TMAFP32StateSmemLayout{});
            auto tma_store = make_tma_copy(SM90_TMA_STORE{}, m_final_fp32, TMAFP32StateSmemLayout{});
            return cute::make_tuple(tma_load, tma_store);
        } else {
            // BF16 state TMA descriptors (or dummy for no-state)
            auto state_ptr_load = HasStateIn
                ? static_cast<BF16 const*>(initial_state_ptr)
                : reinterpret_cast<BF16 const*>(out_ptr);  // dummy, never used
            auto state_ptr_store = HasStateOut
                ? static_cast<BF16*>(final_state_ptr)
                : reinterpret_cast<BF16*>(out_ptr);  // dummy, never used
            auto m_init = make_tensor(make_gmem_ptr(state_ptr_load), state_gmem_layout);
            auto m_final = make_tensor(make_gmem_ptr(state_ptr_store), state_gmem_layout);
            auto tma_load = make_tma_copy(SM90_TMA_LOAD{}, m_init, TMAStateSmemLayout{});
            auto tma_store = make_tma_copy(SM90_TMA_STORE{}, m_final, TMAStateSmemLayout{});
            return cute::make_tuple(tma_load, tma_store);
        }
    };
    auto [tma_load_initial_state, tma_store_final_state] = make_state_tma();

    // ===== Launch Kernel 1 (prepare) =====
#if BLOCK_LEVEL_K1 >= 0
    {
        constexpr int kK1Threads = 256;
        using SharedStorageK1T = SharedStorageK1<K1L>;
        int smem_size_k1 = sizeof(SharedStorageK1T);

        auto kernel1 = _flash_kda_fwd_prepare<
            decltype(tma_load_q), decltype(tma_load_k),
            decltype(tma_load_beta),
            decltype(tma_load_g), decltype(tma_load_dt_bias),
            decltype(tma_store_ws_kd), decltype(tma_store_ws_qd), decltype(tma_store_ws_kr),
            decltype(tma_store_ws_gt), decltype(tma_store_ws_inv), decltype(tma_store_ws_mqk),
            CHUNK, D, kK1Threads, IsVarlen
        >;

        cudaFuncSetAttribute(kernel1, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size_k1);

        dim3 grid_k1(total_tiles, H);
        dim3 block_k1(kK1Threads);

        kernel1<<<grid_k1, block_k1, smem_size_k1, stream>>>(
            tma_load_q, tma_load_k, tma_load_beta,
            tma_load_g, tma_load_dt_bias,
            tma_store_ws_kd, tma_store_ws_qd, tma_store_ws_kr,
            tma_store_ws_gt, tma_store_ws_inv, tma_store_ws_mqk,
            scale, T_total, H, N, cu_seqlens_ptr, total_tiles,
            A_log_ptr, gate_scale
        );
    }
#endif

    // ===== Launch Kernel 2 (recurrence) =====
#if BLOCK_LEVEL_K2 >= 0
    {
        constexpr int kK2Threads = 32 * 2 + 128;
        using SharedStorageK2T = SharedStorageK2<K2L, kInputStages, kOutputStages>;
        int smem_size_k2 = sizeof(SharedStorageK2T);

        auto kernel2 = _flash_kda_fwd_recurrence<
            decltype(tma_load_v), decltype(tma_load_beta2),
            decltype(tma_load_ws_kd), decltype(tma_load_ws_qd), decltype(tma_load_ws_kr),
            decltype(tma_load_ws_gt), decltype(tma_load_ws_inv), decltype(tma_load_ws_mqk),
            decltype(tma_load_initial_state),
            decltype(tma_store_final_state),
            decltype(tma_store_out),
            CHUNK, D, kInputStages, kOutputStages, kK2Threads,
            HasStateIn, HasStateOut, StateFP32, IsVarlen
        >;

        cudaFuncSetAttribute(kernel2, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size_k2);

        dim3 grid_k2(N, H);
        dim3 block_k2(kK2Threads);

        kernel2<<<grid_k2, block_k2, smem_size_k2, stream>>>(
            tma_load_v, tma_load_beta2,
            tma_load_ws_kd, tma_load_ws_qd, tma_load_ws_kr,
            tma_load_ws_gt, tma_load_ws_inv, tma_load_ws_mqk,
            tma_load_initial_state,
            tma_store_final_state,
            tma_store_out,
            out_ptr, T_total, H, N, cu_seqlens_ptr, total_tiles
        );
    }
#endif
}

// Explicit instantiations
#define INSTANTIATE_LAUNCH_FWD(D, HI, HO, FP32, VL) \
    template void launch_fwd<D, HI, HO, FP32, VL>( \
        cutlass::bfloat16_t const*, cutlass::bfloat16_t const*, \
        cutlass::bfloat16_t const*, cutlass::bfloat16_t const*, \
        cutlass::bfloat16_t const*, void const*, float, void*, \
        cutlass::bfloat16_t*, void*, int, int, int, int, \
        int64_t const*, float const*, float const*, float, cudaStream_t);

#define INSTANTIATE_STATE_VARIANTS(VL) \
    INSTANTIATE_LAUNCH_FWD(128, true,  true,  false, VL) \
    INSTANTIATE_LAUNCH_FWD(128, true,  true,  true,  VL) \
    INSTANTIATE_LAUNCH_FWD(128, false, false, false, VL) \
    INSTANTIATE_LAUNCH_FWD(128, false, true,  false, VL) \
    INSTANTIATE_LAUNCH_FWD(128, true,  false, false, VL) \
    INSTANTIATE_LAUNCH_FWD(128, false, true,  true,  VL) \
    INSTANTIATE_LAUNCH_FWD(128, true,  false, true,  VL)

INSTANTIATE_STATE_VARIANTS(true)   // varlen
INSTANTIATE_STATE_VARIANTS(false)  // non-varlen
