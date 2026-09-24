#include "fwd.h"

#include "fwd_kernel1.cuh"
#include "fwd_kernel2.cuh"

namespace flash_kda {
namespace sm80 {

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
    constexpr int kInputStages = kK2InputStages;
    constexpr int kOutputStages = kK2OutputStages;
    constexpr int CHUNK = 16;

    using K1L = K1Layouts<D, CHUNK>;
    using K2L = K2Layouts<D, CHUNK>;
    using WS = WorkspaceSizes<CHUNK, D>;

    // gmem layouts: PyTorch tensors are [B, H, T, D] row-major.  Inside a batch,
    // the effective layout is (H, T, D) with strides (D, H*D, 1), i.e. [T, H, D]
    // contiguous.
    auto gmem_layout = make_layout(make_shape(H, T_total, D), make_stride(D, H * D, 1));
    auto beta_gmem_layout = make_layout(make_shape(H * T_total));
    auto state_gmem_layout = make_layout(make_shape(N * H, D, D), LayoutRight{});
    auto dt_bias_gmem_layout = make_layout(make_shape(H, D), LayoutRight{});

    Tensor m_q   = make_tensor(make_gmem_ptr(q_ptr), gmem_layout);
    Tensor m_k   = make_tensor(make_gmem_ptr(k_ptr), gmem_layout);
    Tensor m_v   = make_tensor(make_gmem_ptr(v_ptr), gmem_layout);
    Tensor m_g   = make_tensor(make_gmem_ptr(g_bf16_ptr), gmem_layout);
    Tensor m_out = make_tensor(make_gmem_ptr(out_ptr), gmem_layout);
    Tensor m_beta = make_tensor(make_gmem_ptr<BF16>(beta_ptr), beta_gmem_layout);
    Tensor m_dt_bias = make_tensor(make_gmem_ptr(dt_bias_ptr), dt_bias_gmem_layout);

    // --- Workspace gmem layouts (separated arrays, one tile per head-tile)
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

    // State tensors (used conditionally by K2)
    auto make_state_tensors = [&]() {
        if constexpr (StateFP32) {
            auto m_init  = make_tensor(make_gmem_ptr(static_cast<float const*>(initial_state_ptr)), state_gmem_layout);
            auto m_final = make_tensor(make_gmem_ptr(static_cast<float*>(final_state_ptr)), state_gmem_layout);
            return cute::make_tuple(m_init, m_final);
        } else {
            BF16 const* load_ptr = HasStateIn
                ? static_cast<BF16 const*>(initial_state_ptr)
                : reinterpret_cast<BF16 const*>(out_ptr);  // dummy, never dereferenced
            BF16* store_ptr = HasStateOut
                ? static_cast<BF16*>(final_state_ptr)
                : reinterpret_cast<BF16*>(out_ptr);        // dummy, never dereferenced
            auto m_init  = make_tensor(make_gmem_ptr(load_ptr), state_gmem_layout);
            auto m_final = make_tensor(make_gmem_ptr(store_ptr), state_gmem_layout);
            return cute::make_tuple(m_init, m_final);
        }
    };
    auto [m_init_state, m_final_state] = make_state_tensors();

    // q/k/g are [T_total, H, D] row-major: distance between time rows is H*D,
    // head offset within a row is head_idx*D (computed inside K1).
    int q_row_stride = H * D;
    int k_row_stride = H * D;
    int g_row_stride = H * D;
    int ws_head_stride = static_cast<int>(WS::kKDecayed / sizeof(BF16));

    // ===== Launch Kernel 1 (prepare) =====
#if BLOCK_LEVEL_K1 >= 0
    {
        constexpr int kK1Threads = 256;
        using SharedStorageK1T = SharedStorageK1<K1L>;
        int smem_size_k1 = sizeof(SharedStorageK1T);

        auto kernel1 = _flash_kda_fwd_prepare_sm80<CHUNK, D, kK1Threads, IsVarlen>;

        cudaFuncSetAttribute(kernel1, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size_k1);

        dim3 grid_k1(total_tiles, H);
        dim3 block_k1(kK1Threads);

        kernel1<<<grid_k1, block_k1, smem_size_k1, stream>>>(
            q_ptr, q_row_stride,
            k_ptr, k_row_stride,
            g_bf16_ptr, g_row_stride,
            beta_ptr,
            dt_bias_ptr,
            ws_kd, ws_qd, ws_kr, ws_gt, ws_inv, ws_mqk,
            ws_head_stride,
            scale, T_total, H, N, cu_seqlens_ptr, total_tiles,
            A_log_ptr, gate_scale
        );
    }
#endif

    // ===== Launch Kernel 2 (recurrence) =====
#if BLOCK_LEVEL_K2 >= 0
    {
        constexpr int kK2Threads = 32 * 2 + 128;
        using SharedStorageK2T = SharedStorageK2<K2L, kInputStages, kOutputStages, StateFP32>;
        int smem_size_k2 = sizeof(SharedStorageK2T);

        auto kernel2 = _flash_kda_fwd_recurrence_sm80<
            decltype(m_v), decltype(m_beta),
            decltype(m_ws_kd), decltype(m_ws_qd), decltype(m_ws_kr),
            decltype(m_ws_gt), decltype(m_ws_inv), decltype(m_ws_mqk),
            decltype(m_init_state),
            decltype(m_final_state),
            decltype(m_out),
            CHUNK, D, kInputStages, kOutputStages, kK2Threads,
            HasStateIn, HasStateOut, StateFP32, IsVarlen
        >;

        cudaFuncSetAttribute(kernel2, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size_k2);

        dim3 grid_k2(N, H);
        dim3 block_k2(kK2Threads);

        kernel2<<<grid_k2, block_k2, smem_size_k2, stream>>>(
            m_v, m_beta,
            m_ws_kd, m_ws_qd, m_ws_kr,
            m_ws_gt, m_ws_inv, m_ws_mqk,
            m_init_state,
            m_final_state,
            m_out,
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

} // namespace sm80
} // namespace flash_kda
