#pragma once

// SM80 (A800) port: TMA + warp-specialized async pipeline replaced by synchronous
// cooperative gmem<->smem copies. The Phase 1-6 MMA recurrence is preserved verbatim.

#include "utils.cuh"

template <int D, int CHUNK = 16>
struct K2Layouts {
    using MMALayout = decltype(tile_to_shape(
        GMMA::Layout_K_INTER_Atom<cute::bfloat16_t>{},
        make_shape(Int<CHUNK>{}, Int<D>{}),
        LayoutLeft{}
    ));
    using TransposedMMALayout = decltype(tile_to_shape(
        GMMA::Layout_MN_INTER_Atom<cute::bfloat16_t>{},
        make_shape(Int<D>{}, Int<CHUNK>{}),
        LayoutRight{}
    ));
    using VOLayout = MMALayout;
    using TransposedVOLayout = TransposedMMALayout;
    using BetaSmemLayout = Layout<Shape<Int<32>>, Stride<Int<1>>>;
    using StateSmemLayout = decltype(tile_to_shape(
        GMMA::Layout_K_INTER_Atom<cute::bfloat16_t>{},
        make_shape(Int<D>{}, Int<D>{}),
        LayoutLeft{}
    ));
    using TransposedStateSmemLayout = decltype(tile_to_shape(
        GMMA::Layout_MN_INTER_Atom<cute::bfloat16_t>{},
        make_shape(Int<D>{}, Int<D>{}),
        LayoutRight{}
    ));
    using GTotalLayout = Layout<Shape<Int<D>>, Stride<Int<1>>>;
    using LMLayout = decltype(tile_to_shape(
        GMMA::Layout_K_INTER_Atom<cute::bfloat16_t>{},
        make_shape(Int<CHUNK>{}, Int<CHUNK>{}),
        LayoutLeft{}
    ));

    using FP32StateSmemLayout = decltype(tile_to_shape(
        GMMA::Layout_K_SW32_Atom<float>{},
        make_shape(Int<D>{}, Int<D>{}),
        LayoutLeft{}
    ));
};

// The kernel actually uses a 2-stage input pipeline (t & 1) and a single
// output buffer (out_stage = 0).  Keep the constants here so launch and kernel
// stay in sync.
constexpr int kK2InputStages = 2;
constexpr int kK2OutputStages = 1;

template <class Layouts, int InputStages, int OutputStages, bool StateFP32>
struct SharedStorageK2 {
    using BF16 = cutlass::bfloat16_t;
    using VOLayout = typename Layouts::VOLayout;
    using BetaSmemLayout = typename Layouts::BetaSmemLayout;
    using StateSmemLayout = typename Layouts::StateSmemLayout;
    using GTotalLayout = typename Layouts::GTotalLayout;
    using LMLayout = typename Layouts::LMLayout;
    using MMALayout = typename Layouts::MMALayout;

    alignas(128) cute::ArrayEngine<BF16, cute::cosize_v<StateSmemLayout>> state_acc;

    struct InputStorage {
        alignas(128) cute::ArrayEngine<BF16, cute::cosize_v<VOLayout>> v;
        alignas(128) cute::ArrayEngine<BF16, cute::cosize_v<BetaSmemLayout>> beta;
        alignas(128) cute::ArrayEngine<BF16, cute::cosize_v<MMALayout>> k_decayed;
        alignas(128) cute::ArrayEngine<BF16, cute::cosize_v<MMALayout>> q_decayed;
        alignas(128) cute::ArrayEngine<BF16, cute::cosize_v<MMALayout>> k_restored;
        alignas(128) cute::ArrayEngine<float, cute::cosize_v<GTotalLayout>> g_total;
        alignas(128) cute::ArrayEngine<BF16, cute::cosize_v<LMLayout>> INV;
        alignas(128) cute::ArrayEngine<BF16, cute::cosize_v<LMLayout>> Mqk;
    };

    struct OutputStorage {
        alignas(128) cute::ArrayEngine<BF16, cute::cosize_v<VOLayout>> out;
    };

    // Anonymous union: pipeline buffers share space with fp32 state conversion buffer.
    // The fp32 buffer is only needed when the state dtype is actually fp32; for
    // bf16 state it collapses to 1 byte so the union size is driven by the
    // pipeline buffers instead of the fp32 conversion scratch.
    union {
        struct {
            InputStorage input[InputStages];
            OutputStorage output[OutputStages];
        };
        alignas(128) char state_fp32_buf[StateFP32 ? cute::cosize_v<StateSmemLayout> * sizeof(float) : 1];
    };
};

// ==================== Kernel 2: Recurrence ====================
template <
    class GmemV,
    class GmemBeta,
    class GmemWsKD, class GmemWsQD, class GmemWsKR,
    class GmemWsGT, class GmemWsINV, class GmemWsMqk,
    class GmemStateLoad,
    class GmemStateStore,
    class GmemOut,
    int CHUNK,
    int D,
    int InputStages,
    int OutputStages,
    int NumThreads,
    bool HasStateIn = true,
    bool HasStateOut = true,
    bool StateFP32 = false,
    bool IsVarlen = true
>
__global__ void __launch_bounds__(NumThreads) _flash_kda_fwd_recurrence_sm80(
    CUTE_GRID_CONSTANT GmemV const m_v,
    CUTE_GRID_CONSTANT GmemBeta const m_beta,
    CUTE_GRID_CONSTANT GmemWsKD const m_ws_kd,
    CUTE_GRID_CONSTANT GmemWsQD const m_ws_qd,
    CUTE_GRID_CONSTANT GmemWsKR const m_ws_kr,
    CUTE_GRID_CONSTANT GmemWsGT const m_ws_gt,
    CUTE_GRID_CONSTANT GmemWsINV const m_ws_inv,
    CUTE_GRID_CONSTANT GmemWsMqk const m_ws_mqk,
    CUTE_GRID_CONSTANT GmemStateLoad const m_init_state,
    CUTE_GRID_CONSTANT GmemStateStore const m_final_state,
    CUTE_GRID_CONSTANT GmemOut const m_out,
    cutlass::bfloat16_t* out_raw_ptr,
    int T_total,
    int H,
    int N,
    int64_t const* cu_seqlens,
    int total_tiles
) {
    using BF16 = cutlass::bfloat16_t;
    using FP16 = cutlass::half_t;
    using Layouts = K2Layouts<D, CHUNK>;
    using MMALayout = typename Layouts::MMALayout;
    using TransposedMMALayout = typename Layouts::TransposedMMALayout;
    using VOLayout = typename Layouts::VOLayout;
    using TransposedVOLayout = typename Layouts::TransposedVOLayout;
    using BetaSmemLayout = typename Layouts::BetaSmemLayout;
    using StateSmemLayout = typename Layouts::StateSmemLayout;
    using TransposedStateSmemLayout = typename Layouts::TransposedStateSmemLayout;
    using GTotalLayout = typename Layouts::GTotalLayout;
    using LMLayout = typename Layouts::LMLayout;
    constexpr int kWarpSize = 32;
    constexpr int kComputeThreads = 128;
    constexpr int kLoadThreads = NumThreads;  // all threads cooperate on load/store

    // --- shared memory
    extern __shared__ __align__(128) unsigned char shared_mem[];
    using SharedStorageT = SharedStorageK2<Layouts, InputStages, OutputStages, StateFP32>;
    SharedStorageT& shared_storage = *reinterpret_cast<SharedStorageT*>(shared_mem);

    int tid = threadIdx.x;

    // --- warp role (only MMA warps run compute; all threads run load/store)
    int warp_id = threadIdx.x / kWarpSize;
    WarpRole warp_role = WarpRole::NonParticipant;
    if (warp_id < kComputeThreads / kWarpSize) {
        warp_role = WarpRole::MMA;
    }

    // --- per-block sequence info
    int seq_idx  = blockIdx.x;
    int head_idx = blockIdx.y;
    int64_t bos, eos;
    int tile_base;

    if constexpr (IsVarlen) {
        bos = cu_seqlens[seq_idx];
        eos = cu_seqlens[seq_idx + 1];
        tile_base = 0;
        for (int i = 0; i < seq_idx; i++) {
            tile_base += (int(cu_seqlens[i + 1] - cu_seqlens[i]) + CHUNK - 1) / CHUNK;
        }
    } else {
        int T_seq = T_total / N;
        bos = seq_idx * T_seq;
        eos = bos + T_seq;
        tile_base = seq_idx * ((T_seq + CHUNK - 1) / CHUNK);
    }
    int seq_len  = int(eos - bos);
    int t_tiles  = (seq_len + CHUNK - 1) / CHUNK;

    // --- Load initial state
    if constexpr (HasStateIn && !StateFP32) {
        auto off = m_init_state.layout()(seq_idx * H + head_idx, 0, 0);
        auto st = make_stride(cute::get<1>(stride(m_init_state.layout())),
                              cute::get<2>(stride(m_init_state.layout())));
        Tensor g_st = make_tensor(m_init_state.data() + off,
            make_layout(make_shape(Int<D>{}, Int<D>{}), st));
        Tensor s_state = make_tensor(make_smem_ptr(shared_storage.state_acc.begin()), StateSmemLayout{});
        coop_copy_2d_vec8<kLoadThreads>(g_st, s_state, tid);
        __syncthreads();
    } else if constexpr (HasStateIn && StateFP32) {
        using FP32StateSmemLayout = typename Layouts::FP32StateSmemLayout;
        auto off = m_init_state.layout()(seq_idx * H + head_idx, 0, 0);
        auto st = make_stride(cute::get<1>(stride(m_init_state.layout())),
                              cute::get<2>(stride(m_init_state.layout())));
        Tensor g_st = make_tensor(m_init_state.data() + off,
            make_layout(make_shape(Int<D>{}, Int<D>{}), st));
        Tensor s_fp32 = make_tensor(
            make_smem_ptr(reinterpret_cast<float*>(shared_storage.state_fp32_buf)),
            FP32StateSmemLayout{});
        coop_copy_2d<kLoadThreads>(g_st, s_fp32, tid);
        __syncthreads();
        smem_cvt_fp32_to_bf16<FP32StateSmemLayout, StateSmemLayout, D, NumThreads>(
            reinterpret_cast<float*>(shared_storage.state_fp32_buf),
            shared_storage.state_acc.begin(),
            threadIdx.x);
        __syncthreads();
    } else {
        BF16* buf = shared_storage.state_acc.begin();
        constexpr int kTotal = cute::cosize_v<StateSmemLayout>;
        for (int i = threadIdx.x; i < kTotal; i += NumThreads) {
            buf[i] = BF16(0);
        }
        __syncthreads();
    }

    // ===== Main recurrence loop: 2-stage cp.async pipeline =====
    // Loads for tile t+1 are issued (cp.async into the other smem buffer) before
    // computing tile t, so gmem latency overlaps with the MMA phases. Numerics
    // are identical to the synchronous version.
    auto issue_loads = [&](int t, int stage) {
        int ws_idx = head_idx * total_tiles + tile_base + t;
        // v [CHUNK, D] — tail rows (past seq end / T_total) are zero-filled via
        // cp.async src-size=0, so no OOB read is ever issued.
        {
            BF16 const* v_base = m_v.data().get() + m_v.layout()(head_idx, int(bos) + t * CHUNK, 0);
            int64_t v_row_stride = cute::get<1>(stride(m_v.layout()));
            Tensor s_tile = make_tensor(make_smem_ptr(shared_storage.input[stage].v.begin()), VOLayout{});
            int v_rows = min(CHUNK, seq_len - t * CHUNK);
            constexpr int NV = D / 8;
            for (int i = tid; i < CHUNK * NV; i += kLoadThreads) {
                int r = i / NV;
                int c = (i - r * NV) * 8;
                cp_async_16b_zfill(&s_tile(r, c), v_base + r * v_row_stride + c, r < v_rows);
            }
        }
        // beta (1D, 8-aligned, 32 elems) — bounds-guarded scalar load (tiny).
        {
            int beta_linear = head_idx * T_total + (int(bos) + t * CHUNK);
            int beta_aligned = beta_linear & ~7;
            BF16 const* beta_base = m_beta.data().get() + beta_aligned;
            BF16* s_beta = shared_storage.input[stage].beta.begin();
            int beta_rem = H * T_total - beta_aligned;  // valid elems from beta_aligned
            for (int i = tid; i < 32; i += kLoadThreads) {
                s_beta[i] = (i < beta_rem) ? beta_base[i] : BF16(0);
            }
        }
        // Workspace tiles are always full [CHUNK, D] / [CHUNK, CHUNK] / [D].
        auto cp_ws_tile = [&](BF16 const* ws_base, BF16* s_ptr, auto const& smem_layout, int rows, int cols) {
            Tensor s_tile = make_tensor(make_smem_ptr(s_ptr), smem_layout);
            int nv = cols / 8;
            for (int i = tid; i < rows * nv; i += kLoadThreads) {
                int r = i / nv;
                int c = (i - r * nv) * 8;
                cp_async_16b_zfill(&s_tile(r, c), ws_base + r * cols + c, true);
            }
        };
        cp_ws_tile(m_ws_kd.data().get() + m_ws_kd.layout()(ws_idx, 0, 0), shared_storage.input[stage].k_decayed.begin(), VOLayout{}, CHUNK, D);
        cp_ws_tile(m_ws_qd.data().get() + m_ws_qd.layout()(ws_idx, 0, 0), shared_storage.input[stage].q_decayed.begin(), VOLayout{}, CHUNK, D);
        cp_ws_tile(m_ws_kr.data().get() + m_ws_kr.layout()(ws_idx, 0, 0), shared_storage.input[stage].k_restored.begin(), VOLayout{}, CHUNK, D);
        cp_ws_tile(m_ws_inv.data().get() + m_ws_inv.layout()(ws_idx, 0, 0), shared_storage.input[stage].INV.begin(), LMLayout{}, CHUNK, CHUNK);
        cp_ws_tile(m_ws_mqk.data().get() + m_ws_mqk.layout()(ws_idx, 0, 0), shared_storage.input[stage].Mqk.begin(), LMLayout{}, CHUNK, CHUNK);
        // g_total (D floats)
        {
            float const* gt_base = m_ws_gt.data().get() + m_ws_gt.layout()(ws_idx, 0);
            Tensor s_tile = make_tensor(make_smem_ptr(shared_storage.input[stage].g_total.begin()), GTotalLayout{});
            for (int i = tid; i < D / 4; i += kLoadThreads) {
                cp_async_16b_zfill(&s_tile(i * 4), gt_base + i * 4, true);
            }
        }
    };

    if (t_tiles > 0) {
        issue_loads(0, 0);
        cute::cp_async_fence();
    }

    for (int t = 0; t < t_tiles; ++t) {
        const int stage = t & 1;

        // Prefetch tile t+1 into the other buffer, then wait for tile t's data.
        if (t + 1 < t_tiles) {
            issue_loads(t + 1, (t + 1) & 1);
        }
        cute::cp_async_fence();
        cute::cp_async_wait<1>();
        __syncthreads();

        // --- COMPUTE (MMA warps only)
        // NOTE: no NamedBarrier here. On SM80 this kernel synchronizes solely with
        // __syncthreads() (hardware barrier 0); a NamedBarrier(id 0) used by only
        // the 128 MMA threads would alias barrier 0 and corrupt the concurrent
        // full-CTA __syncthreads() of the other warps -> device trap.
        if (warp_role == WarpRole::MMA) {
            const int load_stage = stage;
            constexpr int out_stage = 0;

            Tensor v_tile = make_tensor(make_smem_ptr(shared_storage.input[load_stage].v.begin()), VOLayout{});
            Tensor beta_tile = make_tensor(make_smem_ptr(shared_storage.input[load_stage].beta.begin()), BetaSmemLayout{});
            int beta_smem_offset = (head_idx * T_total + int(bos) + t * CHUNK) & 7;
            Tensor out_tile = make_tensor(make_smem_ptr(shared_storage.output[out_stage].out.begin()), VOLayout{});

            Tensor k_decayed = make_tensor(make_smem_ptr(shared_storage.input[load_stage].k_decayed.begin()), MMALayout{});
            Tensor q_decayed = make_tensor(make_smem_ptr(shared_storage.input[load_stage].q_decayed.begin()), MMALayout{});
            Tensor k_restored = make_tensor(make_smem_ptr(shared_storage.input[load_stage].k_restored.begin()), MMALayout{});
            Tensor g_total = make_tensor(make_smem_ptr(shared_storage.input[load_stage].g_total.begin()), GTotalLayout{});
            Tensor INV = make_tensor(make_smem_ptr(shared_storage.input[load_stage].INV.begin()), LMLayout{});
            Tensor Mqk = make_tensor(make_smem_ptr(shared_storage.input[load_stage].Mqk.begin()), LMLayout{});

            Tensor s_acc = make_tensor(make_smem_ptr(shared_storage.state_acc.begin()), StateSmemLayout{});
            Tensor s_acc_T = make_tensor(make_smem_ptr(shared_storage.state_acc.begin()), TransposedStateSmemLayout{});

            // Fused MMA: v_sub, v_beta, U=INV@v, out=q@s, out+=Mqk@U, s_acc_update
            {
            Tensor k_restored_t = make_tensor(make_smem_ptr(shared_storage.input[load_stage].k_restored.begin()), TransposedMMALayout{});

            constexpr int PREFETCH = 1;

            auto mma = make_tiled_mma(
                MMA_Atom<SM80_16x8x16_F32BF16BF16F32_TN>{},
                Layout<Shape<_1,_1>>{},
                Tile<_16,_16,_16>{}
            );

            const int warp_id = threadIdx.x / 32;
            const int lane_id = threadIdx.x % 32;
            const int group_id = (lane_id / 4) % 8;

            ThrMMA thr_mma = mma.get_slice(lane_id);

            auto smem_tiled_copy_A = make_tiled_copy_A(Copy_Atom<SM75_U32x4_LDSM_N, BF16>{}, mma);
            auto smem_thr_copy_A   = smem_tiled_copy_A.get_thread_slice(lane_id);

            auto smem_tiled_copy_A_T = make_tiled_copy_A(Copy_Atom<SM75_U16x8_LDSM_T, BF16>{}, mma);
            auto smem_thr_copy_A_T   = smem_tiled_copy_A_T.get_thread_slice(lane_id);

            auto smem_tiled_copy_B = make_tiled_copy_B(Copy_Atom<SM75_U32x4_LDSM_N, BF16>{}, mma);
            auto smem_thr_copy_B   = smem_tiled_copy_B.get_thread_slice(lane_id);

            auto smem_tiled_load_C  = make_tiled_copy_C(Copy_Atom<SM75_U32x4_LDSM_N, BF16>{}, mma);
            auto smem_thr_load_C    = smem_tiled_load_C.get_slice(lane_id);
            auto smem_tiled_store_C = make_tiled_copy_C(Copy_Atom<AutoVectorizingCopy, BF16>{}, mma);
            auto smem_thr_store_C   = smem_tiled_store_C.get_slice(lane_id);

            auto smem_tiled_load_C_T  = make_tiled_copy_C(Copy_Atom<SM75_U16x8_LDSM_T, BF16>{}, mma);
            auto smem_thr_load_C_T    = smem_tiled_load_C_T.get_slice(lane_id);
            auto smem_tiled_store_C_T = make_tiled_copy_C(Copy_Atom<AutoVectorizingCopy, BF16>{}, mma);
            auto smem_thr_store_C_T   = smem_tiled_store_C_T.get_slice(lane_id);

            Tensor A_ref = local_tile(k_decayed, make_shape(Int<16>{}, Int<16>{}), make_coord(0, 0));
            Tensor B_ref = local_tile(s_acc, make_shape(Int<16>{}, Int<16>{}), make_coord(0, 0));
            Tensor C_ref = local_tile(v_tile, make_shape(Int<16>{}, Int<16>{}), make_coord(0, 0));

            Tensor tCrAi_k = make_fragment_like<BF16>(thr_mma.partition_fragment_A(A_ref));
            auto tCrAi_k_view = smem_thr_copy_A.retile_D(tCrAi_k);
            auto tCrA_k = thr_mma.partition_fragment_A(A_ref);

            Tensor tCrAi_q = make_fragment_like<BF16>(thr_mma.partition_fragment_A(A_ref));
            auto tCrAi_q_view = smem_thr_copy_A.retile_D(tCrAi_q);
            auto tCrA_q = thr_mma.partition_fragment_A(A_ref);

            Tensor tCrBi = make_fragment_like<BF16>(thr_mma.partition_fragment_B(B_ref));
            auto tCrBi_view = smem_thr_copy_B.retile_D(tCrBi);
            auto tCrB = thr_mma.partition_fragment_B(B_ref);

            auto tCrC_ref = thr_mma.partition_C(C_ref);

            using AccFragT = decltype(thr_mma.make_fragment_C(tCrC_ref));
            using SFragT = decltype(make_fragment_like<BF16>(thr_mma.make_fragment_C(tCrC_ref)));
            using AFragT = decltype(thr_mma.partition_fragment_A(A_ref));
            using BFragT_u = decltype(thr_mma.partition_fragment_B(B_ref));

            AccFragT u_acc[2], out_acc[2];
            #pragma unroll
            for (int i = 0; i < 2; ++i) { u_acc[i] = thr_mma.make_fragment_C(tCrC_ref); clear(u_acc[i]); }
            #pragma unroll
            for (int i = 0; i < 2; ++i) { out_acc[i] = thr_mma.make_fragment_C(tCrC_ref); clear(out_acc[i]); }

            // ======== Phase 1: Dual GEMM k@s and q@s (k-loop, 2 blocks per warp) ========
            constexpr int K_BLOCKS = decltype(cute::size<1>(k_decayed))::value / 16;

            copy(smem_tiled_copy_A, smem_thr_copy_A.partition_S(
                local_tile(k_decayed, make_shape(Int<16>{}, Int<16>{}), make_coord(0, 0))), tCrAi_k_view);
            copy(smem_tiled_copy_A, smem_thr_copy_A.partition_S(
                local_tile(q_decayed, make_shape(Int<16>{}, Int<16>{}), make_coord(0, 0))), tCrAi_q_view);
            copy(smem_tiled_copy_B, smem_thr_copy_B.partition_S(
                local_tile(s_acc, make_shape(Int<16>{}, Int<16>{}), make_coord(warp_id * 2, 0))), tCrBi_view);

            #pragma unroll
            for (int k = 0; k < K_BLOCKS; ++k) {
                cute::transform(tCrAi_k, tCrA_k, cute::identity{});
                cute::transform(tCrAi_q, tCrA_q, cute::identity{});
                cute::transform(tCrBi, tCrB, cute::identity{});

                copy(smem_tiled_copy_B, smem_thr_copy_B.partition_S(
                    local_tile(s_acc, make_shape(Int<16>{}, Int<16>{}), make_coord(warp_id * 2 + 1, k))), tCrBi_view);

                gemm(thr_mma, tCrA_k(_,_,Int<0>{}), tCrB(_,_,Int<0>{}), u_acc[0]);
                gemm(thr_mma, tCrA_q(_,_,Int<0>{}), tCrB(_,_,Int<0>{}), out_acc[0]);

                cute::transform(tCrBi, tCrB, cute::identity{});

                if (k + 1 < K_BLOCKS) {
                    copy(smem_tiled_copy_A, smem_thr_copy_A.partition_S(
                        local_tile(k_decayed, make_shape(Int<16>{}, Int<16>{}), make_coord(0, k + 1))), tCrAi_k_view);
                    copy(smem_tiled_copy_A, smem_thr_copy_A.partition_S(
                        local_tile(q_decayed, make_shape(Int<16>{}, Int<16>{}), make_coord(0, k + 1))), tCrAi_q_view);
                    copy(smem_tiled_copy_B, smem_thr_copy_B.partition_S(
                        local_tile(s_acc, make_shape(Int<16>{}, Int<16>{}), make_coord(warp_id * 2, k + 1))), tCrBi_view);
                }

                gemm(thr_mma, tCrA_k(_,_,Int<0>{}), tCrB(_,_,Int<0>{}), u_acc[1]);
                gemm(thr_mma, tCrA_q(_,_,Int<0>{}), tCrB(_,_,Int<0>{}), out_acc[1]);
            }

            // ======== Phase 2: Cast out (keep in regs), load v/INV/beta ========
            // MMA-warp barrier: orders every warp's Phase-1 s_acc LDSM reads
            // before any warp's Phase-6 s_acc stores within this tile
            // (compute-sanitizer racecheck flags the unordered pair).
            // NOTE: barrier id must be >= cutlass FirstUserBarrier (8) — on this
            // stack ids 1-7 alias cutlass-reserved barriers and trap the kernel,
            // and id 0 aliases __syncthreads (bar.sync 0 with a partial count
            // mixed with the full-CTA __syncthreads is UB and traps on SM80).
            asm volatile("bar.sync 8, 128;" ::: "memory");
            SFragT out_bf16[2];
            #pragma unroll
            for (int i = 0; i < 2; ++i)
                cute::transform(out_acc[i], out_bf16[i], [] __device__ (float x) { return BF16(x); });

            SFragT v_bf16[2];
            #pragma unroll
            for (int i = 0; i < 2; ++i) {
                Tensor v_block = local_tile(v_tile, make_shape(Int<16>{}, Int<16>{}), make_coord(0, warp_id * 2 + i));
                copy(smem_tiled_load_C, smem_thr_load_C.partition_S(v_block), smem_thr_load_C.retile_D(v_bf16[i]));
            }

            copy(smem_tiled_copy_A, smem_thr_copy_A.partition_S(INV), tCrAi_k_view);
            cute::transform(tCrAi_k, tCrA_k, cute::identity{});

            BF16 beta0 = BF16(sigmoid_tanh_approx_f32(float(beta_tile(beta_smem_offset + group_id))));
            BF16 beta1 = BF16(sigmoid_tanh_approx_f32(float(beta_tile(beta_smem_offset + group_id + 8))));

            // ======== Phase 3: u = (v - u) * beta; u = INV @ u (per block) ========
            SFragT u_bf16[2];
            uint32_t u_b_regs[4];

            #pragma unroll
            for (int i = 0; i < 2; ++i) {
                cute::transform(u_acc[i], u_bf16[i], [] __device__ (float x) { return BF16(x); });

                #pragma unroll
                for (int a = 0; a < 2; ++a) {
                    #pragma unroll
                    for (int d = 0; d < 2; ++d) {
                        auto c0 = make_coord(make_coord(a, 0), 0, d);
                        auto c1 = make_coord(make_coord(a, 1), 0, d);
                        u_bf16[i](c0) = (v_bf16[i](c0) - u_bf16[i](c0)) * beta0;
                        u_bf16[i](c1) = (v_bf16[i](c1) - u_bf16[i](c1)) * beta1;
                    }
                }

                uint32_t* u_c = reinterpret_cast<uint32_t*>(&u_bf16[i](0));
                SM75_U32x1_MOVM_T::copy(u_c[0], u_b_regs[0]);
                SM75_U32x1_MOVM_T::copy(u_c[1], u_b_regs[1]);
                SM75_U32x1_MOVM_T::copy(u_c[2], u_b_regs[2]);
                SM75_U32x1_MOVM_T::copy(u_c[3], u_b_regs[3]);

                auto tCrB_u_tmp = thr_mma.partition_fragment_B(B_ref);
                uint32_t* b_dst = reinterpret_cast<uint32_t*>(&tCrB_u_tmp(0));
                b_dst[0] = u_b_regs[0]; b_dst[1] = u_b_regs[1];
                b_dst[2] = u_b_regs[2]; b_dst[3] = u_b_regs[3];

                clear(u_acc[i]);
                gemm(thr_mma, tCrA_k(_,_,Int<0>{}), tCrB_u_tmp(_,_,Int<0>{}), u_acc[i]);

                cute::transform(u_acc[i], u_bf16[i], [] __device__ (float x) { return BF16(x); });
            }

            // ======== Phase 4: Load Mqk, MOVM_T → tCrB_u_arr, Mqk@U + add out ========
            copy(smem_tiled_copy_A, smem_thr_copy_A.partition_S(Mqk), tCrAi_k_view);
            cute::transform(tCrAi_k, tCrA_k, cute::identity{});

            BFragT_u tCrB_u_arr[2];

            #pragma unroll
            for (int i = 0; i < 2; ++i) {
                uint32_t* u_c = reinterpret_cast<uint32_t*>(&u_bf16[i](0));
                SM75_U32x1_MOVM_T::copy(u_c[0], u_b_regs[0]);
                SM75_U32x1_MOVM_T::copy(u_c[1], u_b_regs[1]);
                SM75_U32x1_MOVM_T::copy(u_c[2], u_b_regs[2]);
                SM75_U32x1_MOVM_T::copy(u_c[3], u_b_regs[3]);

                tCrB_u_arr[i] = thr_mma.partition_fragment_B(B_ref);
                uint32_t* b_dst = reinterpret_cast<uint32_t*>(&tCrB_u_arr[i](0));
                b_dst[0] = u_b_regs[0]; b_dst[1] = u_b_regs[1];
                b_dst[2] = u_b_regs[2]; b_dst[3] = u_b_regs[3];

                clear(out_acc[i]);
                gemm(thr_mma, tCrA_k(_,_,Int<0>{}), tCrB_u_arr[i](_,_,Int<0>{}), out_acc[i]);

                SFragT gemm_bf16;
                cute::transform(out_acc[i], gemm_bf16, [] __device__ (float x) { return BF16(x); });
                cute::transform(out_bf16[i], gemm_bf16, out_bf16[i], [] __device__ (BF16 c, BF16 a) { return c + a; });
            }

            // ======== Phase 5: Store final out ========
            #pragma unroll
            for (int i = 0; i < 2; ++i) {
                Tensor out_block = local_tile(out_tile, make_shape(Int<16>{}, Int<16>{}), make_coord(0, warp_id * 2 + i));
                copy(smem_tiled_store_C, smem_thr_store_C.retile_S(out_bf16[i]), smem_thr_store_C.partition_D(out_block));
            }

            // ======== Phase 6: s_acc update ========
            constexpr int S_M_BLOCKS = decltype(cute::size<0>(k_restored_t))::value / 16;

            Tensor tCrAi_kr = make_fragment_like<BF16>(thr_mma.partition_fragment_A(A_ref));
            auto tCrAi_kr_view = smem_thr_copy_A_T.retile_D(tCrAi_kr);

            AFragT ring_A_kr[PREFETCH];
            SFragT ring_S_acc[2][PREFETCH];
            float ring_g0[PREFETCH], ring_g1[PREFETCH];

            #pragma unroll
            for (int i = 0; i < PREFETCH; ++i) {
                Tensor kr_block = local_tile(k_restored_t, make_shape(Int<16>{}, Int<16>{}), make_coord(i, 0));
                copy(smem_tiled_copy_A_T, smem_thr_copy_A_T.partition_S(kr_block), tCrAi_kr_view);
                cute::transform(tCrAi_kr, ring_A_kr[i], cute::identity{});

                #pragma unroll
                for (int bi = 0; bi < 2; ++bi) {
                    Tensor s_block = local_tile(s_acc_T, make_shape(Int<16>{}, Int<16>{}), make_coord(i, warp_id * 2 + bi));
                    copy(smem_tiled_load_C_T, smem_thr_load_C_T.partition_S(s_block), smem_thr_load_C_T.retile_D(ring_S_acc[bi][i]));
                }

                ring_g0[i] = g_total(i * 16 + group_id);
                ring_g1[i] = g_total(i * 16 + group_id + 8);
            }

            #pragma unroll
            for (int m = 0; m < S_M_BLOCKS; ++m) {
                const int slot = m % PREFETCH;

                float g0 = ring_g0[slot];
                float g1 = ring_g1[slot];

                #pragma unroll
                for (int bi = 0; bi < 2; ++bi) {
                    clear(u_acc[bi]);
                    gemm(thr_mma, ring_A_kr[slot](_,_,Int<0>{}), tCrB_u_arr[bi](_,_,Int<0>{}), u_acc[bi]);
                }

                if (m + PREFETCH < S_M_BLOCKS) {
                    Tensor kr_next = local_tile(k_restored_t, make_shape(Int<16>{}, Int<16>{}), make_coord(m + PREFETCH, 0));
                    copy(smem_tiled_copy_A_T, smem_thr_copy_A_T.partition_S(kr_next), tCrAi_kr_view);
                    cute::transform(tCrAi_kr, ring_A_kr[slot], cute::identity{});

                    ring_g0[slot] = g_total((m + PREFETCH) * 16 + group_id);
                    ring_g1[slot] = g_total((m + PREFETCH) * 16 + group_id + 8);
                }

                #pragma unroll
                for (int bi = 0; bi < 2; ++bi) {
                    #pragma unroll
                    for (int a = 0; a < 2; ++a) {
                        #pragma unroll
                        for (int d = 0; d < 2; ++d) {
                            auto c0 = make_coord(make_coord(a, 0), 0, d);
                            auto c1 = make_coord(make_coord(a, 1), 0, d);
                            ring_S_acc[bi][slot](c0) = BF16(bf16_to_f32(ring_S_acc[bi][slot](c0)) * g0 + u_acc[bi](c0));
                            ring_S_acc[bi][slot](c1) = BF16(bf16_to_f32(ring_S_acc[bi][slot](c1)) * g1 + u_acc[bi](c1));
                        }
                    }

                    Tensor s_block = local_tile(s_acc_T, make_shape(Int<16>{}, Int<16>{}), make_coord(m, warp_id * 2 + bi));
                    copy(smem_tiled_store_C_T, smem_thr_store_C_T.retile_S(ring_S_acc[bi][slot]), smem_thr_store_C_T.partition_D(s_block));

                    if (m + PREFETCH < S_M_BLOCKS) {
                        Tensor s_next = local_tile(s_acc_T, make_shape(Int<16>{}, Int<16>{}), make_coord(m + PREFETCH, warp_id * 2 + bi));
                        copy(smem_tiled_load_C_T, smem_thr_load_C_T.partition_S(s_next), smem_thr_load_C_T.retile_D(ring_S_acc[bi][slot]));
                    }
                }
            }
            }
        }
        __syncthreads();

        // --- STORE tile t <- output[0] (all threads cooperate)
        {
            int actual_len = min(CHUNK, seq_len - t * CHUNK);
            Tensor s_out = make_tensor(make_smem_ptr(shared_storage.output[0].out.begin()), VOLayout{});

            if (actual_len < CHUNK) {
                // Tail: cooperative scalar store to avoid overwriting next sequence
                int tail_elems = actual_len * D;
                for (int i = tid; i < tail_elems; i += NumThreads) {
                    int row = i / D;
                    int col = i - row * D;
                    int64_t global_base = (bos + t * CHUNK + row) * H * D + head_idx * D;
                    out_raw_ptr[global_base + col] = s_out(row, col);
                }
            } else {
                auto out_off = m_out.layout()(head_idx, int(bos) + t * CHUNK, 0);
                auto st = make_stride(cute::get<1>(stride(m_out.layout())),
                                      cute::get<2>(stride(m_out.layout())));
                Tensor g_out_tile = make_tensor(m_out.data() + out_off,
                    make_layout(make_shape(Int<CHUNK>{}, Int<D>{}), st));
                coop_copy_2d_vec8<kLoadThreads>(s_out, g_out_tile, tid);
            }
        }
        __syncthreads();
    }

    // --- Store final state
    if constexpr (HasStateOut && !StateFP32) {
        Tensor s_state = make_tensor(make_smem_ptr(shared_storage.state_acc.begin()), StateSmemLayout{});
        auto off = m_final_state.layout()(seq_idx * H + head_idx, 0, 0);
        auto st = make_stride(cute::get<1>(stride(m_final_state.layout())),
                              cute::get<2>(stride(m_final_state.layout())));
        Tensor g_final = make_tensor(m_final_state.data() + off,
            make_layout(make_shape(Int<D>{}, Int<D>{}), st));
        coop_copy_2d_vec8<NumThreads>(s_state, g_final, tid);
        __syncthreads();
    } else if constexpr (HasStateOut && StateFP32) {
        using FP32StateSmemLayout = typename Layouts::FP32StateSmemLayout;
        __syncthreads();
        smem_cvt_bf16_to_fp32<StateSmemLayout, FP32StateSmemLayout, D, NumThreads>(
            shared_storage.state_acc.begin(),
            reinterpret_cast<float*>(shared_storage.state_fp32_buf),
            threadIdx.x);
        __syncthreads();
        Tensor s_fp32 = make_tensor(
            make_smem_ptr(reinterpret_cast<float*>(shared_storage.state_fp32_buf)),
            FP32StateSmemLayout{});
        auto off = m_final_state.layout()(seq_idx * H + head_idx, 0, 0);
        auto st = make_stride(cute::get<1>(stride(m_final_state.layout())),
                              cute::get<2>(stride(m_final_state.layout())));
        Tensor g_final = make_tensor(m_final_state.data() + off,
            make_layout(make_shape(Int<D>{}, Int<D>{}), st));
        coop_copy_2d<NumThreads>(s_fp32, g_final, tid);
        __syncthreads();
    }
}
