#pragma once

// TMA_DISABLE_ALL: when defined, disable load/store warps entirely
// and let MMA warps work without pipeline synchronization
// #define TMA_DISABLE_ALL

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

    using TMABetaSmemLayout = BetaSmemLayout;  // 1D TMA, no dummy dim
    using TMAVOLayout = decltype(composition(
        VOLayout{}.layout_a(),
        VOLayout{}.offset(),
        prepend(VOLayout{}.layout_b())
    ));
    using TMAStateSmemLayout = decltype(composition(
        StateSmemLayout{}.layout_a(),
        StateSmemLayout{}.offset(),
        prepend(StateSmemLayout{}.layout_b())
    ));
    using TMALMLayout = decltype(composition(
        LMLayout{}.layout_a(),
        LMLayout{}.offset(),
        prepend(LMLayout{}.layout_b())
    ));
    using TMAGTotalSmemLayout = decltype(prepend(GTotalLayout{}));

    // FP32 state layout (K_SW32 atom, same 8x8 atom structure as K_INTER bf16)
    using FP32StateSmemLayout = decltype(tile_to_shape(
        GMMA::Layout_K_SW32_Atom<float>{},
        make_shape(Int<D>{}, Int<D>{}),
        LayoutLeft{}
    ));
    using TMAFP32StateSmemLayout = decltype(composition(
        FP32StateSmemLayout{}.layout_a(),
        FP32StateSmemLayout{}.offset(),
        prepend(FP32StateSmemLayout{}.layout_b())
    ));
};

template <class Layouts, int InputStages, int OutputStages>
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
    // FP32 state load/store happens before/after the pipeline loop, so no overlap.
    union {
        struct {
            InputStorage input[InputStages];
            OutputStorage output[OutputStages];
        };
        alignas(128) char state_fp32_buf[cute::cosize_v<StateSmemLayout> * sizeof(float)];
    };

    typename cutlass::PipelineTmaAsync<InputStages>::SharedStorage load_pipeline;
    typename cutlass::PipelineAsync<OutputStages>::SharedStorage store_pipeline;
    alignas(16) cutlass::arch::ClusterTransactionBarrier state_acc_tma_barrier;
};

// ==================== Kernel 2: Recurrence ====================
template <
    class TmaLoadV,
    class TmaLoadBeta,
    class TmaLoadWsKD, class TmaLoadWsQD, class TmaLoadWsKR,
    class TmaLoadWsGT, class TmaLoadWsINV, class TmaLoadWsMqk,
    class TmaLoadState,
    class TmaStoreState,
    class TmaStoreOut,
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
__global__ void __launch_bounds__(NumThreads) _flash_kda_fwd_recurrence(
    CUTE_GRID_CONSTANT TmaLoadV const tma_load_v,
    CUTE_GRID_CONSTANT TmaLoadBeta const tma_load_beta,
    CUTE_GRID_CONSTANT TmaLoadWsKD const tma_load_ws_kd,
    CUTE_GRID_CONSTANT TmaLoadWsQD const tma_load_ws_qd,
    CUTE_GRID_CONSTANT TmaLoadWsKR const tma_load_ws_kr,
    CUTE_GRID_CONSTANT TmaLoadWsGT const tma_load_ws_gt,
    CUTE_GRID_CONSTANT TmaLoadWsINV const tma_load_ws_inv,
    CUTE_GRID_CONSTANT TmaLoadWsMqk const tma_load_ws_mqk,
    CUTE_GRID_CONSTANT TmaLoadState const tma_load_initial_state,
    CUTE_GRID_CONSTANT TmaStoreState const tma_store_final_state,
    CUTE_GRID_CONSTANT TmaStoreOut const tma_store_out,
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
    using TMAVOLayout = typename Layouts::TMAVOLayout;
    using TMABetaSmemLayout = typename Layouts::TMABetaSmemLayout;
    using TMAStateSmemLayout = typename Layouts::TMAStateSmemLayout;
    using TMALMLayout = typename Layouts::TMALMLayout;
    using TMAGTotalSmemLayout = typename Layouts::TMAGTotalSmemLayout;
    constexpr int kWarpSize = 32;
    constexpr int kComputeThreads = 128;

    // Transaction bytes: v + beta + k_decayed + q_decayed + k_restored + g_total + INV + Mqk
    constexpr uint32_t kTmaTransactionBytes =
#ifndef TMA_DISABLE_ALL
        uint32_t(cute::cosize_v<VOLayout>) * uint32_t(sizeof(BF16)) +
        uint32_t(32) * uint32_t(sizeof(BF16)) +  // beta (bf16, sigmoid fused)
        uint32_t(cute::cosize_v<MMALayout>) * uint32_t(sizeof(BF16)) * 3 +  // k_decayed, q_decayed, k_restored
        uint32_t(cute::cosize_v<GTotalLayout>) * uint32_t(sizeof(float)) +    // g_total
        uint32_t(cute::cosize_v<LMLayout>) * uint32_t(sizeof(BF16)) * 2 +    // INV, Mqk
#endif
        0u;

    // --- shared memory
    extern __shared__ __align__(128) unsigned char shared_mem[];
    using SharedStorageT = SharedStorageK2<Layouts, InputStages, OutputStages>;
    SharedStorageT& shared_storage = *reinterpret_cast<SharedStorageT*>(shared_mem);

    // --- warp specialization
    int warp_id = threadIdx.x / kWarpSize;
    WarpRole warp_role = WarpRole::NonParticipant;
    if (warp_id < kComputeThreads / kWarpSize) {
        warp_role = WarpRole::MMA;
    } else if (warp_id < kComputeThreads / kWarpSize + 1) {
        warp_role = WarpRole::LOAD_QKG;
    } else if (warp_id < kComputeThreads / kWarpSize + 2) {
        warp_role = WarpRole::STORE;
    }

#ifndef TMA_DISABLE_ALL
    using LoadPipelineState = cutlass::PipelineState<InputStages>;
    using LoadPipeline = cutlass::PipelineTmaAsync<InputStages>;
    LoadPipeline load_pipeline = make_load_pipeline<InputStages>(
        shared_storage.load_pipeline,
        kTmaTransactionBytes,
        warp_role, 1, kComputeThreads
    );
    using StorePipelineState = cutlass::PipelineState<OutputStages>;
    using StorePipeline = cutlass::PipelineAsync<OutputStages>;
    StorePipeline store_pipeline = make_store_pipeline<OutputStages>(
        shared_storage.store_pipeline,
        warp_role, kComputeThreads, 1
    );
#endif

    // --- per-block sequence info
    int seq_idx  = blockIdx.x;
    int head_idx = blockIdx.y;
    int64_t bos, eos;
    int tile_base;

    if constexpr (IsVarlen) {
        bos = cu_seqlens[seq_idx];
        eos = cu_seqlens[seq_idx + 1];
        // Compute tile_base via linear scan (no host-precomputed table)
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
    bool lane_predicate = cute::elect_one_sync();

    // --- Load initial state
#ifndef TMA_DISABLE_ALL
    if constexpr (HasStateIn && !StateFP32) {
        // BF16 state: TMA load directly into state_acc
        if (warp_role == WarpRole::LOAD_QKG && lane_predicate) {
            using BarrierType = cutlass::arch::ClusterTransactionBarrier::ValueType;
            constexpr uint32_t kStateTransactionBytes = cute::cosize_v<StateSmemLayout> * sizeof(BF16);

            shared_storage.state_acc_tma_barrier.init(1);
            shared_storage.state_acc_tma_barrier.arrive_and_expect_tx(kStateTransactionBytes);

            Tensor g_init = tma_load_initial_state.get_tma_tensor(make_shape(N * H, D, D));
            auto init_off = g_init.layout()(seq_idx * H + head_idx, 0, 0);
            Tensor g_init_tile = make_tensor(g_init.data() + init_off,
                make_layout(make_shape(Int<1>{}, Int<D>{}, Int<D>{}), stride(g_init.layout())));
            Tensor s_state = make_tensor(make_smem_ptr(shared_storage.state_acc.begin()), TMAStateSmemLayout{});

            auto cta_tma_load_state = tma_load_initial_state.get_slice(Int<0>{});
            cute::copy(
                tma_load_initial_state.with(reinterpret_cast<BarrierType&>(shared_storage.state_acc_tma_barrier)),
                cta_tma_load_state.partition_S(g_init_tile),
                cta_tma_load_state.partition_D(s_state)
            );
        }
        __syncthreads();
        shared_storage.state_acc_tma_barrier.wait(0);
        cutlass::arch::fence_view_async_shared();
    } else if constexpr (HasStateIn && StateFP32) {
        // FP32 state: TMA load fp32 into pipeline buffer, then convert to bf16 in state_acc
        using FP32StateSmemLayout = typename Layouts::FP32StateSmemLayout;
        using TMAFP32StateSmemLayout = typename Layouts::TMAFP32StateSmemLayout;

        if (warp_role == WarpRole::LOAD_QKG && lane_predicate) {
            using BarrierType = cutlass::arch::ClusterTransactionBarrier::ValueType;
            constexpr uint32_t kFP32StateTransactionBytes = cute::cosize_v<StateSmemLayout> * sizeof(float);

            shared_storage.state_acc_tma_barrier.init(1);
            shared_storage.state_acc_tma_barrier.arrive_and_expect_tx(kFP32StateTransactionBytes);

            Tensor g_init = tma_load_initial_state.get_tma_tensor(make_shape(N * H, D, D));
            auto init_off = g_init.layout()(seq_idx * H + head_idx, 0, 0);
            Tensor g_init_tile = make_tensor(g_init.data() + init_off,
                make_layout(make_shape(Int<1>{}, Int<D>{}, Int<D>{}), stride(g_init.layout())));
            Tensor s_fp32 = make_tensor(
                make_smem_ptr(reinterpret_cast<float*>(shared_storage.state_fp32_buf)),
                TMAFP32StateSmemLayout{});

            auto cta_tma_load_state = tma_load_initial_state.get_slice(Int<0>{});
            cute::copy(
                tma_load_initial_state.with(reinterpret_cast<BarrierType&>(shared_storage.state_acc_tma_barrier)),
                cta_tma_load_state.partition_S(g_init_tile),
                cta_tma_load_state.partition_D(s_fp32)
            );
        }
        __syncthreads();
        shared_storage.state_acc_tma_barrier.wait(0);
        cutlass::arch::fence_view_async_shared();

        // All threads: convert fp32 -> bf16 with layout transformation
        smem_cvt_fp32_to_bf16<FP32StateSmemLayout, StateSmemLayout, D, NumThreads>(
            reinterpret_cast<float*>(shared_storage.state_fp32_buf),
            shared_storage.state_acc.begin(),
            threadIdx.x);
        __syncthreads();
    } else {
        // No state in: zero-initialize state_acc
        {
            BF16* buf = shared_storage.state_acc.begin();
            constexpr int kTotal = cute::cosize_v<StateSmemLayout>;
            for (int i = threadIdx.x; i < kTotal; i += NumThreads) {
                buf[i] = BF16(0);
            }
        }
        __syncthreads();
    }
#endif

#ifndef TMA_DISABLE_ALL
    __syncthreads();

    // --- LOAD warp: issue TMA loads for v, beta, and workspace intermediates
    if (warp_role == WarpRole::LOAD_QKG && lane_predicate) {
        Tensor g_v = tma_load_v.get_tma_tensor(make_shape(H, T_total, D));
        Tensor g_beta = tma_load_beta.get_tma_tensor(make_shape(H * T_total));

        // Workspace gmem tensors
        auto g_ws_kd = tma_load_ws_kd.get_tma_tensor(make_shape(H * total_tiles, CHUNK, D));
        auto g_ws_qd = tma_load_ws_qd.get_tma_tensor(make_shape(H * total_tiles, CHUNK, D));
        auto g_ws_kr = tma_load_ws_kr.get_tma_tensor(make_shape(H * total_tiles, CHUNK, D));
        auto g_ws_gt = tma_load_ws_gt.get_tma_tensor(make_shape(H * total_tiles, D));
        auto g_ws_inv = tma_load_ws_inv.get_tma_tensor(make_shape(H * total_tiles, CHUNK, CHUNK));
        auto g_ws_mqk = tma_load_ws_mqk.get_tma_tensor(make_shape(H * total_tiles, CHUNK, CHUNK));

        LoadPipelineState load_write = cutlass::make_producer_start_state<LoadPipeline>();
        auto cta_tma_load_v = tma_load_v.get_slice(Int<0>{});
        auto cta_tma_load_beta = tma_load_beta.get_slice(Int<0>{});
        auto cta_ws_kd = tma_load_ws_kd.get_slice(Int<0>{});
        auto cta_ws_qd = tma_load_ws_qd.get_slice(Int<0>{});
        auto cta_ws_kr = tma_load_ws_kr.get_slice(Int<0>{});
        auto cta_ws_gt = tma_load_ws_gt.get_slice(Int<0>{});
        auto cta_ws_inv = tma_load_ws_inv.get_slice(Int<0>{});
        auto cta_ws_mqk = tma_load_ws_mqk.get_slice(Int<0>{});

        for (int t = 0; t < t_tiles; ++t) {
            load_pipeline.producer_acquire(load_write);
            using LoadBarrierType = typename LoadPipeline::ProducerBarrierType;
            LoadBarrierType* tma_barrier = load_pipeline.producer_get_barrier(load_write);
            int stage = load_write.index();
            int ws_idx = head_idx * total_tiles + tile_base + t;

            // TMA load v
            auto v_off = g_v.layout()(head_idx, int(bos) + t * CHUNK, 0);
            Tensor g_v_tile = make_tensor(g_v.data() + v_off,
                make_layout(make_shape(Int<1>{}, Int<CHUNK>{}, Int<D>{}), stride(g_v.layout())));
            Tensor s_v_tile = make_tensor(make_smem_ptr(shared_storage.input[stage].v.begin()), TMAVOLayout{});
            cute::copy(tma_load_v.with(*tma_barrier),
                cta_tma_load_v.partition_S(g_v_tile), cta_tma_load_v.partition_D(s_v_tile));

            // TMA load beta (1D)
            int beta_linear = head_idx * T_total + (int(bos) + t * CHUNK);
            int beta_aligned = beta_linear & ~7;
            auto beta_off = g_beta.layout()(beta_aligned);
            Tensor g_beta_tile = make_tensor(g_beta.data() + beta_off, BetaSmemLayout{});
            Tensor s_beta_tile = make_tensor(make_smem_ptr(shared_storage.input[stage].beta.begin()), TMABetaSmemLayout{});
            cute::copy(tma_load_beta.with(*tma_barrier),
                cta_tma_load_beta.partition_S(g_beta_tile), cta_tma_load_beta.partition_D(s_beta_tile));

            // TMA load workspace: k_decayed
            {
                auto off = g_ws_kd.layout()(ws_idx, 0, 0);
                Tensor g_tile = make_tensor(g_ws_kd.data() + off,
                    make_layout(make_shape(Int<1>{}, Int<CHUNK>{}, Int<D>{}), stride(g_ws_kd.layout())));
                Tensor s_tile = make_tensor(make_smem_ptr(shared_storage.input[stage].k_decayed.begin()), TMAVOLayout{});
                cute::copy(tma_load_ws_kd.with(*tma_barrier), cta_ws_kd.partition_S(g_tile), cta_ws_kd.partition_D(s_tile));
            }
            // q_decayed
            {
                auto off = g_ws_qd.layout()(ws_idx, 0, 0);
                Tensor g_tile = make_tensor(g_ws_qd.data() + off,
                    make_layout(make_shape(Int<1>{}, Int<CHUNK>{}, Int<D>{}), stride(g_ws_qd.layout())));
                Tensor s_tile = make_tensor(make_smem_ptr(shared_storage.input[stage].q_decayed.begin()), TMAVOLayout{});
                cute::copy(tma_load_ws_qd.with(*tma_barrier), cta_ws_qd.partition_S(g_tile), cta_ws_qd.partition_D(s_tile));
            }
            // k_restored
            {
                auto off = g_ws_kr.layout()(ws_idx, 0, 0);
                Tensor g_tile = make_tensor(g_ws_kr.data() + off,
                    make_layout(make_shape(Int<1>{}, Int<CHUNK>{}, Int<D>{}), stride(g_ws_kr.layout())));
                Tensor s_tile = make_tensor(make_smem_ptr(shared_storage.input[stage].k_restored.begin()), TMAVOLayout{});
                cute::copy(tma_load_ws_kr.with(*tma_barrier), cta_ws_kr.partition_S(g_tile), cta_ws_kr.partition_D(s_tile));
            }
            // g_total
            {
                auto off = g_ws_gt.layout()(ws_idx, 0);
                Tensor g_tile = make_tensor(g_ws_gt.data() + off,
                    make_layout(make_shape(Int<1>{}, Int<D>{}), stride(g_ws_gt.layout())));
                Tensor s_tile = make_tensor(make_smem_ptr(shared_storage.input[stage].g_total.begin()), TMAGTotalSmemLayout{});
                cute::copy(tma_load_ws_gt.with(*tma_barrier), cta_ws_gt.partition_S(g_tile), cta_ws_gt.partition_D(s_tile));
            }
            // INV
            {
                auto off = g_ws_inv.layout()(ws_idx, 0, 0);
                Tensor g_tile = make_tensor(g_ws_inv.data() + off,
                    make_layout(make_shape(Int<1>{}, Int<CHUNK>{}, Int<CHUNK>{}), stride(g_ws_inv.layout())));
                Tensor s_tile = make_tensor(make_smem_ptr(shared_storage.input[stage].INV.begin()), TMALMLayout{});
                cute::copy(tma_load_ws_inv.with(*tma_barrier), cta_ws_inv.partition_S(g_tile), cta_ws_inv.partition_D(s_tile));
            }
            // Mqk
            {
                auto off = g_ws_mqk.layout()(ws_idx, 0, 0);
                Tensor g_tile = make_tensor(g_ws_mqk.data() + off,
                    make_layout(make_shape(Int<1>{}, Int<CHUNK>{}, Int<CHUNK>{}), stride(g_ws_mqk.layout())));
                Tensor s_tile = make_tensor(make_smem_ptr(shared_storage.input[stage].Mqk.begin()), TMALMLayout{});
                cute::copy(tma_load_ws_mqk.with(*tma_barrier), cta_ws_mqk.partition_S(g_tile), cta_ws_mqk.partition_D(s_tile));
            }

            ++load_write;
        }
        load_pipeline.producer_tail(load_write);
    }
#endif

    // --- MMA warps
    if (warp_role == WarpRole::MMA) {
        cutlass::arch::NamedBarrier compute_barrier(kComputeThreads, 0);
#ifndef TMA_DISABLE_ALL
        LoadPipelineState load_read;
        StorePipelineState out_write = cutlass::make_producer_start_state<StorePipeline>();
#endif
        int compute_tid = threadIdx.x;

        for (int t = 0; t < t_tiles; ++t) {
#ifndef TMA_DISABLE_ALL
            store_pipeline.producer_acquire(out_write);
            load_pipeline.consumer_wait(load_read);
            int load_stage = load_read.index();
            int out_stage = out_write.index();
#else
            constexpr int load_stage = 0;
            constexpr int out_stage = 0;
#endif

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
            // Each warp handles TWO 16x16 column blocks (N=128 / 4 warps = 32 = 2 x 16)
            // U stays in registers via SM75_U32x1_MOVM_T (no smem round-trip)
            {
            Tensor k_restored_t = make_tensor(make_smem_ptr(shared_storage.input[load_stage].k_restored.begin()), TransposedMMALayout{});

            constexpr int PREFETCH = 1;

            auto mma = make_tiled_mma(
                MMA_Atom<SM80_16x8x16_F32BF16BF16F32_TN>{},
                Layout<Shape<_1,_1>>{},
                Tile<_16,_16,_16>{}
            );

            const int warp_id = compute_tid / 32;
            const int lane_id = compute_tid % 32;
            const int group_id = (lane_id / 4) % 8;

            ThrMMA thr_mma = mma.get_slice(lane_id);

            // A copy: K_INTER → LDSM_N (for k_decayed, q_decayed, INV, Mqk)
            auto smem_tiled_copy_A = make_tiled_copy_A(Copy_Atom<SM75_U32x4_LDSM_N, BF16>{}, mma);
            auto smem_thr_copy_A   = smem_tiled_copy_A.get_thread_slice(lane_id);

            // A copy: MN_INTER → LDSM_T (for k_restored_t in Phase 7)
            auto smem_tiled_copy_A_T = make_tiled_copy_A(Copy_Atom<SM75_U16x8_LDSM_T, BF16>{}, mma);
            auto smem_thr_copy_A_T   = smem_tiled_copy_A_T.get_thread_slice(lane_id);

            // B copy: K_INTER → LDSM_N
            auto smem_tiled_copy_B = make_tiled_copy_B(Copy_Atom<SM75_U32x4_LDSM_N, BF16>{}, mma);
            auto smem_thr_copy_B   = smem_tiled_copy_B.get_thread_slice(lane_id);

            // C load/store
            auto smem_tiled_load_C  = make_tiled_copy_C(Copy_Atom<SM75_U32x4_LDSM_N, BF16>{}, mma);
            auto smem_thr_load_C    = smem_tiled_load_C.get_slice(lane_id);
            auto smem_tiled_store_C = make_tiled_copy_C(Copy_Atom<SM90_U32x4_STSM_N, BF16>{}, mma);
            auto smem_thr_store_C   = smem_tiled_store_C.get_slice(lane_id);

            // C load/store transposed (for Phase 6 state access via s_acc_T)
            auto smem_tiled_load_C_T  = make_tiled_copy_C(Copy_Atom<SM75_U16x8_LDSM_T, BF16>{}, mma);
            auto smem_thr_load_C_T    = smem_tiled_load_C_T.get_slice(lane_id);
            auto smem_tiled_store_C_T = make_tiled_copy_C(Copy_Atom<SM90_U16x8_STSM_T, BF16>{}, mma);
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
            // s_acc[D, D] = s_acc * g_total + k_restored_t[D, 16] @ U[16, D]
            // Each warp handles columns [warp_id*32, (warp_id+1)*32] = 2 x 16x16 blocks
            // U is already in tCrB_u_arr[0..1] as B operands (from Phase 4 MOVM_T)
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
            compute_barrier.arrive_and_wait();

#ifndef TMA_DISABLE_ALL
            cutlass::arch::fence_view_async_shared();
            store_pipeline.producer_commit(out_write);
            load_pipeline.consumer_release(load_read);
            ++load_read;
            ++out_write;
#endif
        }
    }

#ifndef TMA_DISABLE_ALL
    if (warp_role == WarpRole::STORE && lane_predicate) {
        Tensor g_out = tma_store_out.get_tma_tensor(make_shape(H, T_total, D));
        auto cta_tma_store = tma_store_out.get_slice(Int<0>{});
        StorePipelineState out_read;
        for (int t = 0; t < t_tiles; ++t) {
            store_pipeline.consumer_wait(out_read);
            int stage = out_read.index();
            int actual_len = min(CHUNK, seq_len - t * CHUNK);

            BF16* out_stage_ptr = shared_storage.output[stage].out.begin();

            if (actual_len < CHUNK) {
                // Manual store for tail tile to avoid overwriting next sequence
                // Only one thread (lane_predicate) runs here, so loop over all D
                Tensor s_out = make_tensor(make_smem_ptr(out_stage_ptr), VOLayout{});
                for (int row = 0; row < actual_len; ++row) {
                    int64_t global_base = (bos + t * CHUNK + row) * H * D + head_idx * D;
                    for (int col = 0; col < D; ++col) {
                        out_raw_ptr[global_base + col] = s_out(row, col);
                    }
                }
            } else {
                // TMA store for full tiles
                auto out_off = g_out.layout()(head_idx, int(bos) + t * CHUNK, 0);
                Tensor g_out_tile = make_tensor(g_out.data() + out_off,
                    make_layout(make_shape(Int<1>{}, Int<CHUNK>{}, Int<D>{}), stride(g_out.layout())));
                Tensor s_out_tile = make_tensor(make_smem_ptr(out_stage_ptr), TMAVOLayout{});
                cute::copy(
                    tma_store_out,
                    cta_tma_store.partition_S(s_out_tile),
                    cta_tma_store.partition_D(g_out_tile)
                );
                tma_store_arrive();
            }

            tma_store_wait<0>();
            store_pipeline.consumer_release(out_read);
            ++out_read;
        }

        if constexpr (HasStateOut && !StateFP32) {
            // BF16 state: TMA store directly from state_acc
            Tensor g_final = tma_store_final_state.get_tma_tensor(make_shape(N * H, D, D));
            auto state_off = g_final.layout()(seq_idx * H + head_idx, 0, 0);
            Tensor g_final_tile = make_tensor(g_final.data() + state_off,
                make_layout(make_shape(Int<1>{}, Int<D>{}, Int<D>{}), stride(g_final.layout())));
            Tensor s_state = make_tensor(make_smem_ptr(shared_storage.state_acc.begin()), TMAStateSmemLayout{});

            auto cta_tma_store_state = tma_store_final_state.get_slice(Int<0>{});
            cute::copy(
                tma_store_final_state,
                cta_tma_store_state.partition_S(s_state),
                cta_tma_store_state.partition_D(g_final_tile)
            );
            tma_store_arrive();
        }
    }

    if constexpr (HasStateOut && StateFP32) {
        // FP32 state: all threads sync, convert bf16->fp32, then STORE warp does TMA
        using FP32StateSmemLayout = typename Layouts::FP32StateSmemLayout;
        using TMAFP32StateSmemLayout = typename Layouts::TMAFP32StateSmemLayout;

        __syncthreads();  // all warps sync — pipeline smem now free

        smem_cvt_bf16_to_fp32<StateSmemLayout, FP32StateSmemLayout, D, NumThreads>(
            shared_storage.state_acc.begin(),
            reinterpret_cast<float*>(shared_storage.state_fp32_buf),
            threadIdx.x);
        __syncthreads();  // conversion complete

        if (warp_role == WarpRole::STORE && lane_predicate) {
            Tensor g_final = tma_store_final_state.get_tma_tensor(make_shape(N * H, D, D));
            auto state_off = g_final.layout()(seq_idx * H + head_idx, 0, 0);
            Tensor g_final_tile = make_tensor(g_final.data() + state_off,
                make_layout(make_shape(Int<1>{}, Int<D>{}, Int<D>{}), stride(g_final.layout())));
            Tensor s_fp32 = make_tensor(
                make_smem_ptr(reinterpret_cast<float*>(shared_storage.state_fp32_buf)),
                TMAFP32StateSmemLayout{});

            auto cta_tma_store_state = tma_store_final_state.get_slice(Int<0>{});
            cute::copy(
                tma_store_final_state,
                cta_tma_store_state.partition_S(s_fp32),
                cta_tma_store_state.partition_D(g_final_tile)
            );
            tma_store_arrive();
        }
    }

    __syncthreads();
#endif
}
