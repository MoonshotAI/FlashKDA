#pragma once

#include "utils.cuh"

template <int D, int CHUNK = 16>
struct K1Layouts {
    using QKLayout = decltype(make_layout(make_shape(Int<CHUNK>{}, Int<D>{}), LayoutRight{}));
    using GLayout = decltype(make_layout(make_shape(Int<CHUNK>{}, Int<D>{}), LayoutRight{}));
    using MMALayout = decltype(tile_to_shape(
        GMMA::Layout_K_INTER_Atom<cute::bfloat16_t>{},
        make_shape(Int<CHUNK>{}, Int<D>{}),
        LayoutLeft{}
    ));
    using BetaSmemLayout = Layout<Shape<Int<32>>, Stride<Int<1>>>;
    using GTotalLayout = Layout<Shape<Int<D>>, Stride<Int<1>>>;
    using LMLayout = decltype(tile_to_shape(
        GMMA::Layout_K_INTER_Atom<cute::bfloat16_t>{},
        make_shape(Int<CHUNK>{}, Int<CHUNK>{}),
        LayoutLeft{}
    ));
    using TransposedLMLayout = decltype(tile_to_shape(
        GMMA::Layout_MN_INTER_Atom<cute::bfloat16_t>{},
        make_shape(Int<CHUNK>{}, Int<CHUNK>{}),
        LayoutRight{}
    ));

    using TMABetaSmemLayout = BetaSmemLayout;  // 1D TMA, no dummy dim
    using TMAQKLayout = decltype(prepend(QKLayout{}));
    using TMAVOLayout = decltype(composition(
        MMALayout{}.layout_a(),
        MMALayout{}.offset(),
        prepend(MMALayout{}.layout_b())
    ));
    using TMAGLayout = decltype(prepend(GLayout{}));
    using TMALMLayout = decltype(composition(
        LMLayout{}.layout_a(),
        LMLayout{}.offset(),
        prepend(LMLayout{}.layout_b())
    ));
    using TMAGTotalSmemLayout = decltype(prepend(GTotalLayout{}));
};

template <class Layouts>
struct SharedStorageK1 {
    using BF16 = cutlass::bfloat16_t;
    using QKLayout = typename Layouts::QKLayout;
    using GLayout = typename Layouts::GLayout;
    using BetaSmemLayout = typename Layouts::BetaSmemLayout;
    using GTotalLayout = typename Layouts::GTotalLayout;
    using LMLayout = typename Layouts::LMLayout;
    using MMALayout = typename Layouts::MMALayout;

    // Phase A: q, k, g alive
    // Phase B: k_decayed, q_decayed, k_inv, L, INV, Mqk alive
    // These don't overlap → union saves ~14KB shared memory
    union {
        struct {
            alignas(128) cute::ArrayEngine<BF16, cute::cosize_v<QKLayout>> q;
            alignas(128) cute::ArrayEngine<BF16, cute::cosize_v<QKLayout>> k;
            alignas(128) cute::ArrayEngine<float, cute::cosize_v<GLayout>> g;
        };
        struct {
            alignas(128) cute::ArrayEngine<BF16, cute::cosize_v<MMALayout>> k_decayed;
            alignas(128) cute::ArrayEngine<BF16, cute::cosize_v<MMALayout>> q_decayed;
            alignas(128) cute::ArrayEngine<BF16, cute::cosize_v<MMALayout>> k_inv;
            alignas(128) cute::ArrayEngine<BF16, cute::cosize_v<LMLayout>> L;
            alignas(128) cute::ArrayEngine<BF16, cute::cosize_v<LMLayout>> INV;
            alignas(128) cute::ArrayEngine<BF16, cute::cosize_v<LMLayout>> Mqk;
        };
    };

    alignas(128) cute::ArrayEngine<BF16, cute::cosize_v<BetaSmemLayout>> beta;

    union {
        alignas(128) cute::ArrayEngine<BF16, cute::cosize_v<QKLayout>> g_bf16;      // TMA load target
        alignas(128) cute::ArrayEngine<BF16, cute::cosize_v<MMALayout>> k_restored;
    };
    union {
        alignas(128) cute::ArrayEngine<float, cute::cosize_v<GTotalLayout>> dt_bias;  // TMA load target
        alignas(128) cute::ArrayEngine<float, cute::cosize_v<GTotalLayout>> g_total;
    };
    alignas(16) cutlass::arch::ClusterTransactionBarrier tma_load_barrier;
};

// ==================== Kernel 1: Prepare ====================
template <
    class TmaLoadQ,
    class TmaLoadK,
    class TmaLoadBeta,
    class TmaLoadG,
    class TmaLoadDtBias,
    class TmaStoreWsKD, class TmaStoreWsQD, class TmaStoreWsKR,
    class TmaStoreWsGT, class TmaStoreWsINV, class TmaStoreWsMqk,
    int CHUNK,
    int D,
    int NumThreads,
    bool IsVarlen = true
>
__global__ void __launch_bounds__(NumThreads, 8) _flash_kda_fwd_prepare(
    CUTE_GRID_CONSTANT TmaLoadQ const tma_load_q,
    CUTE_GRID_CONSTANT TmaLoadK const tma_load_k,
    CUTE_GRID_CONSTANT TmaLoadBeta const tma_load_beta,
    CUTE_GRID_CONSTANT TmaLoadG const tma_load_g,
    CUTE_GRID_CONSTANT TmaLoadDtBias const tma_load_dt_bias,
    CUTE_GRID_CONSTANT TmaStoreWsKD const tma_store_ws_kd,
    CUTE_GRID_CONSTANT TmaStoreWsQD const tma_store_ws_qd,
    CUTE_GRID_CONSTANT TmaStoreWsKR const tma_store_ws_kr,
    CUTE_GRID_CONSTANT TmaStoreWsGT const tma_store_ws_gt,
    CUTE_GRID_CONSTANT TmaStoreWsINV const tma_store_ws_inv,
    CUTE_GRID_CONSTANT TmaStoreWsMqk const tma_store_ws_mqk,
    float scale,
    int T_total,
    int H,
    int N,
    int64_t const* cu_seqlens,
    int total_tiles,
    float const* A_log_ptr,
    float gate_scale
) {
    // --- constants
    using BF16 = cutlass::bfloat16_t;
    using FP16 = cutlass::half_t;
    using Layouts = K1Layouts<D, CHUNK>;
    using MMALayout = typename Layouts::MMALayout;
    using QKLayout = typename Layouts::QKLayout;
    using GLayout = typename Layouts::GLayout;
    using BetaSmemLayout = typename Layouts::BetaSmemLayout;
    using GTotalLayout = typename Layouts::GTotalLayout;
    using LMLayout = typename Layouts::LMLayout;
    using TransposedLMLayout = typename Layouts::TransposedLMLayout;
    using TMAQKLayout = typename Layouts::TMAQKLayout;
    using TMABetaSmemLayout = typename Layouts::TMABetaSmemLayout;
    using TMAVOLayout = typename Layouts::TMAVOLayout;
    using TMALMLayout = typename Layouts::TMALMLayout;
    using TMAGTotalSmemLayout = typename Layouts::TMAGTotalSmemLayout;
    constexpr uint32_t kTmaTransactionBytes =
        uint32_t(cute::cosize_v<QKLayout>) * uint32_t(3 * sizeof(BF16)) +  // q + k + g_bf16
        uint32_t(32) * uint32_t(sizeof(BF16)) +  // beta (bf16, sigmoid fused)
        uint32_t(D) * uint32_t(sizeof(float));  // dt_bias

    // --- shared memory
    extern __shared__ __align__(128) unsigned char shared_mem[];
    using SharedStorageT = SharedStorageK1<Layouts>;
    SharedStorageT& shared_storage = *reinterpret_cast<SharedStorageT*>(shared_mem);

    // --- per-CTA tile info
    int global_tile_idx = blockIdx.x;
    int head_idx = blockIdx.y;
    int seq_idx, tiles_before, local_t;
    int64_t bos, eos;
    int seq_len, t_tiles_this_seq;

    if constexpr (IsVarlen) {
        // Linear scan on cu_seqlens to find (seq_idx, local_t)
        seq_idx = 0;
        tiles_before = 0;
        for (int i = 0; i < N; i++) {
            int slen = int(cu_seqlens[i + 1] - cu_seqlens[i]);
            int n_tiles = (slen + CHUNK - 1) / CHUNK;
            if (tiles_before + n_tiles > global_tile_idx) {
                seq_idx = i;
                break;
            }
            tiles_before += n_tiles;
        }
        local_t = global_tile_idx - tiles_before;
        bos = cu_seqlens[seq_idx];
        eos = cu_seqlens[seq_idx + 1];
    } else {
        int T_seq = T_total / N;
        int tiles_per_seq = (T_seq + CHUNK - 1) / CHUNK;
        seq_idx = global_tile_idx / tiles_per_seq;
        tiles_before = seq_idx * tiles_per_seq;
        local_t = global_tile_idx - tiles_before;
        bos = seq_idx * T_seq;
        eos = bos + T_seq;
    }
    seq_len = int(eos - bos);
    t_tiles_this_seq = (seq_len + CHUNK - 1) / CHUNK;
    // Early exit for excess CTAs (total_tiles is an upper bound)
    if (local_t >= t_tiles_this_seq) return;
    // --- TMA load inputs (single-shot, no pipeline)
    // Only thread 0 issues TMA loads (not elect_one_sync which is per-warp)
    if (threadIdx.x == 0) {
        using BarrierType = cutlass::arch::ClusterTransactionBarrier::ValueType;
        shared_storage.tma_load_barrier.init(1);
        shared_storage.tma_load_barrier.arrive_and_expect_tx(kTmaTransactionBytes);

        Tensor g_q = tma_load_q.get_tma_tensor(make_shape(H, T_total, D));
        Tensor g_k = tma_load_k.get_tma_tensor(make_shape(H, T_total, D));
        Tensor g_beta = tma_load_beta.get_tma_tensor(make_shape(H * T_total));

        auto cta_tma_load_q = tma_load_q.get_slice(Int<0>{});
        auto cta_tma_load_k = tma_load_k.get_slice(Int<0>{});
        auto cta_tma_load_beta = tma_load_beta.get_slice(Int<0>{});

        auto qk_off = g_q.layout()(head_idx, int(bos) + local_t * CHUNK, 0);
        auto tile_shape_3d = make_shape(Int<1>{}, Int<CHUNK>{}, Int<D>{});
        auto tile_stride_3d = stride(g_q.layout());
        Tensor g_q_tile = make_tensor(g_q.data() + qk_off, make_layout(tile_shape_3d, tile_stride_3d));
        Tensor g_k_tile = make_tensor(g_k.data() + qk_off, make_layout(tile_shape_3d, tile_stride_3d));

        int beta_linear = head_idx * T_total + (int(bos) + local_t * CHUNK);
        int beta_aligned = beta_linear & ~7;
        auto beta_off = g_beta.layout()(beta_aligned);
        Tensor g_beta_tile = make_tensor(g_beta.data() + beta_off, BetaSmemLayout{});

        Tensor s_q_tile = make_tensor(make_smem_ptr(shared_storage.q.begin()), TMAQKLayout{});
        Tensor s_k_tile = make_tensor(make_smem_ptr(shared_storage.k.begin()), TMAQKLayout{});
        Tensor s_beta_tile = make_tensor(make_smem_ptr(shared_storage.beta.begin()), TMABetaSmemLayout{});

        cute::copy(tma_load_q.with(reinterpret_cast<BarrierType&>(shared_storage.tma_load_barrier)),
            cta_tma_load_q.partition_S(g_q_tile), cta_tma_load_q.partition_D(s_q_tile));
        cute::copy(tma_load_k.with(reinterpret_cast<BarrierType&>(shared_storage.tma_load_barrier)),
            cta_tma_load_k.partition_S(g_k_tile), cta_tma_load_k.partition_D(s_k_tile));
        cute::copy(tma_load_beta.with(reinterpret_cast<BarrierType&>(shared_storage.tma_load_barrier)),
            cta_tma_load_beta.partition_S(g_beta_tile), cta_tma_load_beta.partition_D(s_beta_tile));

        // TMA load g_bf16 (same gmem layout as q/k)
        Tensor g_g = tma_load_g.get_tma_tensor(make_shape(H, T_total, D));
        auto cta_tma_load_g = tma_load_g.get_slice(Int<0>{});
        Tensor g_g_tile = make_tensor(g_g.data() + qk_off, make_layout(tile_shape_3d, tile_stride_3d));
        Tensor s_g_bf16_tile = make_tensor(make_smem_ptr(shared_storage.g_bf16.begin()), TMAQKLayout{});
        cute::copy(tma_load_g.with(reinterpret_cast<BarrierType&>(shared_storage.tma_load_barrier)),
            cta_tma_load_g.partition_S(g_g_tile), cta_tma_load_g.partition_D(s_g_bf16_tile));

        // TMA load dt_bias [H, D] → [D] slice for current head
        Tensor g_dt = tma_load_dt_bias.get_tma_tensor(make_shape(H, D));
        auto cta_tma_load_dt = tma_load_dt_bias.get_slice(Int<0>{});
        auto dt_off = g_dt.layout()(head_idx, 0);
        Tensor g_dt_tile = make_tensor(g_dt.data() + dt_off,
            make_layout(make_shape(Int<1>{}, Int<D>{}), stride(g_dt.layout())));
        Tensor s_dt_tile = make_tensor(make_smem_ptr(shared_storage.dt_bias.begin()), TMAGTotalSmemLayout{});
        cute::copy(tma_load_dt_bias.with(reinterpret_cast<BarrierType&>(shared_storage.tma_load_barrier)),
            cta_tma_load_dt.partition_S(g_dt_tile), cta_tma_load_dt.partition_D(s_dt_tile));
    }

    // --- Compute a_log_exp (overlaps with TMA)
    float a_log_exp = expf(A_log_ptr[head_idx]);
    // --- Wait for TMA (q, k, beta, g_bf16, dt_bias)
    __syncthreads();
    shared_storage.tma_load_barrier.wait(0);
    cutlass::arch::fence_view_async_shared();
    __syncthreads();

    // --- QK L2 Normalization ---
    int compute_tid = threadIdx.x;
    {
        constexpr int ELEMS_PER_THREAD = 8;
        constexpr int THREADS_PER_ROW = D / ELEMS_PER_THREAD;  // 16
        int my_row = threadIdx.x / THREADS_PER_ROW;
        int my_col = (threadIdx.x % THREADS_PER_ROW) * ELEMS_PER_THREAD;

        BF16* q_smem = shared_storage.q.begin();
        BF16* k_smem = shared_storage.k.begin();

        float q_vals[ELEMS_PER_THREAD], k_vals[ELEMS_PER_THREAD];
        float q_sq = 0.0f, k_sq = 0.0f;

        #pragma unroll
        for (int i = 0; i < ELEMS_PER_THREAD; ++i) {
            float qv = bf16_to_f32(q_smem[my_row * D + my_col + i]);
            float kv = bf16_to_f32(k_smem[my_row * D + my_col + i]);
            q_vals[i] = qv;
            k_vals[i] = kv;
            q_sq += qv * qv;
            k_sq += kv * kv;
        }

        #pragma unroll
        for (int delta = 8; delta >= 1; delta >>= 1) {
            q_sq += __shfl_xor_sync(0xFFFFFFFF, q_sq, delta);
            k_sq += __shfl_xor_sync(0xFFFFFFFF, k_sq, delta);
        }

        float q_inv = rsqrtf(q_sq + 1e-6f);
        float k_inv = rsqrtf(k_sq + 1e-6f);

        #pragma unroll
        for (int i = 0; i < ELEMS_PER_THREAD; ++i) {
            q_smem[my_row * D + my_col + i] = BF16(q_vals[i] * q_inv);
            k_smem[my_row * D + my_col + i] = BF16(k_vals[i] * k_inv);
        }
    }
    __syncthreads();

    // --- Fused gate activation + cumsum + k tail zero-fill ---
    // Threads 0-127: gate(g_bf16 + dt_bias) → cumulative sum, eliminates raw-g smem round-trip
    // Threads 128-255: zero k for tail rows
    {
        int actual_len = min(CHUNK, seq_len - local_t * CHUNK);
        if (compute_tid < 128) {
            int col = compute_tid;
            BF16 const* g_bf16_smem = shared_storage.g_bf16.begin();
            float dt = shared_storage.dt_bias.begin()[col];
            float* g_smem = shared_storage.g.begin();
            float sum = 0.0f;
            #pragma unroll
            for (int row = 0; row < CHUNK; ++row) {
                float g_val;
                if (row < actual_len) {
                    g_val = bf16_to_f32(g_bf16_smem[row * D + col]) + dt;
                    g_val = a_log_exp * g_val;
                    g_val = gate_scale * sigmoid_tanh_approx_f32(g_val);
                } else {
                    g_val = 0.0f;
                }
                sum += g_val;
                g_smem[row * D + col] = sum;
            }
            shared_storage.g_total.begin()[col] = sum;
        } else {
            int col = compute_tid - 128;
            BF16* k_smem = shared_storage.k.begin();
            for (int row = actual_len; row < CHUNK; ++row) {
                k_smem[row * D + col] = BF16(0);
            }
        }
    }
    __syncthreads();

    Tensor q_tile = make_tensor(make_smem_ptr(shared_storage.q.begin()), QKLayout{});
    Tensor k_tile = make_tensor(make_smem_ptr(shared_storage.k.begin()), QKLayout{});
    Tensor g_tile = make_tensor(make_smem_ptr(shared_storage.g.begin()), GLayout{});
    Tensor beta_tile = make_tensor(make_smem_ptr(shared_storage.beta.begin()), BetaSmemLayout{});
    int beta_smem_offset = (head_idx * T_total + int(bos) + local_t * CHUNK) & 7;

    Tensor k_restored = make_tensor(make_smem_ptr(shared_storage.k_restored.begin()), MMALayout{});
    Tensor k_decayed = make_tensor(make_smem_ptr(shared_storage.k_decayed.begin()), MMALayout{});
    Tensor q_decayed = make_tensor(make_smem_ptr(shared_storage.q_decayed.begin()), MMALayout{});
    Tensor k_inv = make_tensor(make_smem_ptr(shared_storage.k_inv.begin()), MMALayout{});
    Tensor g_total = make_tensor(make_smem_ptr(shared_storage.g_total.begin()), GTotalLayout{});

    // exp_g_total: compute exp(g_total) in smem before decay_apply
    if (compute_tid < 128) {
        float x = g_total(compute_tid);
        g_total(compute_tid) = ex2_approx_ftz_f32(x);
    }
    __syncthreads();

// decay_apply
    if (compute_tid < 256) {
        static_assert(D % 64 == 0);
        static_assert(CHUNK % 8 == 0);

        int lane = compute_tid % 32;
        int warp_id = compute_tid / 32;
        int g = lane / 4;
        int t = lane % 4;

        auto vec8_2d = make_shape(_1{}, _8{});
        auto vec8_1d = make_shape(_8{});
        auto thr2_2d = make_shape(_1{}, _2{});
        auto thr2_1d = make_shape(_2{});

        constexpr int N_M = CHUNK / 8;
        constexpr int N_N = D / 64;
        constexpr int N_TILES = N_M * N_N;

        float reg_g[N_TILES][2];
        BF16  reg_q[N_TILES][2];
        BF16  reg_k[N_TILES][2];
        float reg_gt[N_TILES][2];

        #pragma unroll
        for (int m_blk = 0; m_blk < CHUNK; m_blk += 8) {
            #pragma unroll
            for (int n_blk = 0; n_blk < D; n_blk += 64) {
                int tile_idx = (m_blk / 8) * N_N + (n_blk / 64);
                int row = m_blk + ((warp_id + g) % 8);
                int col_base = n_blk + g * 8;
                int col_tile = col_base / 8;

                Tensor tile_g  = local_tile(g_tile, vec8_2d, make_coord(row, col_tile));
                Tensor tile_q  = local_tile(q_tile, vec8_2d, make_coord(row, col_tile));
                Tensor tile_k  = local_tile(k_tile, vec8_2d, make_coord(row, col_tile));
                Tensor tile_gt = local_tile(g_total, vec8_1d, make_coord(col_tile));

                Tensor s_g  = local_tile(tile_g,  thr2_2d, make_coord(0, t));
                Tensor s_q  = local_tile(tile_q,  thr2_2d, make_coord(0, t));
                Tensor s_k  = local_tile(tile_k,  thr2_2d, make_coord(0, t));
                Tensor s_gt = local_tile(tile_gt, thr2_1d, make_coord(t));

                Tensor r_g  = make_tensor_like<float>(s_g);
                Tensor r_q  = make_tensor_like<BF16>(s_q);
                Tensor r_k  = make_tensor_like<BF16>(s_k);
                Tensor r_gt = make_tensor_like<float>(s_gt);

                cute::copy(AutoVectorizingCopy{}, s_g, r_g);
                cute::copy(AutoVectorizingCopy{}, s_q, r_q);
                cute::copy(AutoVectorizingCopy{}, s_k, r_k);
                cute::copy(AutoVectorizingCopy{}, s_gt, r_gt);

                #pragma unroll
                for (int v = 0; v < 2; ++v) {
                    reg_g[tile_idx][v]  = r_g(0, v);
                    reg_q[tile_idx][v]  = r_q(0, v);
                    reg_k[tile_idx][v]  = r_k(0, v);
                    reg_gt[tile_idx][v] = r_gt(v);
                }
            }
        }

        // Sync before writing to union'd smem (q/k/g → k_decayed/q_decayed/k_inv)
        // Safe: all 256 threads enter this if block (compute_tid < 256 always true)
        __syncthreads();

        #pragma unroll
        for (int m_blk = 0; m_blk < CHUNK; m_blk += 8) {
            #pragma unroll
            for (int n_blk = 0; n_blk < D; n_blk += 64) {
                int tile_idx = (m_blk / 8) * N_N + (n_blk / 64);
                int row = m_blk + ((warp_id + g) % 8);
                int col_base = n_blk + g * 8;
                int col_tile = col_base / 8;

                Tensor tile_qd = local_tile(q_decayed, vec8_2d, make_coord(row, col_tile));
                Tensor tile_kd = local_tile(k_decayed, vec8_2d, make_coord(row, col_tile));
                Tensor tile_kr = local_tile(k_restored, vec8_2d, make_coord(row, col_tile));
                Tensor tile_ki = local_tile(k_inv, vec8_2d, make_coord(row, col_tile));

                Tensor s_qd = local_tile(tile_qd, thr2_2d, make_coord(0, t));
                Tensor s_kd = local_tile(tile_kd, thr2_2d, make_coord(0, t));
                Tensor s_kr = local_tile(tile_kr, thr2_2d, make_coord(0, t));
                Tensor s_ki = local_tile(tile_ki, thr2_2d, make_coord(0, t));

                Tensor r_qd = make_tensor_like<BF16>(s_qd);
                Tensor r_kd = make_tensor_like<BF16>(s_kd);
                #pragma unroll
                for (int v = 0; v < 2; ++v) {
                    float g = reg_g[tile_idx][v];
                    BF16 q = reg_q[tile_idx][v];
                    BF16 k = reg_k[tile_idx][v];
                    BF16 exp_cumsum = BF16(ex2_approx_ftz_f32(g));
                    r_qd(0, v) = q * exp_cumsum * BF16(scale);
                    r_kd(0, v) = k * exp_cumsum;
                }
                cute::copy(AutoVectorizingCopy{}, r_qd, s_qd);
                cute::copy(AutoVectorizingCopy{}, r_kd, s_kd);

                Tensor r_ki = make_tensor_like<BF16>(s_ki);
                Tensor r_kr = make_tensor_like<BF16>(s_kr);
                #pragma unroll
                for (int v = 0; v < 2; ++v) {
                    float g = reg_g[tile_idx][v];
                    BF16 k = reg_k[tile_idx][v];
                    BF16 inv_cumsum = BF16(ex2_approx_ftz_f32(-g));
                    r_ki(0, v) = k * inv_cumsum;
                    r_kr(0, v) = k * inv_cumsum * BF16(reg_gt[tile_idx][v]);
                }
                cute::copy(AutoVectorizingCopy{}, r_ki, s_ki);
                cute::copy(AutoVectorizingCopy{}, r_kr, s_kr);
            }
        }

    }
    __syncthreads();

    Tensor L = make_tensor(make_smem_ptr(shared_storage.L.begin()), LMLayout{});
    Tensor Mqk = make_tensor(make_smem_ptr(shared_storage.Mqk.begin()), LMLayout{});
    Tensor L_fp16 = make_tensor(make_smem_ptr(reinterpret_cast<FP16*>(shared_storage.L.begin())), LMLayout{});

// L_Mqk
    if (compute_tid < 32) {
        mma_m16n16_bf16bf16fp16_1warp(k_decayed, k_inv, L_fp16, compute_tid);
    } else if (compute_tid >= 32 && compute_tid < 64) {
        mma_m16n16_bf16bf16bf16_1warp(q_decayed, k_inv, Mqk, compute_tid - 32);
    }
    __syncthreads();

    Tensor INV = make_tensor(make_smem_ptr(shared_storage.INV.begin()), LMLayout{});
    Tensor INV_fp16 = make_tensor(make_smem_ptr(reinterpret_cast<FP16*>(shared_storage.INV.begin())), LMLayout{});

// tril_IL + INV = I - L (merged, same thread same element)
    if (compute_tid < 256) {
        const int col_block_size = 8;
        int block_idx = compute_tid / (CHUNK * col_block_size);
        int i = (compute_tid / col_block_size) % CHUNK;
        int j = compute_tid % col_block_size + block_idx * col_block_size;
        if (i <= j) {
            L_fp16(i, j) = FP16::bitcast(0);
        } else {
            L_fp16(i, j) = L_fp16(i, j) * FP16(sigmoid_tanh_approx_f32(float(beta_tile(beta_smem_offset + i))));
        }
        if (i < j) {
            Mqk(i, j) = BF16::bitcast(0);
        }
        // INV = I - L (same thread reads L(i,j) it just wrote)
        FP16 x = L_fp16(i, j);
        INV_fp16(i, j) = (i == j ? FP16(1.0f) - x : -x);
    }
    __syncthreads();

// inv (Neumann series, fused in registers)
    neumann_inv_fused_1warp(L_fp16, INV_fp16, INV, compute_tid);
    // Fence + sync combined: completion + TMA visibility
    cutlass::arch::fence_view_async_shared();
    __syncthreads();
    if (threadIdx.x == 0) {
        int ws_idx = head_idx * total_tiles + global_tile_idx;
        // Store k_decayed [CHUNK, D] bf16
        {
            auto g_ws = tma_store_ws_kd.get_tma_tensor(make_shape(H * total_tiles, CHUNK, D));
            auto ws_off = g_ws.layout()(ws_idx, 0, 0);
            Tensor g_ws_tile = make_tensor(g_ws.data() + ws_off,
                make_layout(make_shape(Int<1>{}, Int<CHUNK>{}, Int<D>{}), stride(g_ws.layout())));
            Tensor s_kd = make_tensor(make_smem_ptr(shared_storage.k_decayed.begin()), TMAVOLayout{});
            auto cta_tma = tma_store_ws_kd.get_slice(Int<0>{});
            cute::copy(tma_store_ws_kd, cta_tma.partition_S(s_kd), cta_tma.partition_D(g_ws_tile));
            tma_store_arrive();
        }
        // Store q_decayed
        {
            auto g_ws = tma_store_ws_qd.get_tma_tensor(make_shape(H * total_tiles, CHUNK, D));
            auto ws_off = g_ws.layout()(ws_idx, 0, 0);
            Tensor g_ws_tile = make_tensor(g_ws.data() + ws_off,
                make_layout(make_shape(Int<1>{}, Int<CHUNK>{}, Int<D>{}), stride(g_ws.layout())));
            Tensor s_qd = make_tensor(make_smem_ptr(shared_storage.q_decayed.begin()), TMAVOLayout{});
            auto cta_tma = tma_store_ws_qd.get_slice(Int<0>{});
            cute::copy(tma_store_ws_qd, cta_tma.partition_S(s_qd), cta_tma.partition_D(g_ws_tile));
            tma_store_arrive();
        }
        // Store k_restored
        {
            auto g_ws = tma_store_ws_kr.get_tma_tensor(make_shape(H * total_tiles, CHUNK, D));
            auto ws_off = g_ws.layout()(ws_idx, 0, 0);
            Tensor g_ws_tile = make_tensor(g_ws.data() + ws_off,
                make_layout(make_shape(Int<1>{}, Int<CHUNK>{}, Int<D>{}), stride(g_ws.layout())));
            Tensor s_kr = make_tensor(make_smem_ptr(shared_storage.k_restored.begin()), TMAVOLayout{});
            auto cta_tma = tma_store_ws_kr.get_slice(Int<0>{});
            cute::copy(tma_store_ws_kr, cta_tma.partition_S(s_kr), cta_tma.partition_D(g_ws_tile));
            tma_store_arrive();
        }
        // Store g_total [D] float
        {
            auto g_ws = tma_store_ws_gt.get_tma_tensor(make_shape(H * total_tiles, D));
            auto ws_off = g_ws.layout()(ws_idx, 0);
            Tensor g_ws_tile = make_tensor(g_ws.data() + ws_off,
                make_layout(make_shape(Int<1>{}, Int<D>{}), stride(g_ws.layout())));
            Tensor s_gt = make_tensor(make_smem_ptr(shared_storage.g_total.begin()), TMAGTotalSmemLayout{});
            auto cta_tma = tma_store_ws_gt.get_slice(Int<0>{});
            cute::copy(tma_store_ws_gt, cta_tma.partition_S(s_gt), cta_tma.partition_D(g_ws_tile));
            tma_store_arrive();
        }
        // Store INV [CHUNK, CHUNK] bf16
        {
            auto g_ws = tma_store_ws_inv.get_tma_tensor(make_shape(H * total_tiles, CHUNK, CHUNK));
            auto ws_off = g_ws.layout()(ws_idx, 0, 0);
            Tensor g_ws_tile = make_tensor(g_ws.data() + ws_off,
                make_layout(make_shape(Int<1>{}, Int<CHUNK>{}, Int<CHUNK>{}), stride(g_ws.layout())));
            Tensor s_inv = make_tensor(make_smem_ptr(shared_storage.INV.begin()), TMALMLayout{});
            auto cta_tma = tma_store_ws_inv.get_slice(Int<0>{});
            cute::copy(tma_store_ws_inv, cta_tma.partition_S(s_inv), cta_tma.partition_D(g_ws_tile));
            tma_store_arrive();
        }
        // Store Mqk [CHUNK, CHUNK] bf16
        {
            auto g_ws = tma_store_ws_mqk.get_tma_tensor(make_shape(H * total_tiles, CHUNK, CHUNK));
            auto ws_off = g_ws.layout()(ws_idx, 0, 0);
            Tensor g_ws_tile = make_tensor(g_ws.data() + ws_off,
                make_layout(make_shape(Int<1>{}, Int<CHUNK>{}, Int<CHUNK>{}), stride(g_ws.layout())));
            Tensor s_mqk = make_tensor(make_smem_ptr(shared_storage.Mqk.begin()), TMALMLayout{});
            auto cta_tma = tma_store_ws_mqk.get_slice(Int<0>{});
            cute::copy(tma_store_ws_mqk, cta_tma.partition_S(s_mqk), cta_tma.partition_D(g_ws_tile));
            tma_store_arrive();
        }
    }
    tma_store_wait<0>();
    __syncthreads();
}
