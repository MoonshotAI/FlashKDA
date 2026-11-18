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
        alignas(128) cute::ArrayEngine<BF16, cute::cosize_v<QKLayout>> g_bf16;
        alignas(128) cute::ArrayEngine<BF16, cute::cosize_v<MMALayout>> k_restored;
    };
    union {
        alignas(128) cute::ArrayEngine<float, cute::cosize_v<GTotalLayout>> dt_bias;
        alignas(128) cute::ArrayEngine<float, cute::cosize_v<GTotalLayout>> g_total;
    };
};

// ==================== Kernel 1: Prepare (SM80 serial version) ====================
template <
    int CHUNK,
    int D,
    int NumThreads,
    bool IsVarlen = true
>
__global__ void __launch_bounds__(NumThreads, 4) _flash_kda_fwd_prepare_sm80(
    cutlass::bfloat16_t const* __restrict__ q_ptr,
    int q_row_stride,
    cutlass::bfloat16_t const* __restrict__ k_ptr,
    int k_row_stride,
    cutlass::bfloat16_t const* __restrict__ g_ptr,
    int g_row_stride,
    cutlass::bfloat16_t const* __restrict__ beta_ptr,
    float const* __restrict__ dt_bias_ptr,
    cutlass::bfloat16_t* __restrict__ ws_kd_ptr,
    cutlass::bfloat16_t* __restrict__ ws_qd_ptr,
    cutlass::bfloat16_t* __restrict__ ws_kr_ptr,
    float*       __restrict__ ws_gt_ptr,
    cutlass::bfloat16_t* __restrict__ ws_inv_ptr,
    cutlass::bfloat16_t* __restrict__ ws_mqk_ptr,
    int ws_head_stride,
    float scale,
    int T_total,
    int H,
    int N,
    int64_t const* cu_seqlens,
    int total_tiles,
    float const* A_log_ptr,
    float gate_scale
) {
    using BF16 = cutlass::bfloat16_t;
    using FP16 = cutlass::half_t;
    using Layouts = K1Layouts<D, CHUNK>;
    using MMALayout = typename Layouts::MMALayout;
    using QKLayout = typename Layouts::QKLayout;
    using GLayout = typename Layouts::GLayout;
    using BetaSmemLayout = typename Layouts::BetaSmemLayout;
    using GTotalLayout = typename Layouts::GTotalLayout;
    using LMLayout = typename Layouts::LMLayout;

    extern __shared__ __align__(128) unsigned char shared_mem[];
    using SharedStorageT = SharedStorageK1<Layouts>;
    SharedStorageT& shared_storage = *reinterpret_cast<SharedStorageT*>(shared_mem);

    int global_tile_idx = blockIdx.x;
    int head_idx = blockIdx.y;
    int seq_idx, tiles_before, local_t;
    int64_t bos, eos;
    int seq_len, t_tiles_this_seq;

    if constexpr (IsVarlen) {
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
    if (local_t >= t_tiles_this_seq) return;

    int t_offset = int(bos) + local_t * CHUNK;
    int ws_idx = head_idx * total_tiles + global_tile_idx;
    int actual_len = min(CHUNK, seq_len - local_t * CHUNK);

    // --- Load inputs via cooperative copies (guard tail rows and beta bounds) ---
    // q/k/g are [T_total, H, D] row-major: element (t, h, d) at t*(H*D) + h*D + d.
    // The *_row_stride args are H*D (distance between consecutive time rows).
    auto g_q_tile = make_tensor(q_ptr + int64_t(t_offset) * q_row_stride + head_idx * D,
        make_layout(make_shape(actual_len, Int<D>{}), make_stride(q_row_stride, Int<1>{})));
    auto g_k_tile = make_tensor(k_ptr + int64_t(t_offset) * k_row_stride + head_idx * D,
        make_layout(make_shape(actual_len, Int<D>{}), make_stride(k_row_stride, Int<1>{})));
    auto g_g_tile = make_tensor(g_ptr + int64_t(t_offset) * g_row_stride + head_idx * D,
        make_layout(make_shape(actual_len, Int<D>{}), make_stride(g_row_stride, Int<1>{})));

    auto s_q_tile = make_tensor(make_smem_ptr(shared_storage.q.begin()), QKLayout{});
    auto s_k_tile = make_tensor(make_smem_ptr(shared_storage.k.begin()), QKLayout{});
    auto s_g_bf16_tile = make_tensor(make_smem_ptr(shared_storage.g_bf16.begin()), QKLayout{});
    auto s_dt_tile = make_tensor(make_smem_ptr(shared_storage.dt_bias.begin()), GTotalLayout{});

    coop_copy_2d_vec8<NumThreads>(g_q_tile, s_q_tile, threadIdx.x);
    coop_copy_2d_vec8<NumThreads>(g_k_tile, s_k_tile, threadIdx.x);
    coop_copy_2d_vec8<NumThreads>(g_g_tile, s_g_bf16_tile, threadIdx.x);

    // Zero-fill q/k/g tail rows so later MMAs see zeros in padded lanes.
    #pragma unroll
    for (int i = threadIdx.x + actual_len * D; i < CHUNK * D; i += NumThreads) {
        shared_storage.q.begin()[i] = BF16();
        shared_storage.k.begin()[i] = BF16();
        shared_storage.g_bf16.begin()[i] = BF16();
    }

    // dt_bias for current head
    auto g_dt_tile = make_tensor(dt_bias_ptr + head_idx * D,
        make_layout(make_shape(Int<D>{}), make_stride(Int<1>{})));
    coop_copy_1d<NumThreads>(g_dt_tile, s_dt_tile, threadIdx.x);

    // Beta: load up to 32 contiguous elements aligned to 8; zero the rest.
    int beta_len = H * T_total;
    int beta_linear = head_idx * T_total + t_offset;
    int beta_aligned = beta_linear & ~7;
    int beta_smem_offset = beta_linear & 7;
    int beta_load_len = min(32, beta_len - beta_aligned);
    auto g_beta_tile = make_tensor(beta_ptr + beta_aligned,
        make_layout(make_shape(beta_load_len), make_stride(Int<1>{})));
    auto s_beta_tile = make_tensor(make_smem_ptr(shared_storage.beta.begin()), BetaSmemLayout{});
    #pragma unroll
    for (int i = threadIdx.x; i < 32; i += NumThreads) {
        shared_storage.beta.begin()[i] = BF16();
    }
    coop_copy_1d<NumThreads>(g_beta_tile, s_beta_tile, threadIdx.x);

    __syncthreads();

    // --- Compute a_log_exp (same as original) ---
    float a_log_exp = expf(A_log_ptr[head_idx]);

    // --- QK L2 Normalization ---
    int compute_tid = threadIdx.x;
    {
        constexpr int ELEMS_PER_THREAD = 8;
        constexpr int THREADS_PER_ROW = D / ELEMS_PER_THREAD;
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

    Tensor k_restored = make_tensor(make_smem_ptr(shared_storage.k_restored.begin()), MMALayout{});
    Tensor k_decayed = make_tensor(make_smem_ptr(shared_storage.k_decayed.begin()), MMALayout{});
    Tensor q_decayed = make_tensor(make_smem_ptr(shared_storage.q_decayed.begin()), MMALayout{});
    Tensor k_inv = make_tensor(make_smem_ptr(shared_storage.k_inv.begin()), MMALayout{});
    Tensor g_total = make_tensor(make_smem_ptr(shared_storage.g_total.begin()), GTotalLayout{});

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

    // tril_IL + INV = I - L
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
        FP16 x = L_fp16(i, j);
        INV_fp16(i, j) = (i == j ? FP16(1.0f) - x : -x);
    }
    __syncthreads();

    // inv (Neumann series)
    neumann_inv_fused_1warp(L_fp16, INV_fp16, INV, compute_tid);
    __syncthreads();

    // --- Store outputs to gmem workspace via cooperative copies ---
    auto g_ws_kd_tile = make_tensor(ws_kd_ptr + ws_idx * ws_head_stride,
        make_layout(make_shape(Int<CHUNK>{}, Int<D>{}), make_stride(Int<D>{}, Int<1>{})));
    auto g_ws_qd_tile = make_tensor(ws_qd_ptr + ws_idx * ws_head_stride,
        make_layout(make_shape(Int<CHUNK>{}, Int<D>{}), make_stride(Int<D>{}, Int<1>{})));
    auto g_ws_kr_tile = make_tensor(ws_kr_ptr + ws_idx * ws_head_stride,
        make_layout(make_shape(Int<CHUNK>{}, Int<D>{}), make_stride(Int<D>{}, Int<1>{})));
    auto g_ws_gt_tile = make_tensor(ws_gt_ptr + ws_idx * D,
        make_layout(make_shape(Int<D>{}), make_stride(Int<1>{})));
    auto g_ws_inv_tile = make_tensor(ws_inv_ptr + ws_idx * (CHUNK * CHUNK),
        make_layout(make_shape(Int<CHUNK>{}, Int<CHUNK>{}), make_stride(Int<CHUNK>{}, Int<1>{})));
    auto g_ws_mqk_tile = make_tensor(ws_mqk_ptr + ws_idx * (CHUNK * CHUNK),
        make_layout(make_shape(Int<CHUNK>{}, Int<CHUNK>{}), make_stride(Int<CHUNK>{}, Int<1>{})));

    coop_copy_2d_vec8<NumThreads>(make_tensor(make_smem_ptr(shared_storage.k_decayed.begin()), MMALayout{}),
                                  g_ws_kd_tile, threadIdx.x);
    __syncthreads();
    coop_copy_2d_vec8<NumThreads>(make_tensor(make_smem_ptr(shared_storage.q_decayed.begin()), MMALayout{}),
                                  g_ws_qd_tile, threadIdx.x);
    __syncthreads();
    coop_copy_2d_vec8<NumThreads>(make_tensor(make_smem_ptr(shared_storage.k_restored.begin()), MMALayout{}),
                                  g_ws_kr_tile, threadIdx.x);
    __syncthreads();
    coop_copy_1d_vec4<NumThreads>(make_tensor(make_smem_ptr(shared_storage.g_total.begin()), GTotalLayout{}),
                                  g_ws_gt_tile, threadIdx.x);
    __syncthreads();
    coop_copy_2d_vec8<NumThreads>(make_tensor(make_smem_ptr(shared_storage.INV.begin()), LMLayout{}),
                                  g_ws_inv_tile, threadIdx.x);
    __syncthreads();
    coop_copy_2d_vec8<NumThreads>(make_tensor(make_smem_ptr(shared_storage.Mqk.begin()), LMLayout{}),
                                  g_ws_mqk_tile, threadIdx.x);
}
