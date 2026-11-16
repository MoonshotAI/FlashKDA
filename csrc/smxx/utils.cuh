#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <cstdio>
#include <cassert>

#include <cute/tensor.hpp>
#include <cute/algorithm/cooperative_copy.hpp>
#include <cute/algorithm/cooperative_gemm.hpp>
#include <cute/arch/cluster_sm90.hpp>
#include <cute/arch/copy.hpp>
#include <cute/arch/copy_sm90_tma.hpp>
#include <cute/arch/mma_sm80.hpp>
#include <cute/pointer_flagged.hpp>
#include <cute/stride.hpp>
#include <cutlass/cluster_launch.hpp>
#include <cutlass/arch/barrier.h>
#include <cutlass/pipeline/sm90_pipeline.hpp>
#include <cutlass/bfloat16.h>
#include <cutlass/tfloat32.h>

#include "cute/arch/copy_sm75.hpp"
#include "cute/arch/copy_sm90.hpp"
#include "cute/layout.hpp"
#include "cute/numeric/integral_constant.hpp"
#include "cute/tensor_impl.hpp"

#ifndef BLOCK_LEVEL_K1
#define BLOCK_LEVEL_K1 1
#endif

#ifndef BLOCK_LEVEL_K2
#define BLOCK_LEVEL_K2 1
#endif

__device__ __forceinline__ float ex2_approx_ftz_f32(float x) {
    float result;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(result) : "f"(x));
    return result;
}

__device__ __forceinline__ float tanh_approx_f32(float x) {
    float result;
    asm("tanh.approx.f32 %0, %1;" : "=f"(result) : "f"(x));
    return result;
}

__device__ __forceinline__ float sigmoid_approx_f32(float x) {
    // Avoid cancellation in 0.5 * tanh(x / 2) + 0.5 for small gates.
    return 1.0f / (1.0f + ex2_approx_ftz_f32(-x * 1.4426950408889634f));
}

__device__ __forceinline__ float bf16_to_f32(cutlass::bfloat16_t x) {
    float result;
    asm("cvt.f32.bf16 %0, %1;\n" : "=f"(result) : "h"(x.storage));
    return result;
}

using namespace cute;

// Workspace per-tile byte sizes (all naturally 128-byte aligned)
template <int CHUNK, int D>
struct WorkspaceSizes {
    static_assert(CHUNK * D * 2 % 128 == 0);
    static_assert(D * 4 % 128 == 0);
    static_assert(CHUNK * CHUNK * 2 % 128 == 0);

    static constexpr int kKDecayed  = CHUNK * D * 2;        // 4096
    static constexpr int kQDecayed  = CHUNK * D * 2;        // 4096
    static constexpr int kKRestored = CHUNK * D * 2;        // 4096
    static constexpr int kGTotal    = D * 4;                 // 512
    static constexpr int kINV       = CHUNK * CHUNK * 2;     // 512
    static constexpr int kMqk       = CHUNK * CHUNK * 2;     // 512
    static constexpr int64_t kPerTile = kKDecayed + kQDecayed + kKRestored + kGTotal + kINV + kMqk;
};

enum class WarpRole {
    MMA,
    LOAD_QKG,
    STORE,
    NonParticipant,
};

template <int Stages>
CUTLASS_DEVICE
cutlass::PipelineTmaAsync<Stages> make_load_pipeline(
    typename cutlass::PipelineTmaAsync<Stages>::SharedStorage& storage,
    uint32_t transaction_bytes,
    WarpRole warp_role,
    uint32_t num_producers,
    uint32_t num_consumers
) {
    using Pipeline = cutlass::PipelineTmaAsync<Stages>;
    typename Pipeline::Params params;

    auto role = Pipeline::ThreadCategory::NonParticipant;
    bool is_leader = false;
    if (warp_role == WarpRole::LOAD_QKG) {
        role = Pipeline::ThreadCategory::Producer;
        is_leader = cute::elect_one_sync();
    } else if (warp_role == WarpRole::MMA) {
        role = Pipeline::ThreadCategory::Consumer;
    }

    params.transaction_bytes = transaction_bytes;
    params.role = role;
    params.is_leader = is_leader;
    params.num_consumers = num_consumers;
    params.num_producers = num_producers;

    Pipeline pipeline(storage, params, Shape<_1,_1>{});
    cutlass::pipeline_init_wait(1);
    return pipeline;
}

template <int Stages>
CUTLASS_DEVICE
cutlass::PipelineAsync<Stages> make_store_pipeline(
    typename cutlass::PipelineAsync<Stages>::SharedStorage& storage,
    WarpRole warp_role,
    uint32_t num_producers,
    uint32_t num_consumers
) {
    using Pipeline = cutlass::PipelineAsync<Stages>;
    typename Pipeline::Params params;

    auto role = Pipeline::ThreadCategory::NonParticipant;
    if (warp_role == WarpRole::MMA) {
        role = Pipeline::ThreadCategory::Producer;
    } else if (warp_role == WarpRole::STORE) {
        role = Pipeline::ThreadCategory::Consumer;
    }

    params.role = role;
    params.producer_arv_count = num_producers;
    params.consumer_arv_count = num_consumers;

    Pipeline pipeline(storage, params);
    cutlass::pipeline_init_wait(1);
    return pipeline;
}

template <class TensorA, class TensorB, class TensorC>
CUTLASS_DEVICE void mma_m16n16_bf16bf16bf16_1warp(
    TensorA const& A,
    TensorB const& B,
    TensorC& C,
    int mma_tid
) {
    auto mma = make_tiled_mma(
        SM80_16x8x16_F32BF16BF16F32_TN{},
        Layout<Shape<_1,_1>>{},
        Tile<_16,_16,_16>{}
    );

    if (mma_tid >= int(size(mma))) return;

    using BF16 = cutlass::bfloat16_t;

    auto sC_store_op = [] __device__ (float x) { return BF16(x); };

    cooperative_gemm(mma_tid, mma, 1.0f, A, B, 0.0f, C, cute::identity{}, cute::identity{}, cute::identity{}, sC_store_op, SM75_U32x4_LDSM_N{}, SM75_U32x4_LDSM_N{}, SM75_U32x4_LDSM_N{}, SM90_U32x4_STSM_N{});
}

template <class TensorA, class TensorB, class TensorC>
CUTLASS_DEVICE void mma_m16n16_bf16bf16fp32_1warp(
    TensorA const& A,
    TensorB const& B,
    TensorC& C,
    int mma_tid
) {
    auto mma = make_tiled_mma(
        SM80_16x8x16_F32BF16BF16F32_TN{},
        Layout<Shape<_1,_1>>{},
        Tile<_16,_16,_16>{}
    );

    if (mma_tid >= int(size(mma))) return;

    // Same accumulation chain as the fp16/bf16-store variants; only the
    // epilogue differs (raw fp32 accumulator stored elementwise).
    auto thr_mma = mma.get_slice(mma_tid);
    Tensor tCsC = thr_mma.partition_C(C);
    Tensor tCrC = thr_mma.make_fragment_C(tCsC);
    clear(tCrC);

    cooperative_gemm(mma_tid, mma, A, B, tCrC, cute::identity{}, cute::identity{}, SM75_U32x4_LDSM_N{}, SM75_U32x4_LDSM_N{});

    cute::copy(tCrC, tCsC);
}

// (I + L)^-1 via 8x8 fp32 forward substitution + 16x16 bf16-HMMA block merge.
//
// X = I + L (unit lower triangular, 16x16) is split into 8x8 blocks:
//   X = [A 0; C B]   =>   X^-1 = [A^-1  0; -(B^-1 C) A^-1  B^-1]
// The seed L stays fp32 all the way into the forward substitution (no
// input quantization). The diagonal 8x8 inverses are computed by fp32
// forward substitution (sequential rank-1 updates, exact FMA order). Unlike
// the previous fp16 Neumann series (I-L)(I+L^2)(I+L^4)(I+L^8), this never
// forms L^k intermediates, so it stays accurate for near-collinear keys
// where |L| -> 1 (the Neumann powers reach ~1e3 there and cancel
// catastrophically in fp16). The off-diagonal merge reuses the fused HMMA
// plumbing with bf16 operands and fp32 accumulation, quantizing fp32 ->
// bf16 only at the HMMA inputs:
//   P  = [A^-1 0; 0 B^-1]   (bf16, staged into INV_bf16)
//   M  = [0 0; C 0]         (bf16, staged into M_bf16, C = seed = L)
//   dc = P @ M              (bf16 HMMA, fp32 acc; rows 8-15 hold B^-1 C)
//   o  = bf16(-dc) @ P      (negate fp32 -> bf16, then fp32 acc)
//   INV = P + bf16(o)       (elementwise bf16 add; nonzero blocks disjoint)
// Result is stored bf16 like before.
template <class TensorL32, class TensorM, class TensorINV>
CUTLASS_DEVICE void inv_fwd_subst_fused_1warp(
    TensorL32 const& L_fp32,
    TensorM& M_bf16,
    TensorINV& INV_bf16,
    int tid
) {
    using BF16 = cutlass::bfloat16_t;

    auto mma = make_tiled_mma(
        SM80_16x8x16_F32BF16BF16F32_TN{},
        Layout<Shape<_1,_1>>{},
        Tile<_16,_16,_16>{}
    );
    if (tid >= int(size(mma))) return;

    // ---- 8x8 forward substitution (fp32, one row per lane) ----
    // Each 8-lane group handles one diagonal block of seed = L; lane i owns
    // row i and keeps it in registers. Groups 0/2 do block A (rows 0-7),
    // groups 1/3 do block B (rows 8-15); the redundant copies keep every
    // shuffle converged. At step s the finalized pivot row s is broadcast
    // from its owner lane.
    const int i = tid & 7;
    const int row0 = tid & 8;
    float inv[8];
    #pragma unroll
    for (int p = 0; p < 8; ++p) {
        inv[p] = (p < i) ? L_fp32(row0 + i, row0 + p)
                         : (p == i ? 1.0f : 0.0f);
    }
    #pragma unroll
    for (int s = 0; s < 7; ++s) {
        const float row_scale = -inv[s];
        #pragma unroll
        for (int p = 0; p < s; ++p) {
            const float pivot = __shfl_sync(0xFFFFFFFFu, inv[p], (tid & ~7) | s);
            if (i > s) inv[p] = fmaf(row_scale, pivot, inv[p]);
        }
        if (i > s) inv[s] = row_scale;
    }

    // ---- stage P (into INV_bf16) and M (into M_bf16) ----
    // L_fp32 is read-only here, so no sync is needed before staging.
    const int group = tid >> 3;
    if (group == 0) {
        // P rows 0-7: A^-1 in the left half (zero above the diagonal), 0 right
        #pragma unroll
        for (int j = 0; j < 16; ++j)
            INV_bf16(i, j) = (j < 8) ? BF16(inv[j]) : BF16::bitcast(0);
    } else if (group == 1) {
        // P rows 8-15: 0 left, B^-1 in the right half
        #pragma unroll
        for (int j = 0; j < 16; ++j)
            INV_bf16(8 + i, j) = (j < 8) ? BF16::bitcast(0) : BF16(inv[j - 8]);
    } else if (group == 2) {
        // M upper half = 0
        #pragma unroll
        for (int j = 0; j < 16; ++j) M_bf16(i, j) = BF16::bitcast(0);
    } else {
        // M lower half: C = seed in the left half (fp32 -> bf16), 0 right
        #pragma unroll
        for (int j = 0; j < 16; ++j)
            M_bf16(8 + i, j) = (j < 8) ? BF16(L_fp32(8 + i, j)) : BF16::bitcast(0);
    }
    __syncwarp();

    auto thr_mma = mma.get_slice(tid);

    auto smem_copy_A = make_tiled_copy_A(Copy_Atom<SM75_U32x4_LDSM_N, BF16>{}, mma);
    auto thr_copy_A = smem_copy_A.get_thread_slice(tid);

    Tensor tCrP = thr_mma.partition_fragment_A(INV_bf16);
    {
        Tensor tmp = make_fragment_like<BF16>(tCrP);
        copy(smem_copy_A, thr_copy_A.partition_S(INV_bf16), thr_copy_A.retile_D(tmp));
        cute::transform(tmp, tCrP, cute::identity{});
    }

    Tensor tCrM = thr_mma.partition_fragment_A(M_bf16);
    {
        Tensor tmp = make_fragment_like<BF16>(tCrM);
        copy(smem_copy_A, thr_copy_A.partition_S(M_bf16), thr_copy_A.retile_D(tmp));
        cute::transform(tmp, tCrM, cute::identity{});
    }

    uint32_t* P_a = reinterpret_cast<uint32_t*>(&tCrP(0));
    uint32_t* M_a = reinterpret_cast<uint32_t*>(&tCrM(0));

    uint32_t M_b[4], P_b[4], negdc_a[4], o_b[4], INV_c[4];
    float dc_c[8], o_c[8];

    auto transpose_u32x4 = [](uint32_t const* src, uint32_t* dst) {
        SM75_U32x1_MOVM_T::copy(src[0], dst[0]);
        SM75_U32x1_MOVM_T::copy(src[1], dst[1]);
        SM75_U32x1_MOVM_T::copy(src[2], dst[2]);
        SM75_U32x1_MOVM_T::copy(src[3], dst[3]);
    };

    // 16x16 MMA (fp32 acc) = two m16n8k16 atoms along N
    auto mma_16x16 = [](float* d, uint32_t const* a, uint32_t const* b, float const* c) {
        SM80_16x8x16_F32BF16BF16F32_TN::fma(d[0], d[1], d[2], d[3], a[0], a[1], a[2], a[3], b[0], b[1], c[0], c[1], c[2], c[3]);
        SM80_16x8x16_F32BF16BF16F32_TN::fma(d[4], d[5], d[6], d[7], a[0], a[1], a[2], a[3], b[2], b[3], c[4], c[5], c[6], c[7]);
    };

    auto clear_f32x8 = [](float* x) {
        #pragma unroll
        for (int j = 0; j < 8; ++j) x[j] = 0.0f;
    };

    // Pack two fp32 -> one bf16x2 u32 (RNE; first element in the low half,
    // matching the A-fragment pair order of the C accumulator).
    auto pack_bf16x2 = [](float lo, float hi) -> uint32_t {
        union U32B2 { uint32_t u; __nv_bfloat162 b2; } t;
        t.b2 = __floats2bfloat162_rn(lo, hi);
        return t.u;
    };

    // dc = P @ M (rows 8-15 hold B^-1 C; rows 0-7 are zero)
    transpose_u32x4(M_a, M_b);
    clear_f32x8(dc_c);
    mma_16x16(dc_c, P_a, M_b, dc_c);

    // o = bf16(-dc) @ P (fp32 negate with fast-math ftz, then quantize)
    transpose_u32x4(P_a, P_b);
    #pragma unroll
    for (int j = 0; j < 4; ++j)
        negdc_a[j] = pack_bf16x2(dc_c[2 * j] * -1.0f, dc_c[2 * j + 1] * -1.0f);
    clear_f32x8(o_c);
    mma_16x16(o_c, negdc_a, P_b, o_c);

    // INV = P + bf16(o) (nonzero blocks disjoint, so the add is exact)
    #pragma unroll
    for (int j = 0; j < 4; ++j)
        o_b[j] = pack_bf16x2(o_c[2 * j], o_c[2 * j + 1]);
    {
        union U32B2 { uint32_t u; __nv_bfloat162 b2; };
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            U32B2 p{P_a[j]}, o{o_b[j]}, r;
            r.b2 = __hadd2(p.b2, o.b2);
            INV_c[j] = r.u;
        }
    }

    // Store: STSM bf16 result to smem
    Tensor tCsC_mma = thr_mma.partition_C(INV_bf16);
    Tensor tCrC = thr_mma.make_fragment_C(tCsC_mma);
    Tensor tCrC_bf16 = make_fragment_like<BF16>(tCrC);
    uint32_t* out_regs = reinterpret_cast<uint32_t*>(&tCrC_bf16(0));
    out_regs[0] = INV_c[0]; out_regs[1] = INV_c[1]; out_regs[2] = INV_c[2]; out_regs[3] = INV_c[3];

    auto smem_tiled_store = make_tiled_copy_C(Copy_Atom<SM90_U32x4_STSM_N, BF16>{}, mma);
    auto smem_thr_store = smem_tiled_store.get_slice(tid);
    Tensor tCsC_st = smem_thr_store.partition_D(INV_bf16);
    Tensor tCrC_st_view = smem_thr_store.retile_S(tCrC_bf16);
    copy(smem_tiled_store, tCrC_st_view, tCsC_st);
}

// ==================== FP32 <-> BF16 state conversion in SMEM ====================
// Both FP32 (K_SW32) and BF16 (K_INTER) layouts resolve to the same 8x8 atom
// structure with Swizzle<0,0,3>. Conversion operates per-atom:
//   - Each warp handles one 8x8 atom (64 elements)
//   - Each thread converts 2 elements
//   - Warp-level iteration over all atoms in the D x D state

template <class FP32Layout, class BF16Layout, int D, int NumThreads>
__device__ void smem_cvt_fp32_to_bf16(
    float* __restrict__ fp32_smem,
    cutlass::bfloat16_t* __restrict__ bf16_smem,
    int tid
) {
    using BF16 = cutlass::bfloat16_t;
    constexpr int kBlock = 8;
    constexpr int kBlocksPerDim = D / kBlock;
    constexpr int kTotalBlocks = kBlocksPerDim * kBlocksPerDim;
    constexpr int kWarpSize = 32;

    auto fp32_view = make_tensor(make_smem_ptr(fp32_smem), FP32Layout{});
    auto bf16_view = make_tensor(make_smem_ptr(bf16_smem), BF16Layout{});

    int warp_id = tid / kWarpSize;
    int lane_id = tid % kWarpSize;
    int num_warps = NumThreads / kWarpSize;

    for (int blk = warp_id; blk < kTotalBlocks; blk += num_warps) {
        int br = (blk / kBlocksPerDim) * kBlock;
        int bc = (blk % kBlocksPerDim) * kBlock;
        int e0 = lane_id * 2;
        int e1 = lane_id * 2 + 1;
        int r0 = br + e0 / kBlock, c0 = bc + e0 % kBlock;
        int r1 = br + e1 / kBlock, c1 = bc + e1 % kBlock;
        bf16_view(r0, c0) = BF16(fp32_view(r0, c0));
        bf16_view(r1, c1) = BF16(fp32_view(r1, c1));
    }
}

template <class BF16Layout, class FP32Layout, int D, int NumThreads>
__device__ void smem_cvt_bf16_to_fp32(
    cutlass::bfloat16_t* __restrict__ bf16_smem,
    float* __restrict__ fp32_smem,
    int tid
) {
    constexpr int kBlock = 8;
    constexpr int kBlocksPerDim = D / kBlock;
    constexpr int kTotalBlocks = kBlocksPerDim * kBlocksPerDim;
    constexpr int kWarpSize = 32;

    auto bf16_view = make_tensor(make_smem_ptr(bf16_smem), BF16Layout{});
    auto fp32_view = make_tensor(make_smem_ptr(fp32_smem), FP32Layout{});

    int warp_id = tid / kWarpSize;
    int lane_id = tid % kWarpSize;
    int num_warps = NumThreads / kWarpSize;

    for (int blk = warp_id; blk < kTotalBlocks; blk += num_warps) {
        int br = (blk / kBlocksPerDim) * kBlock;
        int bc = (blk % kBlocksPerDim) * kBlock;
        int e0 = lane_id * 2;
        int e1 = lane_id * 2 + 1;
        int r0 = br + e0 / kBlock, c0 = bc + e0 % kBlock;
        int r1 = br + e1 / kBlock, c1 = bc + e1 % kBlock;
        fp32_view(r0, c0) = bf16_to_f32(bf16_view(r0, c0));
        fp32_view(r1, c1) = bf16_to_f32(bf16_view(r1, c1));
    }
}

