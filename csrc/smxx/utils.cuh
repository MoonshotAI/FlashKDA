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

__device__ __forceinline__ float sigmoid_tanh_approx_f32(float x) {
    float th = tanh_approx_f32(x * 0.5f);
    return th * 0.5f + 0.5f;
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
CUTLASS_DEVICE void mma_m16n16_bf16bf16fp16_1warp(
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

    using FP16 = cutlass::half_t;

    auto sC_store_op = [] __device__ (float x) { return FP16(x); };

    cooperative_gemm(mma_tid, mma, 1.0f, A, B, 0.0f, C, cute::identity{}, cute::identity{}, cute::identity{}, sC_store_op, SM75_U32x4_LDSM_N{}, SM75_U32x4_LDSM_N{}, SM75_U32x4_LDSM_N{}, SM90_U32x4_STSM_N{});
}

template <class TensorL, class TensorINV_fp16, class TensorINV_bf16>
CUTLASS_DEVICE void neumann_inv_fused_1warp(
    TensorL const& L_fp16,
    TensorINV_fp16 const& INV_fp16,
    TensorINV_bf16& INV_bf16_out,
    int tid
) {
    using FP16 = cutlass::half_t;
    using BF16 = cutlass::bfloat16_t;

    auto mma = make_tiled_mma(
        SM80_16x8x16_F16F16F16F16_TN{},
        Layout<Shape<_1,_1>>{},
        Tile<_16,_16,_16>{}
    );
    if (tid >= int(size(mma))) return;

    auto thr_mma = mma.get_slice(tid);

    auto smem_copy_A = make_tiled_copy_A(Copy_Atom<SM75_U32x4_LDSM_N, FP16>{}, mma);
    auto thr_copy_A = smem_copy_A.get_thread_slice(tid);

    Tensor tCrL = thr_mma.partition_fragment_A(L_fp16);
    {
        Tensor tmp = make_fragment_like<FP16>(tCrL);
        copy(smem_copy_A, thr_copy_A.partition_S(L_fp16), thr_copy_A.retile_D(tmp));
        cute::transform(tmp, tCrL, cute::identity{});
    }

    Tensor tCrINV = thr_mma.partition_fragment_A(INV_fp16);
    {
        Tensor tmp = make_fragment_like<FP16>(tCrINV);
        copy(smem_copy_A, thr_copy_A.partition_S(INV_fp16), thr_copy_A.retile_D(tmp));
        cute::transform(tmp, tCrINV, cute::identity{});
    }

    uint32_t* L_a = reinterpret_cast<uint32_t*>(&tCrL(0));
    uint32_t* INV_a = reinterpret_cast<uint32_t*>(&tCrINV(0));

    uint32_t Lpow_c[4], Lpow_b[4], INV_c[4], tmp_a[4], mm_c[4];

    auto clear_u32x4 = [](uint32_t* x) {
        x[0] = x[1] = x[2] = x[3] = 0;
    };

    auto add_fp16x2_u32x4 = [] (uint32_t* dst, uint32_t const* src) {
        union U32H2 { uint32_t u; __half2 h2; };
        U32H2 a0{dst[0]}, b0{src[0]}, a1{dst[1]}, b1{src[1]};
        U32H2 a2{dst[2]}, b2{src[2]}, a3{dst[3]}, b3{src[3]};
        a0.h2 = __hadd2(a0.h2, b0.h2);
        a1.h2 = __hadd2(a1.h2, b1.h2);
        a2.h2 = __hadd2(a2.h2, b2.h2);
        a3.h2 = __hadd2(a3.h2, b3.h2);
        dst[0] = a0.u; dst[1] = a1.u; dst[2] = a2.u; dst[3] = a3.u;
    };

    auto transpose_u32x4 = [](uint32_t const* src, uint32_t* dst) {
        SM75_U32x1_MOVM_T::copy(src[0], dst[0]);
        SM75_U32x1_MOVM_T::copy(src[1], dst[1]);
        SM75_U32x1_MOVM_T::copy(src[2], dst[2]);
        SM75_U32x1_MOVM_T::copy(src[3], dst[3]);
    };

    auto copy_u32x4 = [](uint32_t const* src, uint32_t* dst) {
        dst[0] = src[0]; dst[1] = src[1]; dst[2] = src[2]; dst[3] = src[3];
    };

    // 16x16 MMA = two m16n8k16 atoms along N
    auto mma_16x16 = [](uint32_t* d, uint32_t const* a, uint32_t const* b, uint32_t const* c) {
        SM80_16x8x16_F16F16F16F16_TN::fma(d[0], d[1], a[0], a[1], a[2], a[3], b[0], b[1], c[0], c[1]);
        SM80_16x8x16_F16F16F16F16_TN::fma(d[2], d[3], a[0], a[1], a[2], a[3], b[2], b[3], c[2], c[3]);
    };

    // L^2 = L × L
    transpose_u32x4(L_a, Lpow_b);
    clear_u32x4(Lpow_c);
    mma_16x16(Lpow_c, L_a, Lpow_b, Lpow_c);

    // INV += INV × L^2
    transpose_u32x4(Lpow_c, Lpow_b);
    copy_u32x4(INV_a, INV_c);
    clear_u32x4(mm_c);
    mma_16x16(mm_c, INV_a, Lpow_b, mm_c);
    add_fp16x2_u32x4(INV_c, mm_c);

    // L^4 = L^2 × L^2
    copy_u32x4(Lpow_c, tmp_a);
    clear_u32x4(Lpow_c);
    mma_16x16(Lpow_c, tmp_a, Lpow_b, Lpow_c);

    // INV += INV × L^4
    transpose_u32x4(Lpow_c, Lpow_b);
    copy_u32x4(INV_c, tmp_a);
    clear_u32x4(mm_c);
    mma_16x16(mm_c, tmp_a, Lpow_b, mm_c);
    add_fp16x2_u32x4(INV_c, mm_c);

    // L^8 = L^4 × L^4
    copy_u32x4(Lpow_c, tmp_a);
    clear_u32x4(Lpow_c);
    mma_16x16(Lpow_c, tmp_a, Lpow_b, Lpow_c);

    // INV += INV × L^8
    transpose_u32x4(Lpow_c, Lpow_b);
    copy_u32x4(INV_c, tmp_a);
    clear_u32x4(mm_c);
    mma_16x16(mm_c, tmp_a, Lpow_b, mm_c);
    add_fp16x2_u32x4(INV_c, mm_c);

    // Store: convert C-format fp16 → bf16, write to smem
    Tensor tCsC_mma = thr_mma.partition_C(INV_fp16);
    Tensor tCrC = thr_mma.make_fragment_C(tCsC_mma);
    uint32_t* C_regs = reinterpret_cast<uint32_t*>(&tCrC(0));
    C_regs[0] = INV_c[0]; C_regs[1] = INV_c[1]; C_regs[2] = INV_c[2]; C_regs[3] = INV_c[3];

    Tensor tCrC_bf16 = make_fragment_like<BF16>(tCrC);
    cute::transform(tCrC, tCrC_bf16, [] __device__ (FP16 x) -> BF16 { return BF16(x); });

    auto smem_tiled_store = make_tiled_copy_C(Copy_Atom<SM90_U32x4_STSM_N, BF16>{}, mma);
    auto smem_thr_store = smem_tiled_store.get_slice(tid);
    Tensor tCsC_st = smem_thr_store.partition_D(INV_bf16_out);
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

