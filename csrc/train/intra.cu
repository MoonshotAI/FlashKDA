// KDA forward intra-chunk kernels, replicating fla/ops/kda/chunk_intra.py
// (chunk_kda_fwd_kernel_intra_sub_chunk, chunk_kda_fwd_kernel_inter_solve_fused)
// and fla/ops/kda/chunk_intra_token_parallel.py.
//
// All gate math is fp32 in the log2 domain (g is the chunk-local inclusive
// cumsum scaled by RCP_LN2). 16x16 GEMMs run on tensor cores through the CuTe
// SM80 tf32 atom (SM80_16x8x8_F32TF32TF32F32_TN) to match Triton's tf32 dots;
// the token_parallel kernel replicates Triton's fp32 elementwise + tl.sum
// semantics and therefore intentionally uses no MMA.

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>

#include <cute/tensor.hpp>
#include <cute/arch/mma_sm80.hpp>

#include "common.cuh"

// Named namespace (not anonymous): nvcc's registration stub cannot spell
// template instantiations from anonymous namespaces (ambiguous mangled-name
// references in the generated stub).
namespace kda_train_intra {

using namespace cute;

using BF16 = cutlass::bfloat16_t;
using FP16 = cutlass::half_t;
using TF32 = cutlass::tfloat32_t;

constexpr int kBC = 16;   // sub-chunk size (fla BC)
constexpr int kKC2 = 32;  // K staging chunk; keeps the ascending-K mma order

// tl.math.exp2 lowers to ex2.approx; keep the same instruction.
__device__ __forceinline__ float exp2_ftz(float x) {
    float result;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(result) : "f"(x));
    return result;
}

// fp32 -> tf32 with the same rounding Triton applies before tf32 mma.
__device__ __forceinline__ TF32 f32_to_tf32(float x) {
    uint32_t r;
    asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(r) : "f"(x));
    return TF32::bitcast(r);
}

__device__ __forceinline__ uint4 pack4_tf32(float const* v) {
    return make_uint4(f32_to_tf32(v[0]).storage, f32_to_tf32(v[1]).storage,
                      f32_to_tf32(v[2]).storage, f32_to_tf32(v[3]).storage);
}

// One warp computing C[16,16] (fp32) += A[16,Kc] @ B[16,Kc]^T with tf32
// operands in shared memory (row-major, K contiguous). The tiled MMA is the
// 16x8x8 tf32 atom value-tiled to 16x16x8.
using MmaTF32_16 = decltype(make_tiled_mma(
    SM80_16x8x8_F32TF32TF32F32_TN{},
    Layout<Shape<_1, _1, _1>>{},
    Tile<_16, _16, _8>{}
));

template <class Mma, class ThrMma, class Acc, class Coord>
struct Mma16Ctx {
    Mma mma;
    ThrMma thr;
    Acc acc;
    Coord coord;  // identity-tensor partition mapping acc(v) -> (i, j)
};

template <class Mma>
__device__ __forceinline__ auto make_mma16_ctx(int lane) {
    Mma mma{};
    auto thr = mma.get_slice(lane);
    auto coord = thr.partition_C(make_identity_tensor(Shape<_16, _16>{}));
    auto acc = thr.make_fragment_C(coord);
    clear(acc);
    return Mma16Ctx<Mma, decltype(thr), decltype(acc), decltype(coord)>{mma, thr, acc, coord};
}

// Row strides are padded (+4 floats) so the tf32 fragment loads of lanes
// grouped 4-apart (rows lane/4) land on distinct bank quads: stride%32 == 4
// keeps rows r and r+1 eight banks apart, making the scalar ld.shared in
// mma16_accum / merge_gemm_16 conflict-free.
constexpr int kPadHalf = 32 + 4;
constexpr int kPad16 = 16 + 4;
using SmemLayoutHalf = Layout<Shape<_16, _32>, Stride<Int<kPadHalf>, _1>>;  // [16, 32] staging tiles
using SmemLayout16 = Layout<Shape<_16, _16>, Stride<Int<kPad16>, _1>>;    // [16, 16] merge tiles

template <class Ctx, class SmemLayout>
__device__ __forceinline__ void mma16_accum(
    Ctx& ctx,
    TF32 const* sA,  // [16, K] row-major
    TF32 const* sB,  // [16, K] row-major; computes A @ B^T
    SmemLayout const& lay
) {
    Tensor sAt = make_tensor(make_smem_ptr(sA), lay);
    Tensor sBt = make_tensor(make_smem_ptr(sB), lay);
    Tensor tCsA = ctx.thr.partition_A(sAt);
    Tensor tCsB = ctx.thr.partition_B(sBt);
    Tensor tCrA = ctx.thr.partition_fragment_A(sAt);
    Tensor tCrB = ctx.thr.partition_fragment_B(sBt);
    copy(tCsA, tCrA);
    copy(tCsB, tCrB);
    CUTE_UNROLL
    for (int kb = 0; kb < size<2>(tCrA); ++kb) {
        gemm(ctx.mma, tCrA(_, _, kb), tCrB(_, _, kb), ctx.acc);
    }
}

// In-place forward substitution on a 16x16 fp32 smem block holding -tril(C,-1):
// turns it into (I + tril(C,-1))^-1 minus the identity (the caller adds I).
// Equivalent to the gmem read-back loop in the Triton kernels: the row read
// there is exactly the current (negated) row of the block. lanes 0..15 only;
// each lane owns one column.
__device__ __forceinline__ void fwd_subst_16(float* sAi, int lim, int lane) {
    // only lanes 0..15 participate; sync with an explicit partial mask
    float a[kBC];
    for (int i = 2; i < lim; ++i) {
        CUTE_UNROLL
        for (int r = 0; r < kBC; ++r) a[r] = (r < i) ? sAi[i * kBC + r] : 0.f;
        float acc = a[lane];
        CUTE_UNROLL
        for (int r = 0; r < kBC; ++r) acc += a[r] * sAi[r * kBC + lane];
        __syncwarp(0xffffu);
        sAi[i * kBC + lane] = acc;
        __syncwarp(0xffffu);
    }
}

struct VarlenArgs {
    int64_t const* cu_seqlens = nullptr;
    int64_t const* chunk_indices = nullptr;
    int64_t NT = 0;
};

VarlenArgs resolve_varlen_intra(
    std::optional<torch::Tensor> const& cu_seqlens,
    std::optional<torch::Tensor> const& chunk_indices,
    int64_t B,
    int64_t T,
    int64_t chunk_size
) {
    VarlenArgs args;
    if (cu_seqlens.has_value()) {
        TORCH_CHECK(B == 1, "B must be 1 when cu_seqlens is provided");
        TORCH_CHECK(chunk_indices.has_value(), "chunk_indices must be provided when cu_seqlens is provided");
        auto const& ci = chunk_indices.value();
        TORCH_CHECK(ci.dtype() == torch::kLong && ci.is_cuda() && ci.is_contiguous());
        TORCH_CHECK(ci.dim() == 2 && ci.size(1) == 2);
        auto const& cu = cu_seqlens.value();
        TORCH_CHECK(cu.dtype() == torch::kLong && cu.is_cuda() && cu.is_contiguous());
        args.cu_seqlens = cu.data_ptr<int64_t>();
        args.chunk_indices = ci.data_ptr<int64_t>();
        args.NT = ci.size(0);
    } else {
        args.NT = (T + chunk_size - 1) / chunk_size;
    }
    return args;
}

void check_intra_inputs(
    torch::Tensor const& q,
    torch::Tensor const& k,
    torch::Tensor const& g,
    torch::Tensor const& beta,
    torch::Tensor const& Aqk,
    torch::Tensor const& Akkd,
    int64_t chunk_size
) {
    TORCH_CHECK(q.is_cuda() && q.is_contiguous(), "q must be contiguous CUDA tensor");
    TORCH_CHECK(k.is_cuda() && k.is_contiguous(), "k must be contiguous CUDA tensor");
    TORCH_CHECK(g.is_cuda() && g.is_contiguous(), "g must be contiguous CUDA tensor");
    TORCH_CHECK(beta.is_cuda() && beta.is_contiguous(), "beta must be contiguous CUDA tensor");
    TORCH_CHECK(q.dim() == 4 && k.dim() == 4, "q/k must be [B, T, H, K]");
    TORCH_CHECK(q.sizes() == k.sizes() && q.scalar_type() == k.scalar_type(), "q/k must match");
    TORCH_CHECK(q.scalar_type() == at::kBFloat16 || q.scalar_type() == at::kHalf, "q/k must be bf16/fp16");
    int64_t B = k.size(0), T = k.size(1), H = k.size(2), K = k.size(3);
    TORCH_CHECK(K <= 256, "K must be <= 256");
    TORCH_CHECK(chunk_size == 32 || chunk_size == 64, "chunk_size must be 32 or 64");
    TORCH_CHECK(g.dim() == 4 && g.scalar_type() == at::kFloat, "g must be fp32 [B, T, HV, K]");
    TORCH_CHECK(g.size(0) == B && g.size(1) == T && g.size(3) == K, "g shape mismatch");
    int64_t HV = g.size(2);
    TORCH_CHECK(HV % H == 0, "HV must be a multiple of H");
    TORCH_CHECK(beta.dim() == 3 && beta.size(0) == B && beta.size(1) == T && beta.size(2) == HV,
                "beta must be [B, T, HV]");
    TORCH_CHECK(beta.scalar_type() == at::kBFloat16 || beta.scalar_type() == at::kHalf ||
                beta.scalar_type() == at::kFloat, "beta must be bf16/fp16/fp32");
    TORCH_CHECK(Aqk.is_cuda() && Aqk.is_contiguous() && Aqk.scalar_type() == k.scalar_type());
    TORCH_CHECK(Aqk.dim() == 4 && Aqk.size(0) == B && Aqk.size(1) == T && Aqk.size(2) == HV &&
                Aqk.size(3) == chunk_size, "Aqk must be [B, T, HV, BT]");
    TORCH_CHECK(Akkd.is_cuda() && Akkd.is_contiguous() && Akkd.scalar_type() == at::kFloat);
    TORCH_CHECK(Akkd.dim() == 4 && Akkd.size(0) == B && Akkd.size(1) == T && Akkd.size(2) == HV &&
                Akkd.size(3) == kBC, "Akkd must be [B, T, HV, 16] fp32");
    TORCH_CHECK(B * HV <= 65535, "B*HV exceeds gridDim.z limit");
}

// ---------------------------------------------------------------------------
// Kernel 1: chunk_kda_fwd_kernel_intra_sub_chunk (safe_gate path)
// grid (NT, B*HV), block 128. Each warp computes one 16x16 diagonal block of
// the chunk (warp w <-> sub-chunk w): the Aqk diag block (scaled, bf16) and
// the inverted Akk diag block (fp32 into Akkd). Warps are fully independent
// (private smem scratch, warp-local mma and forward substitution, no
// block-wide barriers), so the four diag blocks of a chunk are computed
// concurrently instead of serialized on warp 0 as in v1. Per-element math
// and the ascending-K mma order are unchanged.
// ---------------------------------------------------------------------------
template <typename QKT, typename BetaT, int BT, bool IS_VARLEN>
__global__ void __launch_bounds__(128) chunk_kda_fwd_kernel_intra_sub_chunk_cuda(
    QKT const* __restrict__ q,
    QKT const* __restrict__ k,
    float const* __restrict__ g,
    BetaT const* __restrict__ beta,
    QKT* __restrict__ Aqk,
    float* __restrict__ Akkd,
    float scale,
    int64_t const* __restrict__ cu_seqlens,
    int64_t const* __restrict__ chunk_indices,
    int T,
    int H,
    int HV,
    int K
) {
    constexpr int NC = BT / kBC;
    int64_t i_t = blockIdx.x;
    int64_t i_bh = blockIdx.y;
    int i_hv = int(i_bh % HV);
    int i_h = i_hv / (HV / H);

    int64_t bos;
    int Tseq = T;
    if (IS_VARLEN) {
        int64_t i_n = chunk_indices[i_t * 2];
        i_t = chunk_indices[i_t * 2 + 1];
        bos = cu_seqlens[i_n];
        Tseq = int(cu_seqlens[i_n + 1] - bos);
    } else {
        bos = (i_bh / HV) * (int64_t)T;
    }
    if (i_t * BT >= Tseq) return;

    int const tid = threadIdx.x;
    int const lane = tid & 31;
    int const warp = tid >> 5;
    int const i_i = warp;  // sub-chunk handled by this warp
    if (i_i >= NC) return;  // no block-wide barriers in this kernel
    int const i_ti = int(i_t) * BT + i_i * kBC;
    if (i_ti >= Tseq) return;

    __shared__ TF32 sW[4][3][kBC * kPadHalf];  // per-warp Aq/Ak/B staging tiles
    __shared__ float sAi[4][kBC * kBC];
    __shared__ float sBeta[NC][kBC];

    TF32* sAq = &sW[warp][0][0];
    TF32* sAk = &sW[warp][1][0];
    TF32* sB = &sW[warp][2][0];
    float* sAiW = &sAi[warp][0];

    if (lane < kBC) {
        int t = i_ti + lane;
        sBeta[warp][lane] = (t < Tseq) ? to_f32(beta[(bos + t) * HV + i_hv]) : 0.f;
    }
    // midpoint reference row of the sub-chunk (numerical stability)
    int const gn_row = i_ti + min(kBC / 2, Tseq - i_ti - 1);

    auto ctx_qk = make_mma16_ctx<MmaTF32_16>(lane);
    auto ctx_kk = make_mma16_ctx<MmaTF32_16>(lane);

    int const n_kchunks = (K + kKC2 - 1) / kKC2;
    bool const kvec = (K % 8) == 0;
    // lane covers col group cg = (lane%4)*8 and rows lane/4, lane/4+8
    int const cg = (lane & 3) * 8;
    int const r0 = lane >> 2;
    float const* gn_base = g + ((bos + gn_row) * HV + i_hv) * (int64_t)K;
    if (kvec) {
        // Pass-level pipeline: the next 8-col pass's raw gmem payload is
        // issued before the current pass is gated/packed/stored, hiding the
        // gmem latency behind the exp2/pack math and the fragment MMAs.
        // Raw payload of one pass: reference-row gates + row gates + q/k.
        struct RawPass {
            float4 gn0, gn1, gv0, gv1;
            uint4 rq, rk;
            unsigned flags;  // bit0: row valid, bit1: full 8-col group
        };
        auto load_pass = [&](int kc, int pass, RawPass& R) {
            int const kk = kc * kKC2 + cg;
            int const r = r0 + pass * 8;
            int const t = i_ti + r;
            bool const ok = t < Tseq;
            bool const v8 = (K - kk) >= 8;  // K%8==0: group is full or empty
            R.flags = (ok ? 1u : 0u) | (v8 ? 2u : 0u);
            if (v8) {
                float4 const* pg = reinterpret_cast<float4 const*>(gn_base + kk);
                R.gn0 = pg[0]; R.gn1 = pg[1];
                if (ok) {
                    float4 const* pr = reinterpret_cast<float4 const*>(g + ((bos + t) * HV + i_hv) * (int64_t)K + kk);
                    R.gv0 = pr[0]; R.gv1 = pr[1];
                    R.rq = *reinterpret_cast<uint4 const*>(q + ((bos + t) * H + i_h) * (int64_t)K + kk);
                    R.rk = *reinterpret_cast<uint4 const*>(k + ((bos + t) * H + i_h) * (int64_t)K + kk);
                }
            }
        };
        auto store_pass = [&](RawPass const& R, int pass) {
            int const r = r0 + pass * 8;
            bool const ok = (R.flags & 1u) != 0;
            bool const v8 = (R.flags & 2u) != 0;
            float gnv[8] = {}, gv[8] = {}, qv[8] = {}, kv[8] = {};
            if (v8) {
                gnv[0] = R.gn0.x; gnv[1] = R.gn0.y; gnv[2] = R.gn0.z; gnv[3] = R.gn0.w;
                gnv[4] = R.gn1.x; gnv[5] = R.gn1.y; gnv[6] = R.gn1.z; gnv[7] = R.gn1.w;
                if (ok) {
                    gv[0] = R.gv0.x; gv[1] = R.gv0.y; gv[2] = R.gv0.z; gv[3] = R.gv0.w;
                    gv[4] = R.gv1.x; gv[5] = R.gv1.y; gv[6] = R.gv1.z; gv[7] = R.gv1.w;
                    QKT const* hq = reinterpret_cast<QKT const*>(&R.rq);
                    QKT const* hk = reinterpret_cast<QKT const*>(&R.rk);
                    CUTE_UNROLL
                    for (int j = 0; j < 8; ++j) { qv[j] = to_f32(hq[j]); kv[j] = to_f32(hk[j]); }
                }
            }
            float oq[8], okk2[8], ob[8];
            CUTE_UNROLL
            for (int j = 0; j < 8; ++j) {
                float gq = ok ? exp2_ftz(gv[j] - gnv[j]) : 0.f;
                float gk = ok ? exp2_ftz(gnv[j] - gv[j]) : 0.f;
                oq[j] = qv[j] * gq;
                okk2[j] = kv[j] * gq;
                ob[j] = kv[j] * gk;
            }
            int const soff = r * kPadHalf + cg;
            *reinterpret_cast<uint4*>(sAq + soff) = pack4_tf32(oq);
            *reinterpret_cast<uint4*>(sAq + soff + 4) = pack4_tf32(oq + 4);
            *reinterpret_cast<uint4*>(sAk + soff) = pack4_tf32(okk2);
            *reinterpret_cast<uint4*>(sAk + soff + 4) = pack4_tf32(okk2 + 4);
            *reinterpret_cast<uint4*>(sB + soff) = pack4_tf32(ob);
            *reinterpret_cast<uint4*>(sB + soff + 4) = pack4_tf32(ob + 4);
        };
        RawPass cur, nxt;
        load_pass(0, 0, cur);
        for (int kc = 0; kc < n_kchunks; ++kc) {
            CUTE_UNROLL
            for (int pass = 0; pass < 2; ++pass) {
                int const nkc = kc + pass;
                if (nkc < n_kchunks) load_pass(nkc, (pass + 1) & 1, nxt);
                store_pass(cur, pass);
                if (pass == 1) {
                    __syncwarp();
                    mma16_accum(ctx_qk, sAq, sB, SmemLayoutHalf{});
                    mma16_accum(ctx_kk, sAk, sB, SmemLayoutHalf{});
                    __syncwarp();
                }
                cur = nxt;
            }
        }
    } else {
    for (int kc = 0; kc < n_kchunks; ++kc) {
        int const kk = kc * kKC2 + cg;
        int const rem = K - kk;
        CUTE_UNROLL
        for (int pass = 0; pass < 2; ++pass) {
            int const r = r0 + pass * 8;
            int const t = i_ti + r;
            bool const ok = t < Tseq;
            float gnv[8] = {}, gv[8] = {}, qv[8] = {}, kv[8] = {};
            CUTE_UNROLL
            for (int j = 0; j < 8; ++j) {
                bool okk = j < rem;
                gnv[j] = okk ? gn_base[kk + j] : 0.f;
                if (ok && okk) {
                    gv[j] = g[((bos + t) * HV + i_hv) * (int64_t)K + kk + j];
                    qv[j] = to_f32(q[((bos + t) * H + i_h) * (int64_t)K + kk + j]);
                    kv[j] = to_f32(k[((bos + t) * H + i_h) * (int64_t)K + kk + j]);
                }
            }
            float oq[8], okk2[8], ob[8];
            CUTE_UNROLL
            for (int j = 0; j < 8; ++j) {
                float gq = (ok && j < rem) ? exp2_ftz(gv[j] - gnv[j]) : 0.f;
                float gk = (ok && j < rem) ? exp2_ftz(gnv[j] - gv[j]) : 0.f;
                oq[j] = qv[j] * gq;
                okk2[j] = kv[j] * gq;
                ob[j] = kv[j] * gk;
            }
            int const soff = r * kPadHalf + cg;
            *reinterpret_cast<uint4*>(sAq + soff) = pack4_tf32(oq);
            *reinterpret_cast<uint4*>(sAq + soff + 4) = pack4_tf32(oq + 4);
            *reinterpret_cast<uint4*>(sAk + soff) = pack4_tf32(okk2);
            *reinterpret_cast<uint4*>(sAk + soff + 4) = pack4_tf32(okk2 + 4);
            *reinterpret_cast<uint4*>(sB + soff) = pack4_tf32(ob);
            *reinterpret_cast<uint4*>(sB + soff + 4) = pack4_tf32(ob + 4);
        }
        __syncwarp();
        mma16_accum(ctx_qk, sAq, sB, SmemLayoutHalf{});
        mma16_accum(ctx_kk, sAk, sB, SmemLayoutHalf{});
        __syncwarp();
    }
    }

    // Aqk diagonal block: lower triangle incl. diagonal, scale applied
    CUTE_UNROLL
    for (int v = 0; v < size(ctx_qk.acc); ++v) {
        auto c = ctx_qk.coord(v);
        int i = int(get<0>(c)), j = int(get<1>(c));
        if (i_ti + i < Tseq) {
            float val = (i >= j) ? ctx_qk.acc(v) * scale : 0.f;
            Aqk[((bos + i_ti + i) * HV + i_hv) * (int64_t)BT + i_i * kBC + j] = QKT(val);
        }
    }
    // sAi = -tril(beta * Akk, -1)
    CUTE_UNROLL
    for (int v = 0; v < size(ctx_kk.acc); ++v) {
        auto c = ctx_kk.coord(v);
        int i = int(get<0>(c)), j = int(get<1>(c));
        sAiW[i * kBC + j] = (i > j) ? -ctx_kk.acc(v) * sBeta[warp][i] : 0.f;
    }
    __syncwarp();

    if (lane < kBC) {
        fwd_subst_16(sAiW, min(kBC, Tseq - i_ti), lane);
        sAiW[lane * kBC + lane] += 1.f;
    }
    __syncwarp();

    for (int idx = lane; idx < kBC * kBC; idx += 32) {
        int i = idx / kBC;
        if (i_ti + i < Tseq) {
            Akkd[((bos + i_ti + i) * HV + i_hv) * (int64_t)kBC + idx % kBC] = sAiW[idx];
        }
    }
}

// ---------------------------------------------------------------------------
// Kernel 2: chunk_kda_fwd_kernel_intra_token_parallel (non safe_gate path)
// grid (B*T, cdiv(HV, 4)), block 128 (one warp per value head). Replicates
// Triton's fp32 elementwise + tl.sum semantics (no MMA by design).
// ---------------------------------------------------------------------------
template <typename QKT, typename BetaT, int BT, bool IS_VARLEN>
__global__ void __launch_bounds__(128) chunk_kda_fwd_kernel_intra_token_parallel_cuda(
    QKT const* __restrict__ q,
    QKT const* __restrict__ k,
    float const* __restrict__ g,
    BetaT const* __restrict__ beta,
    QKT* __restrict__ Aqk,
    float* __restrict__ Akkd,
    float scale,
    int64_t const* __restrict__ cu_seqlens,
    int64_t N,
    int T,
    int H,
    int HV,
    int K
) {
    int64_t i_tg = blockIdx.x;

    int64_t bos;
    int64_t i_t;
    int Tseq;
    if (IS_VARLEN) {
        // unrolled binary search for the sequence containing token i_tg
        int left = 0, right = int(N);
        CUTE_UNROLL
        for (int it = 0; it < 20; ++it) {
            if (left < right) {
                int mid = (left + right) >> 1;
                if (i_tg < cu_seqlens[mid + 1]) right = mid;
                else left = mid + 1;
            }
        }
        bos = cu_seqlens[left];
        Tseq = int(cu_seqlens[left + 1] - bos);
        i_t = i_tg - bos;
    } else {
        bos = (i_tg / T) * T;
        i_t = i_tg % T;
        Tseq = T;
    }
    if (i_t >= Tseq) return;

    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int i_hv = blockIdx.y * 4 + warp;
    if (i_hv >= HV) return;
    int i_h = i_hv / (HV / H);

    int64_t i_c = i_t / BT;
    int i_s = int((i_t % BT) / kBC);
    int64_t i_ts = i_c * BT + i_s * kBC;

    constexpr int MAXE = 8;  // K <= 256, 32 lanes * 8
    float q_r[MAXE], kb_r[MAXE], g_r[MAXE];
    int64_t qk_base = ((bos + i_t) * H + i_h) * (int64_t)K;
    int64_t g_base = ((bos + i_t) * HV + i_hv) * (int64_t)K;
    float beta_i = to_f32(beta[(bos + i_t) * HV + i_hv]);
    CUTE_UNROLL
    for (int e = 0; e < MAXE; ++e) {
        int kk = lane + e * 32;
        bool ok = kk < K;
        q_r[e] = ok ? to_f32(q[qk_base + kk]) : 0.f;
        kb_r[e] = ok ? to_f32(k[qk_base + kk]) * beta_i : 0.f;
        g_r[e] = ok ? g[g_base + kk] : 0.f;
    }

    int64_t j_end = i_t + 1;
    if ((int64_t)Tseq < j_end) j_end = Tseq;
    if (i_ts + kBC < j_end) j_end = i_ts + kBC;
    for (int64_t j = i_ts; j < j_end; ++j) {
        int64_t kjb = ((bos + j) * H + i_h) * (int64_t)K;
        int64_t gjb = ((bos + j) * HV + i_hv) * (int64_t)K;
        float aq = 0.f, ak = 0.f;
        CUTE_UNROLL
        for (int e = 0; e < MAXE; ++e) {
            int kk = lane + e * 32;
            if (kk < K) {
                float kgj = to_f32(k[kjb + kk]) * exp2_ftz(g_r[e] - g[gjb + kk]);
                aq += q_r[e] * kgj;
                ak += kb_r[e] * kgj;
            }
        }
        CUTE_UNROLL
        for (int off = 16; off > 0; off >>= 1) {
            aq += __shfl_down_sync(0xffffffffu, aq, off);
            ak += __shfl_down_sync(0xffffffffu, ak, off);
        }
        if (lane == 0) {
            Aqk[((bos + i_t) * HV + i_hv) * (int64_t)BT + (j % BT)] = QKT(aq * scale);
            Akkd[((bos + i_t) * HV + i_hv) * (int64_t)kBC + (j - i_ts)] = (j < i_t) ? ak : 0.f;
        }
    }
}

// ---------------------------------------------------------------------------
// Kernel 3: chunk_kda_fwd_kernel_inter_solve_fused
// grid (NT, B*HV), block 128. Computes off-diagonal Aqk/Akk blocks, inverts
// the diagonal blocks (non safe_gate only; safe_gate blocks arrive inverted
// from kernel 1), merges the block-triangular inverse, and writes Akk (bf16).
// ---------------------------------------------------------------------------

// Warp-local 16x16 GEMM: D = alpha * (A @ B) with fp32 smem operands via tf32
// MMA. Same staging math, atom, and K order as v1's block-cooperative version;
// sMA/sMB are this warp's private tf32 scratch. Contains __syncwarp only.
__device__ __forceinline__ void merge_gemm_16_warp(
    float const* A,
    float const* B,
    float* D,
    float alpha,
    TF32* sMA,
    TF32* sMB,
    int lane
) {
    CUTE_UNROLL
    for (int idx = lane; idx < kBC * kBC; idx += 32) {
        int n = idx / kBC, kk = idx % kBC;
        sMA[n * kPad16 + kk] = f32_to_tf32(A[idx]);
        sMB[n * kPad16 + kk] = f32_to_tf32(B[kk * kBC + n]);  // sMB[n][kk] = B[kk][n]: mma16_accum computes A @ sMB^T = A @ B
    }
    __syncwarp();
    auto ctx = make_mma16_ctx<MmaTF32_16>(lane);
    mma16_accum(ctx, sMA, sMB, SmemLayout16{});
    CUTE_UNROLL
    for (int v = 0; v < size(ctx.acc); ++v) {
        auto c = ctx.coord(v);
        D[int(get<0>(c)) * kBC + int(get<1>(c))] = alpha * ctx.acc(v);
    }
    __syncwarp();
}

template <typename QKT, typename BetaT, int BT, bool SAFE_GATE, bool IS_VARLEN>
__global__ void __launch_bounds__(128) chunk_kda_fwd_kernel_inter_solve_fused_cuda(
    QKT const* __restrict__ q,
    QKT const* __restrict__ k,
    float const* __restrict__ g,
    BetaT const* __restrict__ beta,
    QKT* __restrict__ Aqk,
    float const* __restrict__ Akkd,
    QKT* __restrict__ Akk,
    float scale,
    int64_t const* __restrict__ cu_seqlens,
    int64_t const* __restrict__ chunk_indices,
    int T,
    int H,
    int HV,
    int K
) {
    constexpr int NC = BT / kBC;
    constexpr int NP = (NC == 4) ? 6 : 1;  // off-diagonal block pairs
    int64_t i_t = blockIdx.x;
    int64_t i_bh = blockIdx.y;
    int i_hv = int(i_bh % HV);
    int i_h = i_hv / (HV / H);

    int64_t bos;
    int Tseq = T;
    if (IS_VARLEN) {
        int64_t i_n = chunk_indices[i_t * 2];
        i_t = chunk_indices[i_t * 2 + 1];
        bos = cu_seqlens[i_n];
        Tseq = int(cu_seqlens[i_n + 1] - bos);
    } else {
        bos = (i_bh / HV) * (int64_t)T;
    }
    if (i_t * BT >= Tseq) return;

    // Phase 1+2 warp scratch: per-warp Aq/Ak/B staging tiles [kBC][kKC2].
    // (Replaces the old block-wide sAq/sAk/sB staging; see below.)
    __shared__ TF32 sW[4][3][kBC * kPadHalf];
    __shared__ float sAkkOff[6][kBC * kBC];  // off-diagonal Akk blocks (beta applied)
    __shared__ float sAi[NC][kBC * kBC];     // diagonal inverse blocks
    __shared__ float sAiX[6][kBC * kBC];     // merged off-diagonal inverse blocks
    __shared__ float sBeta[NC][kBC];

    // Phase-5 scratch overlays sW (dead once phase 1+2 finishes): per-warp
    // tf32 operand staging plus the fp32 partial products of the parallel
    // block-triangular solve.
    struct P5Scratch {
        TF32 sMA[4][kBC * kPad16];
        TF32 sMB[4][kBC * kPad16];
        float T[3][kBC * kBC];  // stage-A products, later t20/t31/t30 partial sums
        float P[6][kBC * kBC];  // independent products of the off-diagonal merge
    };
    static_assert(sizeof(P5Scratch) <= sizeof(TF32) * 4 * 3 * kBC * kPadHalf);

    int tid = threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5;

    int i_tc[NC];
    CUTE_UNROLL
    for (int c = 0; c < NC; ++c) i_tc[c] = int(i_t) * BT + c * kBC;

    for (int idx = tid; idx < NP * kBC * kBC; idx += blockDim.x) {
        sAkkOff[idx / (kBC * kBC)][idx % (kBC * kBC)] = 0.f;
    }
    CUTE_UNROLL
    for (int c = 0; c < NC; ++c) {
        if (tid < kBC) {
            int t = i_tc[c] + tid;
            sBeta[c][tid] = (t < Tseq) ? to_f32(beta[(bos + t) * HV + i_hv]) : 0.f;
        }
    }
    // Diagonal blocks from Akkd (fp32), hoisted ahead of phase 1+2 so the gmem
    // latency overlaps the off-diagonal staging/MMA work.
    for (int idx = tid; idx < NC * kBC * kBC; idx += blockDim.x) {
        int c = idx / (kBC * kBC);
        int i = (idx / kBC) % kBC;
        int j = idx % kBC;
        sAi[c][i * kBC + j] =
            (i_tc[c] + i < Tseq) ? Akkd[((bos + i_tc[c] + i) * HV + i_hv) * (int64_t)kBC + j] : 0.f;
    }
    __syncthreads();

    // Phase 1+2: off-diagonal blocks. Pair p covers block row c, block col cp.
    // Each warp owns whole pairs (p = warp, warp+4) and works independently
    // through its private smem scratch, so stagings/MMAs of different pairs
    // overlap and no block-wide barrier sits inside the loop. The K loop of a
    // pair runs as a pass-level pipeline: the next 8-col pass's raw gmem
    // payload is issued before the current pass is gated/packed/stored, so
    // the gmem latency hides behind the exp2/pack math and the fragment MMAs.
    // Elementwise fallback when K % 8 != 0.
    // Raw gmem payload of one staging pass (one 8-col group of one row).
    struct RawPass {
        float4 gn0, gn1;  // reference-row gates
        float4 gv0, gv1;  // row-block gates
        float4 gp0, gp1;  // col-block gates
        uint4 rq, rk;     // row-block q/k
        uint4 rp;         // col-block k
        unsigned flags;   // bit0: row valid, bit1: col valid, bit2: full 8-col group
    };
    int const n_kchunks = (K + kKC2 - 1) / kKC2;
    bool const kvec = (K % 8) == 0;
    // lane covers col group cg = (lane%4)*8 and rows lane/4, lane/4+8
    int const cg = (lane & 3) * 8;
    int const r0 = lane >> 2;
    for (int p = warp; p < NP; p += 4) {
        int const c = (p == 0) ? 1 : (p < 3 ? 2 : 3);
        int const cp = p - c * (c - 1) / 2;
        if (i_tc[c] >= Tseq) continue;
        int gn_row = i_tc[c];  // reference: first row of the row block
        TF32* sAq = &sW[warp][0][0];
        TF32* sAk = &sW[warp][1][0];
        TF32* sB = &sW[warp][2][0];
        auto ctx_qk = make_mma16_ctx<MmaTF32_16>(lane);
        auto ctx_kk = make_mma16_ctx<MmaTF32_16>(lane);
        float const* gn_base = g + ((bos + gn_row) * HV + i_hv) * (int64_t)K;
        if (kvec) {
            auto load_pass = [&](int kc, int pass, RawPass& R) {
                int const kk = kc * kKC2 + cg;
                int const r = r0 + pass * 8;
                int const t = i_tc[c] + r;    // row block token
                int const tp = i_tc[cp] + r;  // col block token
                bool const ok_r = t < Tseq;
                bool const ok_c = tp < Tseq;
                bool const v8 = (K - kk) >= 8;  // K%8==0: group is full or empty
                R.flags = (ok_r ? 1u : 0u) | (ok_c ? 2u : 0u) | (v8 ? 4u : 0u);
                if (v8) {
                    float4 const* pg = reinterpret_cast<float4 const*>(gn_base + kk);
                    R.gn0 = pg[0]; R.gn1 = pg[1];
                    if (ok_r) {
                        float4 const* pr = reinterpret_cast<float4 const*>(g + ((bos + t) * HV + i_hv) * (int64_t)K + kk);
                        R.gv0 = pr[0]; R.gv1 = pr[1];
                        R.rq = *reinterpret_cast<uint4 const*>(q + ((bos + t) * H + i_h) * (int64_t)K + kk);
                        R.rk = *reinterpret_cast<uint4 const*>(k + ((bos + t) * H + i_h) * (int64_t)K + kk);
                    }
                    if (ok_c) {
                        float4 const* pp = reinterpret_cast<float4 const*>(g + ((bos + tp) * HV + i_hv) * (int64_t)K + kk);
                        R.gp0 = pp[0]; R.gp1 = pp[1];
                        R.rp = *reinterpret_cast<uint4 const*>(k + ((bos + tp) * H + i_h) * (int64_t)K + kk);
                    }
                }
            };
            auto store_pass = [&](RawPass const& R, int pass) {
                int const r = r0 + pass * 8;
                bool const ok_r = (R.flags & 1u) != 0;
                bool const ok_c = (R.flags & 2u) != 0;
                bool const v8 = (R.flags & 4u) != 0;
                float gnv[8] = {}, gv[8] = {}, qv[8] = {}, kv[8] = {}, gvp[8] = {}, kvp[8] = {};
                if (v8) {
                    gnv[0] = R.gn0.x; gnv[1] = R.gn0.y; gnv[2] = R.gn0.z; gnv[3] = R.gn0.w;
                    gnv[4] = R.gn1.x; gnv[5] = R.gn1.y; gnv[6] = R.gn1.z; gnv[7] = R.gn1.w;
                    if (ok_r) {
                        gv[0] = R.gv0.x; gv[1] = R.gv0.y; gv[2] = R.gv0.z; gv[3] = R.gv0.w;
                        gv[4] = R.gv1.x; gv[5] = R.gv1.y; gv[6] = R.gv1.z; gv[7] = R.gv1.w;
                        QKT const* hq = reinterpret_cast<QKT const*>(&R.rq);
                        QKT const* hk = reinterpret_cast<QKT const*>(&R.rk);
                        CUTE_UNROLL
                        for (int j = 0; j < 8; ++j) { qv[j] = to_f32(hq[j]); kv[j] = to_f32(hk[j]); }
                    }
                    if (ok_c) {
                        gvp[0] = R.gp0.x; gvp[1] = R.gp0.y; gvp[2] = R.gp0.z; gvp[3] = R.gp0.w;
                        gvp[4] = R.gp1.x; gvp[5] = R.gp1.y; gvp[6] = R.gp1.z; gvp[7] = R.gp1.w;
                        QKT const* hp = reinterpret_cast<QKT const*>(&R.rp);
                        CUTE_UNROLL
                        for (int j = 0; j < 8; ++j) kvp[j] = to_f32(hp[j]);
                    }
                }
                float oq[8], ok2[8], ob[8];
                CUTE_UNROLL
                for (int j = 0; j < 8; ++j) {
                    float gqn = ok_r ? exp2_ftz(gv[j] - gnv[j]) : 0.f;
                    float gkn = ok_c ? exp2_ftz(gnv[j] - gvp[j]) : 0.f;
                    oq[j] = qv[j] * gqn;
                    ok2[j] = kv[j] * gqn;
                    ob[j] = kvp[j] * gkn;
                }
                int soff = r * kPadHalf + cg;
                *reinterpret_cast<uint4*>(sAq + soff) = pack4_tf32(oq);
                *reinterpret_cast<uint4*>(sAq + soff + 4) = pack4_tf32(oq + 4);
                *reinterpret_cast<uint4*>(sAk + soff) = pack4_tf32(ok2);
                *reinterpret_cast<uint4*>(sAk + soff + 4) = pack4_tf32(ok2 + 4);
                *reinterpret_cast<uint4*>(sB + soff) = pack4_tf32(ob);
                *reinterpret_cast<uint4*>(sB + soff + 4) = pack4_tf32(ob + 4);
            };
            RawPass cur, nxt;
            load_pass(0, 0, cur);
            for (int kc = 0; kc < n_kchunks; ++kc) {
                CUTE_UNROLL
                for (int pass = 0; pass < 2; ++pass) {
                    int const nkc = kc + pass;
                    if (nkc < n_kchunks) load_pass(nkc, (pass + 1) & 1, nxt);
                    store_pass(cur, pass);
                    if (pass == 1) {
                        __syncwarp();
                        mma16_accum(ctx_qk, sAq, sB, SmemLayoutHalf{});
                        mma16_accum(ctx_kk, sAk, sB, SmemLayoutHalf{});
                        __syncwarp();
                    }
                    cur = nxt;
                }
            }
        } else {
            for (int kc = 0; kc < n_kchunks; ++kc) {
                int kk = kc * kKC2 + cg;
                int rem = K - kk;  // valid cols in this 8-col group (may be <= 0)
                CUTE_UNROLL
                for (int pass = 0; pass < 2; ++pass) {
                    int r = r0 + pass * 8;
                    int t = i_tc[c] + r;    // row block token
                    int tp = i_tc[cp] + r;  // col block token
                    bool ok_r = t < Tseq;
                    bool ok_c = tp < Tseq;
                    float gnv[8] = {}, gv[8] = {}, qv[8] = {}, kv[8] = {}, gvp[8] = {}, kvp[8] = {};
                    CUTE_UNROLL
                    for (int j = 0; j < 8; ++j) {
                        bool okk = j < rem;
                        gnv[j] = okk ? gn_base[kk + j] : 0.f;
                        if (ok_r && okk) {
                            gv[j] = g[((bos + t) * HV + i_hv) * (int64_t)K + kk + j];
                            qv[j] = to_f32(q[((bos + t) * H + i_h) * (int64_t)K + kk + j]);
                            kv[j] = to_f32(k[((bos + t) * H + i_h) * (int64_t)K + kk + j]);
                        }
                        if (ok_c && okk) {
                            gvp[j] = g[((bos + tp) * HV + i_hv) * (int64_t)K + kk + j];
                            kvp[j] = to_f32(k[((bos + tp) * H + i_h) * (int64_t)K + kk + j]);
                        }
                    }
                    float oq[8], ok2[8], ob[8];
                    CUTE_UNROLL
                    for (int j = 0; j < 8; ++j) {
                        float gqn = (ok_r && j < rem) ? exp2_ftz(gv[j] - gnv[j]) : 0.f;
                        float gkn = (ok_c && j < rem) ? exp2_ftz(gnv[j] - gvp[j]) : 0.f;
                        oq[j] = qv[j] * gqn;
                        ok2[j] = kv[j] * gqn;
                        ob[j] = kvp[j] * gkn;
                    }
                    int soff = r * kPadHalf + cg;
                    *reinterpret_cast<uint4*>(sAq + soff) = pack4_tf32(oq);
                    *reinterpret_cast<uint4*>(sAq + soff + 4) = pack4_tf32(oq + 4);
                    *reinterpret_cast<uint4*>(sAk + soff) = pack4_tf32(ok2);
                    *reinterpret_cast<uint4*>(sAk + soff + 4) = pack4_tf32(ok2 + 4);
                    *reinterpret_cast<uint4*>(sB + soff) = pack4_tf32(ob);
                    *reinterpret_cast<uint4*>(sB + soff + 4) = pack4_tf32(ob + 4);
                }
                __syncwarp();
                mma16_accum(ctx_qk, sAq, sB, SmemLayoutHalf{});
                mma16_accum(ctx_kk, sAk, sB, SmemLayoutHalf{});
                __syncwarp();
            }
        }
        // Aqk off-diagonal block: scale applied at store
        CUTE_UNROLL
        for (int v = 0; v < size(ctx_qk.acc); ++v) {
            auto crd = ctx_qk.coord(v);
            int i = int(get<0>(crd)), j = int(get<1>(crd));
            if (i_tc[c] + i < Tseq) {
                Aqk[((bos + i_tc[c] + i) * HV + i_hv) * (int64_t)BT + cp * kBC + j] =
                    QKT(ctx_qk.acc(v) * scale);
            }
        }
        // Akk off-diagonal block: beta on rows, kept fp32 in smem
        CUTE_UNROLL
        for (int v = 0; v < size(ctx_kk.acc); ++v) {
            auto crd = ctx_kk.coord(v);
            int i = int(get<0>(crd)), j = int(get<1>(crd));
            sAkkOff[p][i * kBC + j] = ctx_kk.acc(v) * sBeta[c][i];
        }
    }
    __syncthreads();

    // Phase 4: invert diagonal blocks (non safe_gate only; warp c handles block c)
    if (!SAFE_GATE) {
        for (int idx = tid; idx < NC * kBC * kBC; idx += blockDim.x) {
            int i = (idx / kBC) % kBC;
            int j = idx % kBC;
            float* p = &sAi[0][0] + idx;
            *p = (i > j) ? -*p : 0.f;
        }
        __syncthreads();
        if (warp < NC && lane < kBC) {
            int c = warp;
            fwd_subst_16(&sAi[c][0], min(kBC, Tseq - i_tc[c]), lane);
            sAi[c][lane * kBC + lane] += 1.f;
        }
        __syncthreads();
    }

    // Phase 5: merged inverse (block lower-triangular solve), tf32 dots.
    // sAiX index for block pair (c, cp) is c*(c-1)/2 + cp.
    // The solve is a chain of tiny 16x16 GEMMs; independent links run on
    // separate warps through per-warp scratch (sW is dead after phase 1+2),
    // cutting the serial depth from 15 chained GEMMs to 6 stages. Every
    // product and fp32 partial-sum add keeps the v1 operand and accumulation
    // order bit-for-bit (t30 = ((Akk30@Ai00 + Akk31@Ai10) + Akk32@Ai20)).
    auto* p5 = reinterpret_cast<P5Scratch*>(&sW[0][0][0]);
    TF32* sMAw = p5->sMA[warp];
    TF32* sMBw = p5->sMB[warp];
    if constexpr (NC == 4) {
        float (*T)[kBC * kBC] = p5->T;
        float (*P)[kBC * kBC] = p5->P;
        // stage A: diagonal-times-Akk products, one per warp
        if (warp == 0) merge_gemm_16_warp(&sAi[1][0], &sAkkOff[0][0], T[0], 1.f, sMAw, sMBw, lane);
        if (warp == 1) merge_gemm_16_warp(&sAi[2][0], &sAkkOff[2][0], T[1], 1.f, sMAw, sMBw, lane);
        if (warp == 2) merge_gemm_16_warp(&sAi[3][0], &sAkkOff[5][0], T[2], 1.f, sMAw, sMBw, lane);
        if (warp == 3) merge_gemm_16_warp(&sAkkOff[3][0], &sAi[0][0], P[0], 1.f, sMAw, sMBw, lane);
        __syncthreads();
        // stage B: two-link merges Ai10/Ai21/Ai32 plus Akk31@Ai11
        if (warp == 0) merge_gemm_16_warp(T[0], &sAi[0][0], &sAiX[0][0], -1.f, sMAw, sMBw, lane);
        if (warp == 1) merge_gemm_16_warp(T[1], &sAi[1][0], &sAiX[2][0], -1.f, sMAw, sMBw, lane);
        if (warp == 2) merge_gemm_16_warp(T[2], &sAi[2][0], &sAiX[5][0], -1.f, sMAw, sMBw, lane);
        if (warp == 3) merge_gemm_16_warp(&sAkkOff[4][0], &sAi[1][0], P[1], 1.f, sMAw, sMBw, lane);
        __syncthreads();
        // stage C: remaining products of the three-link sums
        if (warp == 0) merge_gemm_16_warp(&sAkkOff[2][0], &sAiX[0][0], P[3], 1.f, sMAw, sMBw, lane);
        if (warp == 1) merge_gemm_16_warp(&sAkkOff[1][0], &sAi[0][0], P[2], 1.f, sMAw, sMBw, lane);
        if (warp == 2) merge_gemm_16_warp(&sAkkOff[5][0], &sAiX[2][0], P[4], 1.f, sMAw, sMBw, lane);
        if (warp == 3) merge_gemm_16_warp(&sAkkOff[4][0], &sAiX[0][0], P[5], 1.f, sMAw, sMBw, lane);
        __syncthreads();
        for (int idx = tid; idx < kBC * kBC; idx += blockDim.x) {
            T[0][idx] = P[2][idx] + P[3][idx];  // t20 = Akk20@Ai00 + Akk21@Ai10
            T[1][idx] = P[1][idx] + P[4][idx];  // t31 = Akk31@Ai11 + Akk32@Ai21
            T[2][idx] = P[0][idx] + P[5][idx];  // t30 partial = Akk30@Ai00 + Akk31@Ai10
        }
        __syncthreads();
        // stage D: Ai20 and Ai31
        if (warp == 0) merge_gemm_16_warp(&sAi[2][0], T[0], &sAiX[1][0], -1.f, sMAw, sMBw, lane);
        if (warp == 2) merge_gemm_16_warp(&sAi[3][0], T[1], &sAiX[4][0], -1.f, sMAw, sMBw, lane);
        __syncthreads();
        // stage E: last product of the t30 sum
        if (warp == 1) merge_gemm_16_warp(&sAkkOff[5][0], &sAiX[1][0], P[2], 1.f, sMAw, sMBw, lane);
        __syncthreads();
        for (int idx = tid; idx < kBC * kBC; idx += blockDim.x) T[2][idx] += P[2][idx];
        __syncthreads();
        // stage F: Ai30
        if (warp == 0) merge_gemm_16_warp(&sAi[3][0], T[2], &sAiX[3][0], -1.f, sMAw, sMBw, lane);
        __syncthreads();
    } else {
        // NC == 2: single off-diagonal block Ai10 = -(Ai11 @ Akk10) @ Ai00
        if (warp == 0) {
            merge_gemm_16_warp(&sAi[1][0], &sAkkOff[0][0], p5->T[0], 1.f, sMAw, sMBw, lane);
            merge_gemm_16_warp(p5->T[0], &sAi[0][0], &sAiX[0][0], -1.f, sMAw, sMBw, lane);
        }
        __syncthreads();
    }

    // Phase 6: store the full block-lower-triangular inverse to Akk (bf16).
    // Each 16-col row segment is contiguous, so a warp-group of 32 threads
    // stores a 16x16 block as 32 16B vectors instead of 256 scalar stores.
    // Pair index p enumerates (c, cp) with p = c*(c+1)/2 + cp.
    {
        int const slot = tid >> 5;         // pair slot per pass
        int const r = (tid >> 1) & (kBC - 1);  // row within the block
        int const h8 = tid & 1;            // which 8-col half of the row
        constexpr int NPAIR = NC * (NC + 1) / 2;
        for (int p = slot; p < NPAIR; p += int(blockDim.x >> 5)) {
            int c = 0;
            while ((c + 1) * (c + 2) / 2 <= p) ++c;
            int cp = p - c * (c + 1) / 2;
            if (i_tc[c] >= Tseq || i_tc[c] + r >= Tseq) continue;
            float const* src = (cp == c) ? &sAi[c][0] : &sAiX[c * (c - 1) / 2 + cp][0];
            QKT vals[8];
            CUTE_UNROLL
            for (int j = 0; j < 8; ++j) vals[j] = QKT(src[r * kBC + h8 * 8 + j]);
            *reinterpret_cast<uint4*>(
                Akk + ((bos + i_tc[c] + r) * HV + i_hv) * (int64_t)BT + cp * kBC + h8 * 8) =
                *reinterpret_cast<uint4 const*>(vals);
        }
    }
}

// ---------------------------------------------------------------------------
// Host launchers
// ---------------------------------------------------------------------------

template <typename QKT, typename BetaT>
void launch_sub_chunk(
    torch::Tensor const& q,
    torch::Tensor const& k,
    torch::Tensor const& g,
    torch::Tensor const& beta,
    torch::Tensor& Aqk,
    torch::Tensor& Akkd,
    float scale,
    int64_t chunk_size,
    VarlenArgs const& varlen,
    int64_t B,
    int64_t T,
    int64_t H,
    int64_t HV,
    int64_t K,
    cudaStream_t stream
) {
    dim3 grid(varlen.NT, B * HV);
    dim3 block(128);

    #define LAUNCH_SUB_CHUNK(BT, IS_VARLEN) \
        chunk_kda_fwd_kernel_intra_sub_chunk_cuda<QKT, BetaT, BT, IS_VARLEN><<<grid, block, 0, stream>>>( \
            reinterpret_cast<QKT const*>(q.data_ptr()), reinterpret_cast<QKT const*>(k.data_ptr()), \
            g.data_ptr<float>(), reinterpret_cast<BetaT const*>(beta.data_ptr()), \
            reinterpret_cast<QKT*>(Aqk.data_ptr()), Akkd.data_ptr<float>(), \
            scale, varlen.cu_seqlens, varlen.chunk_indices, int(T), int(H), int(HV), int(K))

    if (chunk_size == 64) {
        if (varlen.cu_seqlens) { LAUNCH_SUB_CHUNK(64, true); } else { LAUNCH_SUB_CHUNK(64, false); }
    } else {
        if (varlen.cu_seqlens) { LAUNCH_SUB_CHUNK(32, true); } else { LAUNCH_SUB_CHUNK(32, false); }
    }
    #undef LAUNCH_SUB_CHUNK
}

template <typename QKT, typename BetaT>
void launch_token_parallel(
    torch::Tensor const& q,
    torch::Tensor const& k,
    torch::Tensor const& g,
    torch::Tensor const& beta,
    torch::Tensor& Aqk,
    torch::Tensor& Akkd,
    float scale,
    int64_t chunk_size,
    std::optional<torch::Tensor> const& cu_seqlens,
    int64_t B,
    int64_t T,
    int64_t H,
    int64_t HV,
    int64_t K,
    cudaStream_t stream
) {
    int64_t N = cu_seqlens.has_value() ? cu_seqlens->numel() - 1 : B;
    dim3 grid(B * T, (HV + 3) / 4);
    dim3 block(128);
    int64_t const* cu_ptr =
        cu_seqlens.has_value() ? cu_seqlens->data_ptr<int64_t>() : nullptr;

    #define LAUNCH_TOKEN_PARALLEL(BT, IS_VARLEN) \
        chunk_kda_fwd_kernel_intra_token_parallel_cuda<QKT, BetaT, BT, IS_VARLEN><<<grid, block, 0, stream>>>( \
            reinterpret_cast<QKT const*>(q.data_ptr()), reinterpret_cast<QKT const*>(k.data_ptr()), \
            g.data_ptr<float>(), reinterpret_cast<BetaT const*>(beta.data_ptr()), \
            reinterpret_cast<QKT*>(Aqk.data_ptr()), Akkd.data_ptr<float>(), \
            scale, cu_ptr, N, int(T), int(H), int(HV), int(K))

    if (chunk_size == 64) {
        if (cu_ptr) { LAUNCH_TOKEN_PARALLEL(64, true); } else { LAUNCH_TOKEN_PARALLEL(64, false); }
    } else {
        if (cu_ptr) { LAUNCH_TOKEN_PARALLEL(32, true); } else { LAUNCH_TOKEN_PARALLEL(32, false); }
    }
    #undef LAUNCH_TOKEN_PARALLEL
}

template <typename QKT, typename BetaT>
void launch_inter_solve_fused(
    torch::Tensor const& q,
    torch::Tensor const& k,
    torch::Tensor const& g,
    torch::Tensor const& beta,
    torch::Tensor& Aqk,
    torch::Tensor const& Akkd,
    torch::Tensor& Akk,
    float scale,
    int64_t chunk_size,
    bool safe_gate,
    VarlenArgs const& varlen,
    int64_t B,
    int64_t T,
    int64_t H,
    int64_t HV,
    int64_t K,
    cudaStream_t stream
) {
    dim3 grid(varlen.NT, B * HV);
    dim3 block(128);

    #define LAUNCH_FUSED(BT, SAFE_GATE, IS_VARLEN) \
        chunk_kda_fwd_kernel_inter_solve_fused_cuda<QKT, BetaT, BT, SAFE_GATE, IS_VARLEN><<<grid, block, 0, stream>>>( \
            reinterpret_cast<QKT const*>(q.data_ptr()), reinterpret_cast<QKT const*>(k.data_ptr()), \
            g.data_ptr<float>(), reinterpret_cast<BetaT const*>(beta.data_ptr()), \
            reinterpret_cast<QKT*>(Aqk.data_ptr()), Akkd.data_ptr<float>(), \
            reinterpret_cast<QKT*>(Akk.data_ptr()), \
            scale, varlen.cu_seqlens, varlen.chunk_indices, int(T), int(H), int(HV), int(K))

    #define DISPATCH_FUSED_VARLEN(BT, SAFE_GATE) \
        if (varlen.cu_seqlens) { LAUNCH_FUSED(BT, SAFE_GATE, true); } \
        else { LAUNCH_FUSED(BT, SAFE_GATE, false); }
    #define DISPATCH_FUSED_SAFE(BT) \
        if (safe_gate) { DISPATCH_FUSED_VARLEN(BT, true); } \
        else { DISPATCH_FUSED_VARLEN(BT, false); }

    if (chunk_size == 64) { DISPATCH_FUSED_SAFE(64); }
    else { DISPATCH_FUSED_SAFE(32); }

    #undef DISPATCH_FUSED_SAFE
    #undef DISPATCH_FUSED_VARLEN
    #undef LAUNCH_FUSED
}

template <template <typename, typename> class Launcher, typename... Args>
void dispatch_qk_beta(torch::ScalarType qk_ty, torch::ScalarType beta_ty, Args&&... args) {
    // NOTE: template arguments must be spelled with global types here.
    // Instantiating __global__ templates with aliases from this anonymous
    // namespace breaks nvcc's registration stub (it spells the alias's
    // anonymous mangled name in the stub cast, which fails to parse).
    #define DISPATCH_BETA_T(QKT) \
        if (beta_ty == at::kBFloat16) { Launcher<QKT, cutlass::bfloat16_t>()(std::forward<Args>(args)...); } \
        else if (beta_ty == at::kHalf) { Launcher<QKT, cutlass::half_t>()(std::forward<Args>(args)...); } \
        else if (beta_ty == at::kFloat) { Launcher<QKT, float>()(std::forward<Args>(args)...); } \
        else { TORCH_CHECK(false, "unsupported beta dtype"); }

    if (qk_ty == at::kBFloat16) { DISPATCH_BETA_T(cutlass::bfloat16_t); }
    else if (qk_ty == at::kHalf) { DISPATCH_BETA_T(cutlass::half_t); }
    else { TORCH_CHECK(false, "unsupported q/k dtype"); }
    #undef DISPATCH_BETA_T
}

template <typename QKT, typename BetaT>
struct SubChunkLauncher {
    void operator()(torch::Tensor const& q, torch::Tensor const& k, torch::Tensor const& g,
                    torch::Tensor const& beta, torch::Tensor& Aqk, torch::Tensor& Akkd,
                    float scale, int64_t chunk_size, VarlenArgs const& varlen,
                    int64_t B, int64_t T, int64_t H, int64_t HV, int64_t K, cudaStream_t stream) {
        launch_sub_chunk<QKT, BetaT>(q, k, g, beta, Aqk, Akkd, scale, chunk_size, varlen,
                                     B, T, H, HV, K, stream);
    }
};

template <typename QKT, typename BetaT>
struct TokenParallelLauncher {
    void operator()(torch::Tensor const& q, torch::Tensor const& k, torch::Tensor const& g,
                    torch::Tensor const& beta, torch::Tensor& Aqk, torch::Tensor& Akkd,
                    float scale, int64_t chunk_size, std::optional<torch::Tensor> const& cu_seqlens,
                    int64_t B, int64_t T, int64_t H, int64_t HV, int64_t K, cudaStream_t stream) {
        launch_token_parallel<QKT, BetaT>(q, k, g, beta, Aqk, Akkd, scale, chunk_size, cu_seqlens,
                                          B, T, H, HV, K, stream);
    }
};

template <typename QKT, typename BetaT>
struct InterSolveFusedLauncher {
    void operator()(torch::Tensor const& q, torch::Tensor const& k, torch::Tensor const& g,
                    torch::Tensor const& beta, torch::Tensor& Aqk, torch::Tensor const& Akkd,
                    torch::Tensor& Akk, float scale, int64_t chunk_size, bool safe_gate,
                    VarlenArgs const& varlen,
                    int64_t B, int64_t T, int64_t H, int64_t HV, int64_t K, cudaStream_t stream) {
        launch_inter_solve_fused<QKT, BetaT>(q, k, g, beta, Aqk, Akkd, Akk, scale, chunk_size,
                                             safe_gate, varlen, B, T, H, HV, K, stream);
    }
};

}  // namespace kda_train_intra

using kda_train_intra::InterSolveFusedLauncher;
using kda_train_intra::SubChunkLauncher;
using kda_train_intra::TokenParallelLauncher;
using kda_train_intra::check_intra_inputs;
using kda_train_intra::dispatch_qk_beta;
using kda_train_intra::resolve_varlen_intra;

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
) {
    check_intra_inputs(q, k, g, beta, Aqk, Akkd, chunk_size);
    int64_t B = k.size(0), T = k.size(1), H = k.size(2), K = k.size(3), HV = g.size(2);
    auto varlen = resolve_varlen_intra(cu_seqlens, chunk_indices, B, T, chunk_size);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    dispatch_qk_beta<SubChunkLauncher>(q.scalar_type(), beta.scalar_type(),
                                       q, k, g, beta, Aqk, Akkd, float(scale), chunk_size,
                                       varlen, B, T, H, HV, K, stream);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

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
) {
    check_intra_inputs(q, k, g, beta, Aqk, Akkd, chunk_size);
    int64_t B = k.size(0), T = k.size(1), H = k.size(2), K = k.size(3), HV = g.size(2);
    if (cu_seqlens.has_value()) {
        TORCH_CHECK(cu_seqlens->dtype() == torch::kLong && cu_seqlens->is_cuda() &&
                    cu_seqlens->is_contiguous(), "cu_seqlens must be contiguous int64 CUDA tensor");
        TORCH_CHECK(B == 1, "B must be 1 when cu_seqlens is provided");
    }
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    dispatch_qk_beta<TokenParallelLauncher>(q.scalar_type(), beta.scalar_type(),
                                            q, k, g, beta, Aqk, Akkd, float(scale), chunk_size,
                                            cu_seqlens, B, T, H, HV, K, stream);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

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
) {
    check_intra_inputs(q, k, g, beta, Aqk, Akkd, chunk_size);
    TORCH_CHECK(Akk.is_cuda() && Akk.is_contiguous() && Akk.scalar_type() == k.scalar_type());
    TORCH_CHECK(Akk.dim() == 4 && Akk.size(0) == k.size(0) && Akk.size(1) == k.size(1) &&
                Akk.size(2) == g.size(2) && Akk.size(3) == chunk_size, "Akk must be [B, T, HV, BT]");
    int64_t B = k.size(0), T = k.size(1), H = k.size(2), K = k.size(3), HV = g.size(2);
    auto varlen = resolve_varlen_intra(cu_seqlens, chunk_indices, B, T, chunk_size);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    dispatch_qk_beta<InterSolveFusedLauncher>(q.scalar_type(), beta.scalar_type(),
                                              q, k, g, beta, Aqk, Akkd, Akk, float(scale),
                                              chunk_size, safe_gate, varlen, B, T, H, HV, K, stream);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
