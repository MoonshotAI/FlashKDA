// KDA chunked state-recurrence kernels: forward h and backward dhu.
// Replicates fla/ops/common/chunk_delta_h.py restricted to the KDA call shape:
// USE_G=False, USE_GK=True, SAVE_NEW_VALUE=True, chunk_size=64, bf16 operands.
//
// Each thread block owns one (sequence, value head, BV-wide V tile) and walks the
// sequence's chunks serially, mirroring the Triton program. For K <= 128 the
// pipelined kernels (kda_*_pipe_kernel) keep the [K, BV] fp32 state in registers
// as the accumulator fragment of the state-update MMA across the whole chunk
// loop; per chunk the bf16 state copy is staged to smem once (paired 32-bit
// stores, padded rows against bank conflicts) and the operand fragments are
// loaded with ldmatrix. All gmem tiles move through 16B cp.async into padded
// smem with one-chunk-ahead prefetch (phase-shifted single buffering), and all
// gmem outputs are staged through smem for fully coalesced 16B stores.
// For K > 128 the pipelined backward no longer fits the 99KB opt-in dynamic
// smem limit of consumer Blackwell (sm_120), so the legacy smem-state kernels
// below still serve those shapes.

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>

#include <cute/tensor.hpp>

#include <cutlass/bfloat16.h>

#include "common.cuh"

namespace {

using cute::Int;
using cute::Layout;
using cute::Shape;
using cute::Stride;
using cute::Tensor;
using cute::_1;
using cute::_4;
using cute::get;
using cute::make_coord;
using cute::make_identity_tensor;
using cute::make_layout;
using cute::make_shape;
using cute::make_smem_ptr;
using cute::make_stride;
using cute::make_tensor;
using cute::make_tiled_copy_A;
using cute::make_tiled_copy_B;
using cute::make_tiled_mma;

using BF16 = cutlass::bfloat16_t;

constexpr int kBT = 64;   // chunk size
constexpr int kThreads = 128;
constexpr int kPad = 8;   // smem row padding in elements (16B) against bank conflicts

using MmaAtom = cute::SM80_16x8x16_F32BF16BF16F32_TN;
using LdsmN = cute::Copy_Atom<cute::SM75_U32x4_LDSM_N, BF16>;
using LdsmT = cute::Copy_Atom<cute::SM75_U16x8_LDSM_T, BF16>;
// x2 variant for B operands: with Layout<Shape<_4,_1,_1>> the per-warp B tile is
// 8(N) x 16(K) = 4 vals/thread, too small for the x4 atom.
using LdsmT2 = cute::Copy_Atom<cute::SM75_U16x4_LDSM_T, BF16>;

// ---------------------------------------------------------------------------
// cp.async helpers

__device__ __forceinline__ void cp_async16(BF16* dst, BF16 const* src, bool full) {
    uint32_t s = cute::cast_smem_ptr_to_uint(dst);
    int sz = full ? 16 : 0;  // src-size 0 zero-fills, used for masked rows/columns
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
                 :: "r"(s), "l"(src), "r"(sz) : "memory");
}

__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" ::: "memory");
}

template <int N>
__device__ __forceinline__ void cp_async_wait() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N) : "memory");
}

__device__ __forceinline__ uint32_t pack_bf16(float lo, float hi) {
    BF16 a(lo), b(hi);
    return uint32_t(a.storage) | (uint32_t(b.storage) << 16);
}

// ---------------------------------------------------------------------------
// pipelined-kernel shared memory

template <int Kp, int BV>
struct FwdPipeSmem {
    static constexpr int CP = Kp + kPad;
    static constexpr int VP = BV + kPad;
    alignas(128) BF16 s_w[kBT * CP];    // [BT][Kp] natural
    alignas(128) BF16 s_kg[kBT * CP];   // [BT][Kp] natural
    alignas(128) BF16 s_u[kBT * VP];    // [BT][BV] natural
    alignas(128) BF16 s_hb[Kp * VP];    // [Kp][BV] bf16 state copy
    alignas(128) BF16 s_vn[kBT * VP];   // [BT][BV] v_new
};

template <int Kp, int BV>
struct BwdPipeSmem {
    static constexpr int CP = Kp + kPad;
    static constexpr int VP = BV + kPad;
    alignas(128) BF16 s_kg[kBT * CP];   // [BT][Kp] natural
    alignas(128) BF16 s_qg[kBT * CP];   // [BT][Kp] natural
    alignas(128) BF16 s_w[kBT * CP];    // [BT][Kp] natural
    alignas(128) BF16 s_do[kBT * VP];   // [BT][BV] natural
    alignas(128) BF16 s_dv[kBT * VP];   // [BT][BV] natural
    alignas(128) BF16 s_dhb[Kp * VP];   // [Kp][BV] bf16 state-gradient copy
    alignas(128) BF16 s_dv2[kBT * VP];  // [BT][BV] holds NEGATED dv2 (see kernel)
};

// ---------------------------------------------------------------------------
// pipelined-kernel staging helpers (all threads of the block cooperate)

// Queue a [R][C] gmem tile (row stride `row_stride`) into padded smem [R][CP]
// with cp.async, zero-filling rows >= r_valid and columns >= c_valid via
// src-size 0. Falls back to scalar stores when 16B alignment does not hold.
template <int R, int C, int CP>
__device__ void stage_tile_async(BF16* dst, BF16 const* src, BF16 const* src_safe,
                                 int64_t row_stride, int r_valid, int c_valid,
                                 bool vec_ok, int tid) {
    constexpr int kCG = C / 8;
    if (vec_ok && (c_valid % 8) == 0) {
        for (int idx = tid; idx < R * kCG; idx += kThreads) {
            int r = idx / kCG, c = (idx % kCG) * 8;
            bool full = (r < r_valid) && (c < c_valid);
            BF16 const* g = full ? src + (int64_t)r * row_stride + c : src_safe;
            cp_async16(dst + r * CP + c, g, full);
        }
    } else {
        for (int idx = tid; idx < R * C; idx += kThreads) {
            int r = idx / C, c = idx % C;
            dst[r * CP + c] = (r < r_valid && c < c_valid)
                            ? src[(int64_t)r * row_stride + c] : BF16(0.f);
        }
    }
}

// Store a padded smem tile [R][CP] to gmem (row stride `row_stride`), masked to
// r < r_valid and c < c_valid, with 16B vectorized accesses when aligned.
// NEG flips the bf16 sign bit (exact negation) on the way out.
template <int R, int C, int CP, typename OutT, bool NEG = false>
__device__ void store_tile_gmem(OutT* dst, BF16 const* ssm, int64_t row_stride,
                                int r_valid, int c_valid, bool vec_ok, int tid) {
    constexpr int kCG = C / 8;
    if (vec_ok && (c_valid % 8) == 0) {
        for (int idx = tid; idx < R * kCG; idx += kThreads) {
            int r = idx / kCG, c = (idx % kCG) * 8;
            if (r >= r_valid) break;
            if (c + 8 <= c_valid) {
                uint4 v = *reinterpret_cast<uint4 const*>(ssm + r * CP + c);
                if (NEG) {
                    v.x ^= 0x80008000u; v.y ^= 0x80008000u;
                    v.z ^= 0x80008000u; v.w ^= 0x80008000u;
                }
                *reinterpret_cast<uint4*>(dst + (int64_t)r * row_stride + c) = v;
            } else {
                for (int j = 0; j < 8 && c + j < c_valid; ++j) {
                    float val = to_f32(ssm[r * CP + c + j]);
                    dst[(int64_t)r * row_stride + c + j] = OutT(NEG ? -val : val);
                }
            }
        }
    } else {
        for (int idx = tid; idx < R * C; idx += kThreads) {
            int r = idx / C, c = idx % C;
            if (r < r_valid && c < c_valid) {
                float val = to_f32(ssm[r * CP + c]);
                dst[(int64_t)r * row_stride + c] = OutT(NEG ? -val : val);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// kernel params

struct FwdParams {
    BF16 const* kg;
    BF16 const* w;
    BF16 const* u;
    float const* gk;
    BF16* h;
    BF16* v_new;
    float const* h0;
    float* ht;
    int64_t const* cu_seqlens;
    int64_t const* chunk_offsets;
    int T, H, HV, K, V, NV;
    bool varlen, state_v_first;
};

struct BwdParams {
    BF16 const* qg;
    BF16 const* kg;
    BF16 const* w;
    float const* gk;
    BF16 const* do_;
    BF16 const* dv;
    BF16* dv2;
    BF16* dh;
    float* dh0;
    float const* dht;
    int64_t const* cu_seqlens;
    int64_t const* chunk_offsets;
    float scale;
    int T, H, HV, K, V, NV;
    bool varlen, state_v_first;
};

// ---------------------------------------------------------------------------
// pipelined forward: per chunk, h[i_t] = S; v_new = u - w @ S;
// S = diag(exp2(gk_last)) S + kg^T @ v_new
//
// S lives in the accumulator fragment of the update MMA across the chunk loop.
// Per chunk: prefetch (w, u, kg) of chunk i+1 with cp.async; the update GEMM
// accumulates directly into the state fragment after the elementwise decay.

template <int Kp, int BV>
__global__ void __launch_bounds__(kThreads) kda_fwd_h_pipe_kernel(FwdParams const p) {
    constexpr int CP = Kp + kPad;
    constexpr int VP = BV + kPad;
    constexpr int kMM = Kp / 64;  // MMA_M of the state-update GEMM
    extern __shared__ __align__(128) unsigned char smem_raw[];
    auto& sm = *reinterpret_cast<FwdPipeSmem<Kp, BV>*>(smem_raw);
    const int tid = threadIdx.x;

    const int i_v = blockIdx.x % p.NV;
    const int64_t i_nh = blockIdx.x / p.NV;
    const int64_t i_n = i_nh / p.HV;
    const int i_hv = int(i_nh % p.HV);
    const int i_hq = i_hv / (p.HV / p.H);

    int64_t bos;
    int T, NT;
    int64_t boh;
    if (p.varlen) {
        bos = p.cu_seqlens[i_n];
        T = int(p.cu_seqlens[i_n + 1] - bos);
        NT = (T + kBT - 1) / kBT;
        boh = p.chunk_offsets[i_n];
    } else {
        bos = i_n * p.T;
        T = p.T;
        NT = (T + kBT - 1) / kBT;
        boh = i_n * NT;
    }

    BF16 const* kg = p.kg + (bos * p.H + i_hq) * (int64_t)p.K;
    BF16 const* w = p.w + (bos * p.HV + i_hv) * (int64_t)p.K;
    BF16 const* u = p.u + (bos * p.HV + i_hv) * (int64_t)p.V;
    float const* gk = p.gk + (bos * p.HV + i_hv) * (int64_t)p.K;
    BF16* h = p.h + (boh * p.HV + i_hv) * (int64_t)p.K * p.V;
    BF16* v_new = p.v_new + (bos * p.HV + i_hv) * (int64_t)p.V;
    const int v0 = i_v * BV;

    const bool vec_k = (p.K % 8) == 0;
    const bool vec_v = (p.V % 8) == 0;
    const int cv = min(BV, p.V - v0);

    auto mma = make_tiled_mma(MmaAtom{}, Layout<Shape<_4, _1, _1>>{});
    auto thr_mma = mma.get_thread_slice(tid);

    Tensor sW = make_tensor(make_smem_ptr(sm.s_w),
                            make_layout(make_shape(Int<kBT>{}, Int<Kp>{}), make_stride(Int<CP>{}, _1{})));
    Tensor sU = make_tensor(make_smem_ptr(sm.s_u),
                            make_layout(make_shape(Int<kBT>{}, Int<BV>{}), make_stride(Int<VP>{}, _1{})));
    Tensor sVN = make_tensor(make_smem_ptr(sm.s_vn),
                             make_layout(make_shape(Int<kBT>{}, Int<BV>{}), make_stride(Int<VP>{}, _1{})));
    // GEMM operand views: the state copy as B[N=BV, K=Kp], kg as A[M=Kp, K=BT],
    // v_new as B[N=BV, K=BT], all with the contraction dim strided (ldsm .trans).
    Tensor sHBb = make_tensor(make_smem_ptr(sm.s_hb),
                              make_layout(make_shape(Int<BV>{}, Int<Kp>{}), make_stride(_1{}, Int<VP>{})));
    Tensor sKGa = make_tensor(make_smem_ptr(sm.s_kg),
                              make_layout(make_shape(Int<Kp>{}, Int<kBT>{}), make_stride(_1{}, Int<CP>{})));
    Tensor sVNb = make_tensor(make_smem_ptr(sm.s_vn),
                              make_layout(make_shape(Int<BV>{}, Int<kBT>{}), make_stride(_1{}, Int<VP>{})));

    Tensor state = cute::partition_fragment_C(mma, Shape<Int<Kp>, Int<BV>>{});
    Tensor tCcS = thr_mma.partition_C(make_identity_tensor(make_shape(Int<Kp>{}, Int<BV>{})));
    Tensor tCcV = thr_mma.partition_C(make_identity_tensor(make_shape(Int<kBT>{}, Int<BV>{})));

    // initial state, fragment-direct
    {
        float const* h0 = p.h0 == nullptr ? nullptr : p.h0 + i_nh * (int64_t)p.K * p.V;
        for (int i = 0; i < int(size(state)); ++i) {
            int k = get<0>(tCcS(i)), n = get<1>(tCcS(i));
            int vg = v0 + n;
            float val = 0.f;
            if (h0 != nullptr && k < p.K && vg < p.V) {
                val = h0[p.state_v_first ? (int64_t)vg * p.K + k : (int64_t)k * p.V + vg];
            }
            state(i) = val;
        }
    }
    // distinct state rows this thread holds: index i -> row j = (i>>1 & 1) + 2*((i>>2) % kMM)
    int row_k[2 * kMM];
    for (int m = 0; m < kMM; ++m) {
        for (int half = 0; half < 2; ++half) {
            row_k[m * 2 + half] = get<0>(tCcS(make_coord(make_coord(0, half), m, 0)));
        }
    }

    auto stage_wu = [&](int i_t) {
        int tv = i_t < NT ? min(kBT, T - i_t * kBT) : 0;
        int64_t t0 = (int64_t)i_t * kBT;
        BF16 const* wsrc = i_t < NT ? w + t0 * (int64_t)p.HV * p.K : w;
        BF16 const* usrc = i_t < NT ? u + t0 * (int64_t)p.HV * p.V + v0 : u;
        stage_tile_async<kBT, Kp, CP>(sm.s_w, wsrc, w, (int64_t)p.HV * p.K, tv, p.K, vec_k, tid);
        stage_tile_async<kBT, BV, VP>(sm.s_u, usrc, u, (int64_t)p.HV * p.V, tv, cv, vec_v, tid);
        cp_async_commit();
    };
    auto stage_kg = [&](int i_t) {
        int tv = i_t < NT ? min(kBT, T - i_t * kBT) : 0;
        int64_t t0 = (int64_t)i_t * kBT;
        BF16 const* ksrc = i_t < NT ? kg + t0 * (int64_t)p.H * p.K : kg;
        stage_tile_async<kBT, Kp, CP>(sm.s_kg, ksrc, kg, (int64_t)p.H * p.K, tv, p.K, vec_k, tid);
        cp_async_commit();
    };

    stage_wu(0);
    stage_kg(0);

    for (int i_t = 0; i_t < NT; ++i_t) {
        const int t_valid = min(kBT, T - i_t * kBT);
        const int64_t t0 = (int64_t)i_t * kBT;

        cp_async_wait<1>();  // w, u of this chunk (kg may still be in flight)
        __syncthreads();

        // decay factors for the state update at the end of this chunk
        const int last = min((i_t + 1) * kBT, T) - 1;
        float const* gk_last = gk + (int64_t)last * p.HV * p.K;
        float gval[2 * kMM];
        for (int j = 0; j < 2 * kMM; ++j) {
            gval[j] = row_k[j] < p.K ? exp2f(gk_last[row_k[j]]) : 1.0f;
        }

        // bf16 state copy: B operand of the v_new GEMM and the h[i_t] output
        for (int i = 0; i < int(size(state)); i += 2) {
            int k = get<0>(tCcS(i)), n = get<1>(tCcS(i));
            *reinterpret_cast<uint32_t*>(&sm.s_hb[k * VP + n]) = pack_bf16(state(i), state(i + 1));
        }
        __syncthreads();

        // v_new = w @ h operand fragments
        Tensor tCrW = thr_mma.partition_fragment_A(sW);
        Tensor tCrHB = thr_mma.partition_fragment_B(sHBb);
        auto s2r_w = make_tiled_copy_A(LdsmN{}, mma);
        auto thr_w = s2r_w.get_thread_slice(tid);
        copy(s2r_w, thr_w.partition_S(sW), thr_w.retile_D(tCrW));
        auto s2r_hb = make_tiled_copy_B(LdsmT2{}, mma);
        auto thr_hb = s2r_hb.get_thread_slice(tid);
        copy(s2r_hb, thr_hb.partition_S(sHBb), thr_hb.retile_D(tCrHB));

        // h[i_t] = state at chunk entry (bf16)
        BF16* h_t = h + (int64_t)i_t * p.HV * p.K * p.V;
        if (p.state_v_first) {
            for (int idx = tid; idx < Kp * BV; idx += kThreads) {
                int k = idx / BV, n = idx % BV, vg = v0 + n;
                if (k < p.K && vg < p.V) h_t[(int64_t)vg * p.K + k] = sm.s_hb[k * VP + n];
            }
        } else {
            store_tile_gmem<Kp, BV, VP, BF16>(h_t + v0, sm.s_hb, p.V, p.K, cv, vec_v, tid);
        }

        Tensor acc = cute::partition_fragment_C(mma, Shape<Int<kBT>, Int<BV>>{});
        clear(acc);
        gemm(thr_mma, tCrW, tCrHB, acc);

        // v_new = u - w @ h; u is zero-filled beyond t_valid, so rows past the
        // sequence end yield exactly 0 (required by the state-update GEMM).
        Tensor tCsU = thr_mma.partition_C(sU);
        for (int i = 0; i < int(size(acc)); i += 2) {
            int m = get<0>(tCcV(i)), n = get<1>(tCcV(i));
            float vlo = to_f32(tCsU(i)) - acc(i);
            float vhi = to_f32(tCsU(i + 1)) - acc(i + 1);
            *reinterpret_cast<uint32_t*>(&sm.s_vn[m * VP + n]) = pack_bf16(vlo, vhi);
        }
        __syncthreads();  // s_w consumed by the GEMM; s_vn complete

        stage_wu(i_t + 1);
        store_tile_gmem<kBT, BV, VP, BF16>(v_new + t0 * (int64_t)p.HV * p.V + v0,
                                           sm.s_vn, (int64_t)p.HV * p.V, t_valid, cv, vec_v, tid);

        cp_async_wait<1>();  // kg of this chunk (the new w/u group may fly)
        __syncthreads();

        // h = diag(exp2(gk_last)) h + kg^T @ v_new, accumulated into the state
        Tensor tCrKG = thr_mma.partition_fragment_A(sKGa);
        Tensor tCrVN = thr_mma.partition_fragment_B(sVNb);
        auto s2r_kg = make_tiled_copy_A(LdsmT{}, mma);
        auto thr_kg = s2r_kg.get_thread_slice(tid);
        copy(s2r_kg, thr_kg.partition_S(sKGa), thr_kg.retile_D(tCrKG));
        auto s2r_vn = make_tiled_copy_B(LdsmT2{}, mma);
        auto thr_vn = s2r_vn.get_thread_slice(tid);
        copy(s2r_vn, thr_vn.partition_S(sVNb), thr_vn.retile_D(tCrVN));

        for (int i = 0; i < int(size(state)); ++i) {
            state(i) *= gval[((i >> 1) & 1) + 2 * ((i >> 2) % kMM)];
        }
        gemm(thr_mma, tCrKG, tCrVN, state);
        __syncthreads();  // s_kg and s_vn consumed
        stage_kg(i_t + 1);
    }

    if (p.ht != nullptr) {
        float* ht = p.ht + i_nh * (int64_t)p.K * p.V;
        for (int i = 0; i < int(size(state)); ++i) {
            int k = get<0>(tCcS(i)), n = get<1>(tCcS(i)), vg = v0 + n;
            if (k < p.K && vg < p.V) {
                ht[p.state_v_first ? (int64_t)vg * p.K + k : (int64_t)k * p.V + vg] = state(i);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// pipelined backward: per chunk (in reverse), dh[i_t] = S;
// dv2 = dv + kg @ S; S = S * exp2(gk_last) + scale * qg^T @ do - w^T @ dv2
//
// S_dv2 holds NEGATED dv2 (sign flip is exact in bf16) so the w^T @ dv2 term
// accumulates directly into the state fragment; the gmem dv2 store flips the
// sign back. The qg^T @ do term uses a temporary fragment for the scale.

template <int Kp, int BV>
__global__ void __launch_bounds__(kThreads) kda_bwd_dhu_pipe_kernel(BwdParams const p) {
    constexpr int CP = Kp + kPad;
    constexpr int VP = BV + kPad;
    constexpr int kMM = Kp / 64;
    extern __shared__ __align__(128) unsigned char smem_raw[];
    auto& sm = *reinterpret_cast<BwdPipeSmem<Kp, BV>*>(smem_raw);
    const int tid = threadIdx.x;

    const int i_v = blockIdx.x % p.NV;
    const int64_t i_nh = blockIdx.x / p.NV;
    const int64_t i_n = i_nh / p.HV;
    const int i_hv = int(i_nh % p.HV);
    const int i_hq = i_hv / (p.HV / p.H);

    int64_t bos;
    int T, NT;
    int64_t boh;
    if (p.varlen) {
        bos = p.cu_seqlens[i_n];
        T = int(p.cu_seqlens[i_n + 1] - bos);
        NT = (T + kBT - 1) / kBT;
        boh = p.chunk_offsets[i_n];
    } else {
        bos = i_n * p.T;
        T = p.T;
        NT = (T + kBT - 1) / kBT;
        boh = i_n * NT;
    }

    BF16 const* qg = p.qg + (bos * p.H + i_hq) * (int64_t)p.K;
    BF16 const* kg = p.kg + (bos * p.H + i_hq) * (int64_t)p.K;
    BF16 const* w = p.w + (bos * p.HV + i_hv) * (int64_t)p.K;
    float const* gk = p.gk + (bos * p.HV + i_hv) * (int64_t)p.K;
    BF16 const* do_ = p.do_ + (bos * p.HV + i_hv) * (int64_t)p.V;
    BF16 const* dv = p.dv + (bos * p.HV + i_hv) * (int64_t)p.V;
    BF16* dv2 = p.dv2 + (bos * p.HV + i_hv) * (int64_t)p.V;
    BF16* dh = p.dh + (boh * p.HV + i_hv) * (int64_t)p.K * p.V;
    const int v0 = i_v * BV;

    const bool vec_k = (p.K % 8) == 0;
    const bool vec_v = (p.V % 8) == 0;
    const int cv = min(BV, p.V - v0);

    auto mma = make_tiled_mma(MmaAtom{}, Layout<Shape<_4, _1, _1>>{});
    auto thr_mma = mma.get_thread_slice(tid);

    Tensor sKG = make_tensor(make_smem_ptr(sm.s_kg),
                             make_layout(make_shape(Int<kBT>{}, Int<Kp>{}), make_stride(Int<CP>{}, _1{})));
    Tensor sDV = make_tensor(make_smem_ptr(sm.s_dv),
                             make_layout(make_shape(Int<kBT>{}, Int<BV>{}), make_stride(Int<VP>{}, _1{})));
    // GEMM operand views (all contraction-strided, ldsm .trans), except kg which
    // is the K-major A operand of the dv2 GEMM.
    Tensor sDHBb = make_tensor(make_smem_ptr(sm.s_dhb),
                               make_layout(make_shape(Int<BV>{}, Int<Kp>{}), make_stride(_1{}, Int<VP>{})));
    Tensor sQGa = make_tensor(make_smem_ptr(sm.s_qg),
                              make_layout(make_shape(Int<Kp>{}, Int<kBT>{}), make_stride(_1{}, Int<CP>{})));
    Tensor sWa = make_tensor(make_smem_ptr(sm.s_w),
                             make_layout(make_shape(Int<Kp>{}, Int<kBT>{}), make_stride(_1{}, Int<CP>{})));
    Tensor sDOb = make_tensor(make_smem_ptr(sm.s_do),
                              make_layout(make_shape(Int<BV>{}, Int<kBT>{}), make_stride(_1{}, Int<VP>{})));
    Tensor sDV2b = make_tensor(make_smem_ptr(sm.s_dv2),
                               make_layout(make_shape(Int<BV>{}, Int<kBT>{}), make_stride(_1{}, Int<VP>{})));

    Tensor state = cute::partition_fragment_C(mma, Shape<Int<Kp>, Int<BV>>{});
    Tensor tCcS = thr_mma.partition_C(make_identity_tensor(make_shape(Int<Kp>{}, Int<BV>{})));
    Tensor tCcV = thr_mma.partition_C(make_identity_tensor(make_shape(Int<kBT>{}, Int<BV>{})));

    {
        float const* dht = p.dht == nullptr ? nullptr : p.dht + i_nh * (int64_t)p.K * p.V;
        for (int i = 0; i < int(size(state)); ++i) {
            int k = get<0>(tCcS(i)), n = get<1>(tCcS(i));
            int vg = v0 + n;
            float val = 0.f;
            if (dht != nullptr && k < p.K && vg < p.V) {
                val = dht[p.state_v_first ? (int64_t)vg * p.K + k : (int64_t)k * p.V + vg];
            }
            state(i) = val;
        }
    }
    int row_k[2 * kMM];
    for (int m = 0; m < kMM; ++m) {
        for (int half = 0; half < 2; ++half) {
            row_k[m * 2 + half] = get<0>(tCcS(make_coord(make_coord(0, half), m, 0)));
        }
    }

    auto stage_kgdv = [&](int i_t) {
        bool ok = i_t >= 0;
        int tv = ok ? min(kBT, T - i_t * kBT) : 0;
        int64_t t0 = (int64_t)i_t * kBT;
        BF16 const* ksrc = ok ? kg + t0 * (int64_t)p.H * p.K : kg;
        BF16 const* dvsrc = ok ? dv + t0 * (int64_t)p.HV * p.V + v0 : dv;
        stage_tile_async<kBT, Kp, CP>(sm.s_kg, ksrc, kg, (int64_t)p.H * p.K, tv, p.K, vec_k, tid);
        stage_tile_async<kBT, BV, VP>(sm.s_dv, dvsrc, dv, (int64_t)p.HV * p.V, tv, cv, vec_v, tid);
        cp_async_commit();
    };
    auto stage_qgdo = [&](int i_t) {
        bool ok = i_t >= 0;
        int tv = ok ? min(kBT, T - i_t * kBT) : 0;
        int64_t t0 = (int64_t)i_t * kBT;
        BF16 const* qsrc = ok ? qg + t0 * (int64_t)p.H * p.K : qg;
        BF16 const* dosrc = ok ? do_ + t0 * (int64_t)p.HV * p.V + v0 : do_;
        stage_tile_async<kBT, Kp, CP>(sm.s_qg, qsrc, qg, (int64_t)p.H * p.K, tv, p.K, vec_k, tid);
        stage_tile_async<kBT, BV, VP>(sm.s_do, dosrc, do_, (int64_t)p.HV * p.V, tv, cv, vec_v, tid);
        cp_async_commit();
    };
    auto stage_w = [&](int i_t) {
        bool ok = i_t >= 0;
        int tv = ok ? min(kBT, T - i_t * kBT) : 0;
        int64_t t0 = (int64_t)i_t * kBT;
        BF16 const* wsrc = ok ? w + t0 * (int64_t)p.HV * p.K : w;
        stage_tile_async<kBT, Kp, CP>(sm.s_w, wsrc, w, (int64_t)p.HV * p.K, tv, p.K, vec_k, tid);
        cp_async_commit();
    };

    stage_kgdv(NT - 1);
    stage_qgdo(NT - 1);
    stage_w(NT - 1);

    for (int i_t = NT - 1; i_t >= 0; --i_t) {
        const int t_valid = min(kBT, T - i_t * kBT);
        const int64_t t0 = (int64_t)i_t * kBT;

        cp_async_wait<2>();  // kg, dv of this chunk
        __syncthreads();

        const int last = min((i_t + 1) * kBT, T) - 1;
        float const* gk_last = gk + (int64_t)last * p.HV * p.K;
        float gval[2 * kMM];
        for (int j = 0; j < 2 * kMM; ++j) {
            gval[j] = row_k[j] < p.K ? exp2f(gk_last[row_k[j]]) : 1.0f;
        }

        // bf16 state-gradient copy: B operand of the dv2 GEMM and the dh[i_t] output
        for (int i = 0; i < int(size(state)); i += 2) {
            int k = get<0>(tCcS(i)), n = get<1>(tCcS(i));
            *reinterpret_cast<uint32_t*>(&sm.s_dhb[k * VP + n]) = pack_bf16(state(i), state(i + 1));
        }
        __syncthreads();

        // dv2 = kg @ dh operand fragments
        Tensor tCrKG = thr_mma.partition_fragment_A(sKG);
        Tensor tCrDHB = thr_mma.partition_fragment_B(sDHBb);
        auto s2r_kg = make_tiled_copy_A(LdsmN{}, mma);
        auto thr_kg = s2r_kg.get_thread_slice(tid);
        copy(s2r_kg, thr_kg.partition_S(sKG), thr_kg.retile_D(tCrKG));
        auto s2r_dhb = make_tiled_copy_B(LdsmT2{}, mma);
        auto thr_dhb = s2r_dhb.get_thread_slice(tid);
        copy(s2r_dhb, thr_dhb.partition_S(sDHBb), thr_dhb.retile_D(tCrDHB));

        // dh[i_t] = state gradient at chunk entry (bf16)
        BF16* dh_t = dh + (int64_t)i_t * p.HV * p.K * p.V;
        if (p.state_v_first) {
            for (int idx = tid; idx < Kp * BV; idx += kThreads) {
                int k = idx / BV, n = idx % BV, vg = v0 + n;
                if (k < p.K && vg < p.V) dh_t[(int64_t)vg * p.K + k] = sm.s_dhb[k * VP + n];
            }
        } else {
            store_tile_gmem<Kp, BV, VP, BF16>(dh_t + v0, sm.s_dhb, p.V, p.K, cv, vec_v, tid);
        }

        Tensor acc = cute::partition_fragment_C(mma, Shape<Int<kBT>, Int<BV>>{});
        clear(acc);
        gemm(thr_mma, tCrKG, tCrDHB, acc);

        // s_dv2 <- -(dv + kg @ dh); the sign is flipped back on the gmem store.
        Tensor tCsDV = thr_mma.partition_C(sDV);
        for (int i = 0; i < int(size(acc)); i += 2) {
            int m = get<0>(tCcV(i)), n = get<1>(tCcV(i));
            float vlo = acc(i) + to_f32(tCsDV(i));
            float vhi = acc(i + 1) + to_f32(tCsDV(i + 1));
            *reinterpret_cast<uint32_t*>(&sm.s_dv2[m * VP + n]) = pack_bf16(-vlo, -vhi);
        }
        __syncthreads();  // s_kg / s_dv / s_dhb consumed; s_dv2 complete

        stage_kgdv(i_t - 1);
        store_tile_gmem<kBT, BV, VP, BF16, true>(dv2 + t0 * (int64_t)p.HV * p.V + v0,
                                                 sm.s_dv2, (int64_t)p.HV * p.V, t_valid, cv, vec_v, tid);

        cp_async_wait<1>();  // qg, do and w of this chunk
        __syncthreads();

        // dh = dh * exp2(gk_last) + scale * qg^T @ do
        Tensor tmp = cute::partition_fragment_C(mma, Shape<Int<Kp>, Int<BV>>{});
        clear(tmp);
        Tensor tCrQG = thr_mma.partition_fragment_A(sQGa);
        Tensor tCrDO = thr_mma.partition_fragment_B(sDOb);
        auto s2r_qg = make_tiled_copy_A(LdsmT{}, mma);
        auto thr_qg = s2r_qg.get_thread_slice(tid);
        copy(s2r_qg, thr_qg.partition_S(sQGa), thr_qg.retile_D(tCrQG));
        auto s2r_do = make_tiled_copy_B(LdsmT2{}, mma);
        auto thr_do = s2r_do.get_thread_slice(tid);
        copy(s2r_do, thr_do.partition_S(sDOb), thr_do.retile_D(tCrDO));
        gemm(thr_mma, tCrQG, tCrDO, tmp);
        for (int i = 0; i < int(size(state)); ++i) {
            int j = ((i >> 1) & 1) + 2 * ((i >> 2) % kMM);
            state(i) = state(i) * gval[j] + p.scale * tmp(i);
        }
        __syncthreads();  // s_qg / s_do consumed
        stage_qgdo(i_t - 1);

        // dh -= w^T @ dv2, via the negated s_dv2 accumulated into the state
        Tensor tCrW = thr_mma.partition_fragment_A(sWa);
        Tensor tCrDV2 = thr_mma.partition_fragment_B(sDV2b);
        auto s2r_w = make_tiled_copy_A(LdsmT{}, mma);
        auto thr_w = s2r_w.get_thread_slice(tid);
        copy(s2r_w, thr_w.partition_S(sWa), thr_w.retile_D(tCrW));
        auto s2r_dv2 = make_tiled_copy_B(LdsmT2{}, mma);
        auto thr_dv2 = s2r_dv2.get_thread_slice(tid);
        copy(s2r_dv2, thr_dv2.partition_S(sDV2b), thr_dv2.retile_D(tCrDV2));
        gemm(thr_mma, tCrW, tCrDV2, state);
        __syncthreads();  // s_w / s_dv2 consumed
        stage_w(i_t - 1);
    }

    if (p.dh0 != nullptr) {
        float* dh0 = p.dh0 + i_nh * (int64_t)p.K * p.V;
        for (int i = 0; i < int(size(state)); ++i) {
            int k = get<0>(tCcS(i)), n = get<1>(tCcS(i)), vg = v0 + n;
            if (k < p.K && vg < p.V) {
                dh0[p.state_v_first ? (int64_t)vg * p.K + k : (int64_t)k * p.V + vg] = state(i);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// legacy kernels (K > 128): fp32 state in shared memory, scalar fragment loads.

template <int Kp, int BV>
struct FwdSmem {
    // persistent across the chunk loop
    alignas(128) float s_h[Kp * BV];       // [Kp][BV] fp32 state
    alignas(128) BF16 s_vnT[BV * kBT];     // [BV][BT] v_new, transposed (B operand of the update GEMM)
    alignas(128) float s_g[Kp];            // exp2(gk_last) per K channel
    union {
        struct {  // v_new phase
            alignas(128) BF16 s_w[kBT * Kp];   // [BT][Kp] natural
            alignas(128) BF16 s_hb[BV * Kp];   // [BV][Kp] state in bf16, transposed
        } a;
        struct {  // state-update phase
            alignas(128) BF16 s_kgT[Kp * kBT]; // [Kp][BT] kg, transposed
        } b;
    };
};

template <int Kp, int BV>
struct BwdSmem {
    alignas(128) float s_dh[Kp * BV];       // [Kp][BV] fp32 state gradient
    alignas(128) BF16 s_dv2T[BV * kBT];     // [BV][BT] dv2, transposed
    alignas(128) float s_g[Kp];
    union {
        struct {  // dv2 phase
            alignas(128) BF16 s_kg[kBT * Kp];    // [BT][Kp] natural
            alignas(128) BF16 s_dhb[BV * Kp];    // [BV][Kp] dh in bf16, transposed
        } a;
        struct {  // dh-update phase (s_qwT is staged with qg first, then w)
            alignas(128) BF16 s_qwT[Kp * kBT];   // [Kp][BT] qg / w, transposed
            alignas(128) BF16 s_doT[BV * kBT];   // [BV][BT] do, transposed
        } b;
    };
};

// ---------------------------------------------------------------------------
// staging helpers (all threads of the block cooperate)

// Load a [R][C] tile from gmem (row stride `row_stride`) into smem, zero-filling
// rows >= r_valid and columns >= c_valid.
template <int R, int C>
__device__ void stage_tile(BF16* dst, BF16 const* src, int64_t row_stride,
                           int r_valid, int c_valid, int tid) {
    for (int idx = tid; idx < R * C; idx += kThreads) {
        int r = idx / C, c = idx % C;
        dst[idx] = (r < r_valid && c < c_valid)
                 ? src[(int64_t)r * row_stride + c] : BF16(0.f);
    }
}

// Load a gmem [R][C] tile into smem transposed as [C][R], zero-filling like above.
// Reads are 16B-vectorized along C when alignment and validity allow.
template <int R, int C>
__device__ void stage_tile_T(BF16* dst, BF16 const* src, int64_t row_stride,
                             int r_valid, int c_valid, int tid) {
    constexpr int kVec = 8;
    constexpr int kCG = C / kVec;
    bool vec_ok = (row_stride % kVec) == 0;
    for (int idx = tid; idx < R * kCG; idx += kThreads) {
        int r = idx / kCG, c0 = (idx % kCG) * kVec;
        BF16 vals[kVec];
        if (r < r_valid && vec_ok && c0 + kVec <= c_valid) {
            *reinterpret_cast<uint4*>(vals) =
                *reinterpret_cast<uint4 const*>(src + (int64_t)r * row_stride + c0);
        } else {
            for (int j = 0; j < kVec; ++j) {
                int c = c0 + j;
                vals[j] = (r < r_valid && c < c_valid)
                        ? src[(int64_t)r * row_stride + c] : BF16(0.f);
            }
        }
        for (int j = 0; j < kVec; ++j) dst[(c0 + j) * R + r] = vals[j];
    }
}

// bf16 transposed copy of the fp32 state: dst[n][k] = BF16(s_state[k][n]).
template <int Kp, int BV>
__device__ void build_state_T(BF16* dst, float const* s_state, int tid) {
    for (int idx = tid; idx < Kp * BV; idx += kThreads) {
        int k = idx / BV, n = idx % BV;
        dst[n * Kp + k] = BF16(s_state[idx]);
    }
}

// s_g[k] = exp2(gk_last[k]) for k < K, 1 otherwise (matches the Triton masked load
// with other=0 followed by exp2).
template <int Kp>
__device__ void load_decay_g(float* s_g, float const* gk_last, int K, int tid) {
    for (int k = tid; k < Kp; k += kThreads) {
        s_g[k] = (k < K) ? exp2f(gk_last[k]) : 1.0f;
    }
}

template <int Kp, int BV>
__device__ void decay_state(float* s_state, float const* s_g, int tid) {
    for (int idx = tid; idx < Kp * BV; idx += kThreads) {
        s_state[idx] *= s_g[idx / BV];
    }
}

// Store the [Kp][BV] state tile to gmem, masked to k < K and v < V.
// state_v_first: gmem [V, K] layout (v * K + k), else [K, V] (k * V + v).
template <int Kp, int BV, typename OutT>
__device__ void store_state_gmem(OutT* dst, float const* s_state,
                                 int K, int V, int v0, bool state_v_first, int tid) {
    for (int idx = tid; idx < Kp * BV; idx += kThreads) {
        int k = idx / BV, n = idx % BV;
        int vg = v0 + n;
        if (k < K && vg < V) {
            int64_t off = state_v_first ? (int64_t)vg * K + k : (int64_t)k * V + vg;
            dst[off] = OutT(s_state[idx]);
        }
    }
}

// Initialize the fp32 state tile from gmem (h0/dht), zero-filling out-of-range.
template <int Kp, int BV>
__device__ void load_state_gmem(float* s_state, float const* src,
                                int K, int V, int v0, bool state_v_first, int tid) {
    for (int idx = tid; idx < Kp * BV; idx += kThreads) {
        int k = idx / BV, n = idx % BV;
        int vg = v0 + n;
        float val = 0.f;
        if (src != nullptr && k < K && vg < V) {
            int64_t off = state_v_first ? (int64_t)vg * K + k : (int64_t)k * V + vg;
            val = src[off];
        }
        s_state[idx] = val;
    }
}

// ---------------------------------------------------------------------------
// CuTe GEMM: C[M, N] += A[M, Kc] * B[N, Kc] with both operands smem-resident,
// row-major with the contraction dim contiguous. Returns the per-thread fp32
// accumulator fragment. Scalar smem loads (correctness first; ldmatrix is a
// performance follow-up).
template <class TiledMMA, int M, int N, int Kc>
__device__ auto mma_tn(TiledMMA const& tiled_mma, BF16 const* sA, BF16 const* sB, int tid) {
    auto thr_mma = tiled_mma.get_thread_slice(tid);
    Tensor sAt = make_tensor(make_smem_ptr(sA),
                             make_layout(make_shape(Int<M>{}, Int<Kc>{}), make_stride(Int<Kc>{}, _1{})));
    Tensor sBt = make_tensor(make_smem_ptr(sB),
                             make_layout(make_shape(Int<N>{}, Int<Kc>{}), make_stride(Int<Kc>{}, _1{})));
    Tensor acc = cute::partition_fragment_C(tiled_mma, Shape<Int<M>, Int<N>>{});
    clear(acc);
    Tensor tCrA = thr_mma.partition_fragment_A(sAt);
    Tensor tCrB = thr_mma.partition_fragment_B(sBt);
    Tensor tCsA = thr_mma.partition_A(sAt);
    Tensor tCsB = thr_mma.partition_B(sBt);
    for (int i = 0; i < int(size(tCrA)); ++i) tCrA(i) = tCsA(i);
    for (int i = 0; i < int(size(tCrB)); ++i) tCrB(i) = tCsB(i);
    gemm(thr_mma, tCrA, tCrB, acc);
    return acc;
}

// Per-thread (m, n) coordinates of an accumulator fragment.
template <class TiledMMA, int M, int N>
__device__ auto mma_coords(TiledMMA const& tiled_mma, int tid) {
    auto thr_mma = tiled_mma.get_thread_slice(tid);
    Tensor cC = make_identity_tensor(make_shape(Int<M>{}, Int<N>{}));
    return thr_mma.partition_C(cC);
}

// ---------------------------------------------------------------------------
// forward: per chunk, h[i_t] = S; v_new = u - w @ S; S = diag(exp2(gk_last)) S + kg^T @ v_new

template <int Kp, int BV>
__global__ void __launch_bounds__(kThreads) kda_fwd_h_kernel(FwdParams const p) {
    extern __shared__ __align__(128) unsigned char smem_raw[];
    auto& sm = *reinterpret_cast<FwdSmem<Kp, BV>*>(smem_raw);
    const int tid = threadIdx.x;

    const int i_v = blockIdx.x % p.NV;
    const int64_t i_nh = blockIdx.x / p.NV;
    const int64_t i_n = i_nh / p.HV;
    const int i_hv = int(i_nh % p.HV);
    const int i_hq = i_hv / (p.HV / p.H);

    int64_t bos;
    int T, NT;
    int64_t boh;
    if (p.varlen) {
        bos = p.cu_seqlens[i_n];
        T = int(p.cu_seqlens[i_n + 1] - bos);
        NT = (T + kBT - 1) / kBT;
        boh = p.chunk_offsets[i_n];
    } else {
        bos = i_n * p.T;
        T = p.T;
        NT = (T + kBT - 1) / kBT;
        boh = i_n * NT;
    }

    BF16 const* kg = p.kg + (bos * p.H + i_hq) * (int64_t)p.K;
    BF16 const* w = p.w + (bos * p.HV + i_hv) * (int64_t)p.K;
    BF16 const* u = p.u + (bos * p.HV + i_hv) * (int64_t)p.V;
    float const* gk = p.gk + (bos * p.HV + i_hv) * (int64_t)p.K;
    BF16* h = p.h + (boh * p.HV + i_hv) * (int64_t)p.K * p.V;
    BF16* v_new = p.v_new + (bos * p.HV + i_hv) * (int64_t)p.V;
    const int v0 = i_v * BV;

    load_state_gmem<Kp, BV>(sm.s_h,
                            p.h0 == nullptr ? nullptr : p.h0 + i_nh * (int64_t)p.K * p.V,
                            p.K, p.V, v0, p.state_v_first, tid);

    auto mma_vn = make_tiled_mma(MmaAtom{}, Layout<Shape<_1, _4, _1>>{});
    auto mma_up = make_tiled_mma(MmaAtom{}, Layout<Shape<_4, _1, _1>>{});

    __syncthreads();
    for (int i_t = 0; i_t < NT; ++i_t) {
        const int t_valid = min(kBT, T - i_t * kBT);
        const int64_t t0 = (int64_t)i_t * kBT;

        // h[i_t] = state at chunk entry (bf16)
        store_state_gmem<Kp, BV>(h + (int64_t)i_t * p.HV * p.K * p.V,
                                 sm.s_h, p.K, p.V, v0, p.state_v_first, tid);
        // stage w, the bf16 state copy, and the decay factors
        stage_tile<kBT, Kp>(sm.a.s_w, w + t0 * (int64_t)p.HV * p.K, (int64_t)p.HV * p.K,
                            t_valid, p.K, tid);
        build_state_T<Kp, BV>(sm.a.s_hb, sm.s_h, tid);
        const int last = min((i_t + 1) * kBT, T) - 1;
        load_decay_g<Kp>(sm.s_g, gk + (int64_t)last * p.HV * p.K, p.K, tid);
        __syncthreads();

        // v_new = u - w @ h
        {
            auto acc = mma_tn<decltype(mma_vn), kBT, BV, Kp>(mma_vn, sm.a.s_w, sm.a.s_hb, tid);
            auto tCcC = mma_coords<decltype(mma_vn), kBT, BV>(mma_vn, tid);
            BF16 const* u_t = u + t0 * (int64_t)p.HV * p.V;
            BF16* vn_t = v_new + t0 * (int64_t)p.HV * p.V;
            for (int i = 0; i < int(size(acc)); ++i) {
                int m = get<0>(tCcC(i));
                int n = get<1>(tCcC(i));
                int vg = v0 + n;
                bool valid = (m < t_valid) && (vg < p.V);
                float uu = valid ? to_f32(u_t[(int64_t)m * p.HV * p.V + vg]) : 0.f;
                float val = uu - acc(i);
                if (valid) vn_t[(int64_t)m * p.HV * p.V + vg] = BF16(val);
                sm.s_vnT[n * kBT + m] = BF16(val);
            }
        }
        __syncthreads();  // s_kgT overlaps s_w / s_hb
        stage_tile_T<kBT, Kp>(sm.b.s_kgT, kg + t0 * (int64_t)p.H * p.K, (int64_t)p.H * p.K,
                              t_valid, p.K, tid);
        decay_state<Kp, BV>(sm.s_h, sm.s_g, tid);
        __syncthreads();

        // h += kg^T @ v_new
        {
            auto acc = mma_tn<decltype(mma_up), Kp, BV, kBT>(mma_up, sm.b.s_kgT, sm.s_vnT, tid);
            auto tCcC = mma_coords<decltype(mma_up), Kp, BV>(mma_up, tid);
            for (int i = 0; i < int(size(acc)); ++i) {
                int k = get<0>(tCcC(i));
                int n = get<1>(tCcC(i));
                sm.s_h[k * BV + n] += acc(i);
            }
        }
        __syncthreads();
    }

    if (p.ht != nullptr) {
        store_state_gmem<Kp, BV>(p.ht + i_nh * (int64_t)p.K * p.V,
                                 sm.s_h, p.K, p.V, v0, p.state_v_first, tid);
    }
}

// ---------------------------------------------------------------------------
// backward: per chunk (in reverse), dh[i_t] = S;
// dv2 = dv + kg @ S; S = S * exp2(gk_last) + scale * qg^T @ do - w^T @ dv2

template <int Kp, int BV>
__global__ void __launch_bounds__(kThreads) kda_bwd_dhu_kernel(BwdParams const p) {
    extern __shared__ __align__(128) unsigned char smem_raw[];
    auto& sm = *reinterpret_cast<BwdSmem<Kp, BV>*>(smem_raw);
    const int tid = threadIdx.x;

    const int i_v = blockIdx.x % p.NV;
    const int64_t i_nh = blockIdx.x / p.NV;
    const int64_t i_n = i_nh / p.HV;
    const int i_hv = int(i_nh % p.HV);
    const int i_hq = i_hv / (p.HV / p.H);

    int64_t bos;
    int T, NT;
    int64_t boh;
    if (p.varlen) {
        bos = p.cu_seqlens[i_n];
        T = int(p.cu_seqlens[i_n + 1] - bos);
        NT = (T + kBT - 1) / kBT;
        boh = p.chunk_offsets[i_n];
    } else {
        bos = i_n * p.T;
        T = p.T;
        NT = (T + kBT - 1) / kBT;
        boh = i_n * NT;
    }

    BF16 const* qg = p.qg + (bos * p.H + i_hq) * (int64_t)p.K;
    BF16 const* kg = p.kg + (bos * p.H + i_hq) * (int64_t)p.K;
    BF16 const* w = p.w + (bos * p.HV + i_hv) * (int64_t)p.K;
    float const* gk = p.gk + (bos * p.HV + i_hv) * (int64_t)p.K;
    BF16 const* do_ = p.do_ + (bos * p.HV + i_hv) * (int64_t)p.V;
    BF16 const* dv = p.dv + (bos * p.HV + i_hv) * (int64_t)p.V;
    BF16* dv2 = p.dv2 + (bos * p.HV + i_hv) * (int64_t)p.V;
    BF16* dh = p.dh + (boh * p.HV + i_hv) * (int64_t)p.K * p.V;
    const int v0 = i_v * BV;

    load_state_gmem<Kp, BV>(sm.s_dh,
                            p.dht == nullptr ? nullptr : p.dht + i_nh * (int64_t)p.K * p.V,
                            p.K, p.V, v0, p.state_v_first, tid);

    auto mma_dv = make_tiled_mma(MmaAtom{}, Layout<Shape<_1, _4, _1>>{});
    auto mma_up = make_tiled_mma(MmaAtom{}, Layout<Shape<_4, _1, _1>>{});

    __syncthreads();
    for (int i_t = NT - 1; i_t >= 0; --i_t) {
        const int t_valid = min(kBT, T - i_t * kBT);
        const int64_t t0 = (int64_t)i_t * kBT;

        // dh[i_t] = state gradient at chunk entry (bf16)
        store_state_gmem<Kp, BV>(dh + (int64_t)i_t * p.HV * p.K * p.V,
                                 sm.s_dh, p.K, p.V, v0, p.state_v_first, tid);
        stage_tile<kBT, Kp>(sm.a.s_kg, kg + t0 * (int64_t)p.H * p.K, (int64_t)p.H * p.K,
                            t_valid, p.K, tid);
        build_state_T<Kp, BV>(sm.a.s_dhb, sm.s_dh, tid);
        const int last = min((i_t + 1) * kBT, T) - 1;
        load_decay_g<Kp>(sm.s_g, gk + (int64_t)last * p.HV * p.K, p.K, tid);
        __syncthreads();

        // dv2 = dv + kg @ dh
        {
            auto acc = mma_tn<decltype(mma_dv), kBT, BV, Kp>(mma_dv, sm.a.s_kg, sm.a.s_dhb, tid);
            auto tCcC = mma_coords<decltype(mma_dv), kBT, BV>(mma_dv, tid);
            BF16 const* dv_t = dv + t0 * (int64_t)p.HV * p.V;
            BF16* dv2_t = dv2 + t0 * (int64_t)p.HV * p.V;
            for (int i = 0; i < int(size(acc)); ++i) {
                int m = get<0>(tCcC(i));
                int n = get<1>(tCcC(i));
                int vg = v0 + n;
                bool valid = (m < t_valid) && (vg < p.V);
                float dvv = valid ? to_f32(dv_t[(int64_t)m * p.HV * p.V + vg]) : 0.f;
                float val = acc(i) + dvv;
                if (valid) dv2_t[(int64_t)m * p.HV * p.V + vg] = BF16(val);
                sm.s_dv2T[n * kBT + m] = BF16(val);
            }
        }
        __syncthreads();  // phase-b buffers overlap s_kg / s_dhb
        stage_tile_T<kBT, Kp>(sm.b.s_qwT, qg + t0 * (int64_t)p.H * p.K, (int64_t)p.H * p.K,
                              t_valid, p.K, tid);
        stage_tile_T<kBT, BV>(sm.b.s_doT, do_ + t0 * (int64_t)p.HV * p.V + v0,
                              (int64_t)p.HV * p.V, t_valid, min(BV, p.V - v0), tid);
        decay_state<Kp, BV>(sm.s_dh, sm.s_g, tid);
        __syncthreads();

        // dh += scale * qg^T @ do
        {
            auto acc = mma_tn<decltype(mma_up), Kp, BV, kBT>(mma_up, sm.b.s_qwT, sm.b.s_doT, tid);
            auto tCcC = mma_coords<decltype(mma_up), Kp, BV>(mma_up, tid);
            for (int i = 0; i < int(size(acc)); ++i) {
                int k = get<0>(tCcC(i));
                int n = get<1>(tCcC(i));
                sm.s_dh[k * BV + n] += p.scale * acc(i);
            }
        }
        __syncthreads();  // restage s_qwT with w
        stage_tile_T<kBT, Kp>(sm.b.s_qwT, w + t0 * (int64_t)p.HV * p.K, (int64_t)p.HV * p.K,
                              t_valid, p.K, tid);
        __syncthreads();
        // dh -= w^T @ dv2
        {
            auto acc = mma_tn<decltype(mma_up), Kp, BV, kBT>(mma_up, sm.b.s_qwT, sm.s_dv2T, tid);
            auto tCcC = mma_coords<decltype(mma_up), Kp, BV>(mma_up, tid);
            for (int i = 0; i < int(size(acc)); ++i) {
                int k = get<0>(tCcC(i));
                int n = get<1>(tCcC(i));
                sm.s_dh[k * BV + n] -= acc(i);
            }
        }
        __syncthreads();
    }

    if (p.dh0 != nullptr) {
        store_state_gmem<Kp, BV>(p.dh0 + i_nh * (int64_t)p.K * p.V,
                                 sm.s_dh, p.K, p.V, v0, p.state_v_first, tid);
    }
}

// ---------------------------------------------------------------------------
// launchers

template <int Kp, int BV>
void launch_fwd_pipe(FwdParams& p, int64_t grid_blocks, cudaStream_t stream) {
    constexpr int kSmemBytes = int(sizeof(FwdPipeSmem<Kp, BV>));
    static bool configured = [] {
        cudaFuncSetAttribute(kda_fwd_h_pipe_kernel<Kp, BV>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemBytes);
        return true;
    }();
    (void)configured;
    kda_fwd_h_pipe_kernel<Kp, BV><<<dim3(unsigned(grid_blocks)), kThreads, kSmemBytes, stream>>>(p);
}

template <int Kp, int BV>
void launch_bwd_pipe(BwdParams& p, int64_t grid_blocks, cudaStream_t stream) {
    constexpr int kSmemBytes = int(sizeof(BwdPipeSmem<Kp, BV>));
    static bool configured = [] {
        cudaFuncSetAttribute(kda_bwd_dhu_pipe_kernel<Kp, BV>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemBytes);
        return true;
    }();
    (void)configured;
    kda_bwd_dhu_pipe_kernel<Kp, BV><<<dim3(unsigned(grid_blocks)), kThreads, kSmemBytes, stream>>>(p);
}

template <int Kp, int BV>
void launch_fwd(FwdParams& p, int64_t grid_blocks, cudaStream_t stream) {
    constexpr int kSmemBytes = int(sizeof(FwdSmem<Kp, BV>));
    static bool configured = [] {
        cudaFuncSetAttribute(kda_fwd_h_kernel<Kp, BV>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemBytes);
        return true;
    }();
    (void)configured;
    kda_fwd_h_kernel<Kp, BV><<<dim3(unsigned(grid_blocks)), kThreads, kSmemBytes, stream>>>(p);
}

template <int Kp, int BV>
void launch_bwd(BwdParams& p, int64_t grid_blocks, cudaStream_t stream) {
    constexpr int kSmemBytes = int(sizeof(BwdSmem<Kp, BV>));
    static bool configured = [] {
        cudaFuncSetAttribute(kda_bwd_dhu_kernel<Kp, BV>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemBytes);
        return true;
    }();
    (void)configured;
    kda_bwd_dhu_kernel<Kp, BV><<<dim3(unsigned(grid_blocks)), kThreads, kSmemBytes, stream>>>(p);
}

// Padded K (multiple of 64) and the V tile used for it. BV=32 above K=128 keeps
// dynamic smem under the 99KB opt-in limit of sm_120.
int padded_k(int64_t K) {
    TORCH_CHECK(K > 0 && K <= 256, "K must be in (0, 256]");
    return int((K + 63) / 64 * 64);
}

int v_tile(int Kp) {
    return Kp > 64 ? 32 : 64;
}

}  // anonymous namespace (internal linkage: the test harness JIT-loads this file as a
// second module alongside the pip extension; named-namespace kernels interpose
// across modules and the smem opt-in attribute then lands on the wrong kernel)

// ---------------------------------------------------------------------------
// host entry points (fla chunk_gated_delta_rule_fwd_h / _bwd_dhu equivalents,
// KDA call shape only: USE_G=False, USE_GK=True, SAVE_NEW_VALUE=True)

std::tuple<torch::Tensor, torch::Tensor, std::optional<torch::Tensor>>
chunk_gated_delta_rule_fwd_h(
    torch::Tensor kg,
    torch::Tensor w,
    torch::Tensor u,
    torch::Tensor gk,
    std::optional<torch::Tensor> initial_state,
    bool output_final_state,
    int64_t chunk_size,
    bool state_v_first,
    std::optional<torch::Tensor> cu_seqlens,
    std::optional<torch::Tensor> chunk_offsets,
    int64_t nt_total
) {
    TORCH_CHECK(chunk_size == kBT, "only chunk_size=64 is supported");
    TORCH_CHECK(kg.is_cuda() && kg.is_contiguous() && kg.dim() == 4, "kg must be 4D contiguous CUDA");
    TORCH_CHECK(kg.scalar_type() == at::kBFloat16, "only bf16 operands are supported");
    int64_t B = kg.size(0), T = kg.size(1), H = kg.size(2), K = kg.size(3);
    TORCH_CHECK(w.is_cuda() && w.is_contiguous() && w.dim() == 4 && w.scalar_type() == at::kBFloat16);
    TORCH_CHECK(u.is_cuda() && u.is_contiguous() && u.dim() == 4 && u.scalar_type() == at::kBFloat16);
    int64_t HV = w.size(2), V = u.size(3);
    TORCH_CHECK(w.size(0) == B && w.size(1) == T && w.size(3) == K);
    TORCH_CHECK(u.size(0) == B && u.size(1) == T && u.size(2) == HV);
    TORCH_CHECK(gk.is_cuda() && gk.is_contiguous() && gk.scalar_type() == at::kFloat);
    TORCH_CHECK(gk.dim() == 4 && gk.size(0) == B && gk.size(1) == T && gk.size(2) == HV && gk.size(3) == K);
    TORCH_CHECK(HV % H == 0, "HV must be a multiple of H");

    bool varlen = cu_seqlens.has_value();
    int64_t N, NTt;
    if (varlen) {
        TORCH_CHECK(B == 1, "B must be 1 when cu_seqlens is provided");
        TORCH_CHECK(chunk_offsets.has_value(), "chunk_offsets must be provided when cu_seqlens is provided");
        TORCH_CHECK(cu_seqlens->is_cuda() && cu_seqlens->is_contiguous() && cu_seqlens->dtype() == torch::kLong);
        TORCH_CHECK(chunk_offsets->is_cuda() && chunk_offsets->is_contiguous() && chunk_offsets->dtype() == torch::kLong);
        N = cu_seqlens->size(0) - 1;
        NTt = nt_total;
        TORCH_CHECK(NTt > 0, "nt_total must be provided for varlen");
    } else {
        N = B;
        NTt = (T + kBT - 1) / kBT;
    }

    if (initial_state.has_value()) {
        auto const& h0 = initial_state.value();
        TORCH_CHECK(h0.is_cuda() && h0.is_contiguous() && h0.scalar_type() == at::kFloat);
        TORCH_CHECK(h0.dim() == 4 && h0.size(0) == N && h0.size(1) == HV);
    }

    torch::Tensor h = state_v_first
        ? torch::empty({B, NTt, HV, V, K}, kg.options())
        : torch::empty({B, NTt, HV, K, V}, kg.options());
    torch::Tensor v_new = torch::empty_like(u);
    std::optional<torch::Tensor> final_state;
    if (output_final_state) {
        final_state = state_v_first
            ? torch::zeros({N, HV, V, K}, kg.options().dtype(at::kFloat))
            : torch::zeros({N, HV, K, V}, kg.options().dtype(at::kFloat));
    }

    FwdParams p;
    p.kg = reinterpret_cast<BF16 const*>(kg.data_ptr());
    p.w = reinterpret_cast<BF16 const*>(w.data_ptr());
    p.u = reinterpret_cast<BF16 const*>(u.data_ptr());
    p.gk = gk.data_ptr<float>();
    p.h = reinterpret_cast<BF16*>(h.data_ptr());
    p.v_new = reinterpret_cast<BF16*>(v_new.data_ptr());
    p.h0 = initial_state.has_value() ? initial_state->data_ptr<float>() : nullptr;
    p.ht = final_state.has_value() ? final_state->data_ptr<float>() : nullptr;
    p.cu_seqlens = varlen ? cu_seqlens->data_ptr<int64_t>() : nullptr;
    p.chunk_offsets = varlen ? chunk_offsets->data_ptr<int64_t>() : nullptr;
    p.T = int(T); p.H = int(H); p.HV = int(HV); p.K = int(K); p.V = int(V);
    p.varlen = varlen;
    p.state_v_first = state_v_first;

    const int Kp = padded_k(K);
    p.NV = int((V + v_tile(Kp) - 1) / v_tile(Kp));
    int64_t grid_blocks = (int64_t)p.NV * N * HV;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    switch (Kp) {
        case 64: launch_fwd_pipe<64, 64>(p, grid_blocks, stream); break;
        case 128: launch_fwd_pipe<128, 32>(p, grid_blocks, stream); break;
        case 192: launch_fwd<192, 32>(p, grid_blocks, stream); break;
        default: launch_fwd<256, 32>(p, grid_blocks, stream); break;
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {h, v_new, final_state};
}

std::tuple<torch::Tensor, std::optional<torch::Tensor>, torch::Tensor>
chunk_gated_delta_rule_bwd_dhu(
    torch::Tensor qg,
    torch::Tensor kg,
    torch::Tensor w,
    torch::Tensor gk,
    torch::Tensor do_,
    torch::Tensor dv,
    std::optional<torch::Tensor> h0,
    std::optional<torch::Tensor> dht,
    double scale,
    int64_t chunk_size,
    bool state_v_first,
    std::optional<torch::Tensor> cu_seqlens,
    std::optional<torch::Tensor> chunk_offsets,
    int64_t nt_total
) {
    TORCH_CHECK(chunk_size == kBT, "only chunk_size=64 is supported");
    TORCH_CHECK(qg.is_cuda() && qg.is_contiguous() && qg.dim() == 4 && qg.scalar_type() == at::kBFloat16);
    int64_t B = qg.size(0), T = qg.size(1), H = qg.size(2), K = qg.size(3);
    TORCH_CHECK(kg.is_cuda() && kg.is_contiguous() && kg.sizes() == qg.sizes() && kg.scalar_type() == at::kBFloat16);
    TORCH_CHECK(w.is_cuda() && w.is_contiguous() && w.dim() == 4 && w.scalar_type() == at::kBFloat16);
    int64_t HV = w.size(2);
    TORCH_CHECK(do_.is_cuda() && do_.is_contiguous() && do_.dim() == 4 && do_.scalar_type() == at::kBFloat16);
    int64_t V = do_.size(3);
    TORCH_CHECK(dv.is_cuda() && dv.is_contiguous() && dv.sizes() == do_.sizes() && dv.scalar_type() == at::kBFloat16);
    TORCH_CHECK(gk.is_cuda() && gk.is_contiguous() && gk.scalar_type() == at::kFloat);
    TORCH_CHECK(gk.dim() == 4 && gk.size(0) == B && gk.size(1) == T && gk.size(2) == HV && gk.size(3) == K);
    TORCH_CHECK(HV % H == 0, "HV must be a multiple of H");

    bool varlen = cu_seqlens.has_value();
    int64_t N, NTt;
    if (varlen) {
        TORCH_CHECK(B == 1, "B must be 1 when cu_seqlens is provided");
        TORCH_CHECK(chunk_offsets.has_value(), "chunk_offsets must be provided when cu_seqlens is provided");
        TORCH_CHECK(cu_seqlens->is_cuda() && cu_seqlens->is_contiguous() && cu_seqlens->dtype() == torch::kLong);
        TORCH_CHECK(chunk_offsets->is_cuda() && chunk_offsets->is_contiguous() && chunk_offsets->dtype() == torch::kLong);
        N = cu_seqlens->size(0) - 1;
        NTt = nt_total;
        TORCH_CHECK(NTt > 0, "nt_total must be provided for varlen");
    } else {
        N = B;
        NTt = (T + kBT - 1) / kBT;
    }

    torch::Tensor dh = state_v_first
        ? torch::empty({B, NTt, HV, V, K}, qg.options())
        : torch::empty({B, NTt, HV, K, V}, qg.options());
    std::optional<torch::Tensor> dh0;
    if (h0.has_value()) {
        TORCH_CHECK(h0->is_cuda() && h0->is_contiguous() && h0->scalar_type() == at::kFloat);
        dh0 = torch::empty_like(h0.value(), h0->options().dtype(at::kFloat));
    }
    if (dht.has_value()) {
        TORCH_CHECK(dht->is_cuda() && dht->is_contiguous() && dht->scalar_type() == at::kFloat);
    }
    torch::Tensor dv2 = torch::empty_like(dv);

    BwdParams p;
    p.qg = reinterpret_cast<BF16 const*>(qg.data_ptr());
    p.kg = reinterpret_cast<BF16 const*>(kg.data_ptr());
    p.w = reinterpret_cast<BF16 const*>(w.data_ptr());
    p.gk = gk.data_ptr<float>();
    p.do_ = reinterpret_cast<BF16 const*>(do_.data_ptr());
    p.dv = reinterpret_cast<BF16 const*>(dv.data_ptr());
    p.dv2 = reinterpret_cast<BF16*>(dv2.data_ptr());
    p.dh = reinterpret_cast<BF16*>(dh.data_ptr());
    p.dh0 = dh0.has_value() ? dh0->data_ptr<float>() : nullptr;
    p.dht = dht.has_value() ? dht->data_ptr<float>() : nullptr;
    p.cu_seqlens = varlen ? cu_seqlens->data_ptr<int64_t>() : nullptr;
    p.chunk_offsets = varlen ? chunk_offsets->data_ptr<int64_t>() : nullptr;
    p.scale = float(scale);
    p.T = int(T); p.H = int(H); p.HV = int(HV); p.K = int(K); p.V = int(V);
    p.varlen = varlen;
    p.state_v_first = state_v_first;

    const int Kp = padded_k(K);
    p.NV = int((V + v_tile(Kp) - 1) / v_tile(Kp));
    int64_t grid_blocks = (int64_t)p.NV * N * HV;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    switch (Kp) {
        case 64: launch_bwd_pipe<64, 64>(p, grid_blocks, stream); break;
        case 128: launch_bwd_pipe<128, 32>(p, grid_blocks, stream); break;
        case 192: launch_bwd<192, 32>(p, grid_blocks, stream); break;
        default: launch_bwd<256, 32>(p, grid_blocks, stream); break;
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {dh, dh0, dv2};
}
