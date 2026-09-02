// Fused KDA backward kernel producing dq/dk/dv2/dg/db/dAkk for one chunk.
// Replicates fla/ops/kda/chunk_bwd.py::chunk_kda_bwd_kernel_wy_dqkg_fused
// (host wrapper at chunk_bwd.py:366-431).
//
// One CTA handles one (chunk, batch*head) pair: BT=64 token rows, tiled over
// K in BK=64 blocks and V in BV=64 blocks. All GEMMs run on tensor cores via
// CuTe SM80 16x8x16 mma atoms with fp32 accumulators, matching Triton's
// input-dtype mma + fp32 accumulate semantics.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>

#include <cute/tensor.hpp>

#include "common.cuh"

namespace {

constexpr int kBT = 64;   // chunk size
constexpr int kBK = 64;   // K tile
constexpr int kBV = 64;   // V tile
constexpr int kThreads = 256;  // 8 warps, one 4x2 tiled mma; keeps the four
                               // fp32 accumulators register-resident (16 vals
                               // per thread each, no local-memory spills)
constexpr int kPad = 8;   // smem row padding in elements (16B) against bank conflicts
constexpr int kCP = 64 + kPad;  // padded row stride of every 64-wide tile

// ---------------------------------------------------------------------------
// cp.async helpers

__device__ __forceinline__ void cp_async16(void* dst, void const* src, bool full) {
    uint32_t s = cute::cast_smem_ptr_to_uint(dst);
    int sz = full ? 16 : 0;  // src-size 0 zero-fills, used for masked rows
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

template <typename T>
struct FusedSmem {
    static constexpr int kTile = kBT * kCP;
    // persistent
    T A_t[kTile];   // Akk transposed: A_t[c][r] = Akk[t0+r][c]
    // g/k staging for the current i_k, persistent across the V loop so the
    // loads overlap the V-loop staging/compute
    union {
        T raw[3 * kTile];
        struct {
            float g[kBT * 68];  // fp32 [t][k], padded to 68 floats (16B) per row
            T k[kTile];
        } gk;
    } sp;
    float beta[kBT];
    float db[kBT];
    float dgk[64];
    float gn[64];
    union {
        // V-loop tiles. h/dh are stored as [k][v] (B-operand K-major).
        struct {
            T do_[kTile]; T v_new[kTile]; T dv[kTile]; T h[kTile]; T dh[kTile];
            T v[kTile];  // only used when i_k == 0
        } vl;
        // epilogue tiles: kg/dw as [t][k], dwT as [k][t]; q is staged into dw
        // after the epilogue GEMMs. Physically overlaps the vl tiles, which
        // are all dead by this phase.
        struct { T kg[kTile]; T dw[kTile]; T dwT[kTile]; } ep;
        // dA postprocess: masked dA and the first product transposed
        struct { T dAm[kTile]; T c1T[kTile]; } pp;
    };
};

// s[r][c] = g[r*row_stride + c] via 16B cp.async, rows past rows_valid
// zero-filled. Requires 16B-aligned rows (row_stride and base offset multiples
// of 8 elements).
template <typename T>
__device__ __forceinline__ void stage_tile(T* s, T const* g, T const* g_safe, int64_t row_stride,
                                           int rows_valid, int tid) {
    constexpr int kCG = 64 / 8;
    for (int idx = tid; idx < kBT * kCG; idx += kThreads) {
        int r = idx / kCG, c = (idx % kCG) * 8;
        bool full = r < rows_valid;
        cp_async16(s + r * kCP + c, full ? g + (int64_t)r * row_stride + c : g_safe, full);
    }
}

// h/dh state tile stored as s[kk][vv]; k/v dims are always fully valid.
// state_v_first=false: gmem is [K, V], s[kk][vv] = g[(k0+kk)*ld + v0+vv], cp.async
// state_v_first=true:  gmem is [V, K], s[kk][vv] = g[(v0+vv)*ld + k0+kk], scalar
template <typename T>
__device__ __forceinline__ void stage_state_tile(T* s, T const* g, int64_t ld, int64_t k0, int64_t v0,
                                                 bool state_v_first, int tid) {
    if (!state_v_first) {
        constexpr int kCG = 64 / 8;
        for (int idx = tid; idx < 64 * kCG; idx += kThreads) {
            int kk = idx / kCG, vv = (idx % kCG) * 8;
            cp_async16(s + kk * kCP + vv, g + (k0 + kk) * ld + v0 + vv, true);
        }
    } else {
        for (int idx = tid; idx < 64 * 64; idx += kThreads) {
            int kk = idx >> 6, vv = idx & 63;
            s[kk * kCP + vv] = g[(v0 + vv) * ld + k0 + kk];
        }
    }
}

template <typename T, int K, int V>
__global__ void __launch_bounds__(kThreads, 1) chunk_kda_bwd_wy_dqkg_fused_kernel(
    T const* __restrict__ q_g, T const* __restrict__ k_g,
    T const* __restrict__ v_g, T const* __restrict__ v_new_g,
    float const* __restrict__ g_g, float const* __restrict__ beta_g,
    T const* __restrict__ A_g, T const* __restrict__ h_g,
    T const* __restrict__ do_g, T const* __restrict__ dh_g, T const* __restrict__ dv_g,
    float* __restrict__ dq_g, float* __restrict__ dk_g, T* __restrict__ dv2_g,
    float* __restrict__ dg_g, float* __restrict__ db_g, float* __restrict__ dA_g,
    int64_t const* __restrict__ cu_seqlens, int64_t const* __restrict__ chunk_indices,
    float scale, int T_len, int H, int HV,
    bool state_v_first, bool is_varlen
) {
    using namespace cute;

    constexpr int NK = K / kBK;
    constexpr int NV = V / kBV;
    const int tid = threadIdx.x;

    int64_t i_t = blockIdx.x;
    const int i_bh = blockIdx.y;
    const int i_b = i_bh / HV;
    const int i_hv = i_bh % HV;
    const int i_h = i_hv / (HV / H);

    int64_t i_tg, bos;
    int Tl;
    if (is_varlen) {
        i_tg = i_t;
        int64_t i_n = chunk_indices[i_t * 2];
        i_t = chunk_indices[i_t * 2 + 1];
        bos = cu_seqlens[i_n];
        Tl = int(cu_seqlens[i_n + 1] - bos);
    } else {
        int NT = (T_len + kBT - 1) / kBT;
        i_tg = (int64_t)i_b * NT + i_t;
        bos = (int64_t)i_b * T_len;
        Tl = T_len;
    }
    const int t0 = int(i_t) * kBT;
    const int t_end = Tl < t0 + kBT ? Tl : t0 + kBT;
    const int last = t_end - 1;         // local row of the last valid token
    const int rows_valid = t_end - t0;  // valid rows in this chunk

    // base offsets (int64 everywhere, matching the Triton kernel)
    const int64_t qk_base = (bos * H + i_h) * (int64_t)K;
    const int64_t hvK_base = (bos * HV + i_hv) * (int64_t)K;
    const int64_t hvV_base = (bos * HV + i_hv) * (int64_t)V;
    const int64_t h_base = (i_tg * HV + i_hv) * (int64_t)K * V;
    const int64_t A_base = (bos * HV + i_hv) * (int64_t)kBT;
    const int64_t beta_base = bos * HV + i_hv;
    const int64_t qk_row = (int64_t)H * K;
    const int64_t hvK_row = (int64_t)HV * K;
    const int64_t hvV_row = (int64_t)HV * V;
    const int64_t A_row = (int64_t)HV * kBT;

    extern __shared__ __align__(128) unsigned char smem_raw[];
    using Smem = FusedSmem<T>;
    Smem& sm = *reinterpret_cast<Smem*>(smem_raw);

    // preload beta, db init, and Akk (transposed; the natural orientation is
    // read through a strided-B view where needed)
    if (tid < kBT) {
        sm.db[tid] = 0.f;
        sm.beta[tid] = (tid < rows_valid) ? beta_g[beta_base + (int64_t)(t0 + tid) * HV] : 0.f;
    }
    {
        T const* Ap = A_g + A_base + (int64_t)t0 * A_row;
        for (int idx = tid; idx < kBT * 8; idx += kThreads) {
            int r = idx / 8, c = (idx % 8) * 8;
            T vals[8];
            if (r < rows_valid) {
                *reinterpret_cast<uint4*>(vals) =
                    *reinterpret_cast<uint4 const*>(Ap + (int64_t)r * A_row + c);
            } else {
                CUTE_UNROLL
                for (int j = 0; j < 8; ++j) vals[j] = T(0.f);
            }
            CUTE_UNROLL
            for (int j = 0; j < 8; ++j) sm.A_t[(c + j) * kCP + r] = vals[j];
        }
    }

    using MmaOp = std::conditional_t<std::is_same_v<T, cutlass::bfloat16_t>,
                                     SM80_16x8x16_F32BF16BF16F32_TN,
                                     SM80_16x8x16_F32F16F16F32_TN>;
    auto tiled_mma = make_tiled_mma(MMA_Atom<MmaOp>{}, Layout<Shape<_4, _2, _1>>{});
    auto thr_mma = tiled_mma.get_thread_slice(tid);
    Tensor cC = make_identity_tensor(Shape<Int<kBT>, _64>{});
    Tensor tCcC = thr_mma.partition_C(cC);

    // acc += sA[64,K-dim] @ sB[64,K-dim]^T with all tiles K-major in smem.
    // Operand fragments load via ldmatrix; copies/mma run k-block by k-block to
    // bound register pressure.
    Copy_Atom<SM75_U32x4_LDSM_N, T> ldsm_n;
    // x2 variant for B operands: the per-warp B tile per k-block is too small
    // for the x4 atom (same constraint as chunk_h.cu).
    Copy_Atom<SM75_U32x2_LDSM_N, T> ldsm_n2;
    auto s2r_a = make_tiled_copy_A(ldsm_n, tiled_mma);
    auto s2r_b = make_tiled_copy_B(ldsm_n2, tiled_mma);
    auto thr_s2r_a = s2r_a.get_thread_slice(tid);
    auto thr_s2r_b = s2r_b.get_thread_slice(tid);
    auto gemm_sm = [&](auto& acc, T const* sA, T const* sB) {
        Tensor sAt = make_tensor(make_smem_ptr(sA), Layout<Shape<_64, _64>, Stride<Int<kCP>, _1>>{});
        Tensor sBt = make_tensor(make_smem_ptr(sB), Layout<Shape<_64, _64>, Stride<Int<kCP>, _1>>{});
        Tensor tCrA = thr_mma.partition_fragment_A(sAt);
        Tensor tCrB = thr_mma.partition_fragment_B(sBt);
        Tensor tXsA = thr_s2r_a.partition_S(sAt);      // (CPY, M, K)
        Tensor tXsB = thr_s2r_b.partition_S(sBt);      // (CPY, N, K)
        Tensor tXrA = thr_s2r_a.retile_D(tCrA);
        Tensor tXrB = thr_s2r_b.retile_D(tCrB);
        constexpr int KB = decltype(size<2>(tXsA))::value;
        CUTE_UNROLL
        for (int kb = 0; kb < KB; ++kb) {
            copy(s2r_a, tXsA(_, _, kb), tXrA(_, _, kb));
            copy(s2r_b, tXsB(_, _, kb), tXrB(_, _, kb));
            gemm(tiled_mma, tCrA(_, _, kb), tCrB(_, _, kb), acc);
        }
    };
    // Same, but the B operand is a contraction-strided view of a natural
    // [K][N] tile: B[N][K] with stride (1, kCP), loaded with ldsm .trans.
    Copy_Atom<SM75_U16x4_LDSM_T, T> ldsm_t2;
    auto s2r_bt = make_tiled_copy_B(ldsm_t2, tiled_mma);
    auto thr_s2r_bt = s2r_bt.get_thread_slice(tid);
    auto gemm_sm_bt = [&](auto& acc, T const* sA, T const* sB) {
        Tensor sAt = make_tensor(make_smem_ptr(sA), Layout<Shape<_64, _64>, Stride<Int<kCP>, _1>>{});
        Tensor sBt = make_tensor(make_smem_ptr(sB), Layout<Shape<_64, _64>, Stride<_1, Int<kCP>>>{});
        Tensor tCrA = thr_mma.partition_fragment_A(sAt);
        Tensor tCrB = thr_mma.partition_fragment_B(sBt);
        Tensor tXsA = thr_s2r_a.partition_S(sAt);
        Tensor tXsB = thr_s2r_bt.partition_S(sBt);
        Tensor tXrA = thr_s2r_a.retile_D(tCrA);
        Tensor tXrB = thr_s2r_bt.retile_D(tCrB);
        constexpr int KB = decltype(size<2>(tXsA))::value;
        CUTE_UNROLL
        for (int kb = 0; kb < KB; ++kb) {
            copy(s2r_a, tXsA(_, _, kb), tXrA(_, _, kb));
            copy(s2r_bt, tXsB(_, _, kb), tXrB(_, _, kb));
            gemm(tiled_mma, tCrA(_, _, kb), tCrB(_, _, kb), acc);
        }
    };

    Tensor acc_dA = partition_fragment_C(tiled_mma, Shape<Int<kBT>, _64>{});
    clear(acc_dA);

    for (int i_k = 0; i_k < NK; ++i_k) {
        __syncthreads();  // separates this iteration's tile writes from the previous phase
        if (tid < 64) {
            sm.dgk[tid] = 0.f;
            sm.gn[tid] = g_g[hvK_base + (int64_t)last * hvK_row + i_k * kBK + tid];
        }
        // stage g/k for the epilogue phases; the loads overlap the V loop
        {
            float const* gsrc = g_g + hvK_base + (int64_t)t0 * hvK_row + i_k * kBK;
            for (int idx = tid; idx < kBT * 16; idx += kThreads) {
                int r = idx / 16, c = (idx % 16) * 4;
                bool full = r < rows_valid;
                cp_async16(sm.sp.gk.g + r * 68 + c,
                           full ? gsrc + (int64_t)r * hvK_row + c : g_g, full);
            }
            T const* ksrc = k_g + qk_base + (int64_t)t0 * qk_row + i_k * kBK;
            stage_tile(sm.sp.gk.k, ksrc, k_g, qk_row, rows_valid, tid);
        }
        Tensor acc_dq = partition_fragment_C(tiled_mma, Shape<Int<kBT>, _64>{});
        Tensor acc_dk = partition_fragment_C(tiled_mma, Shape<Int<kBT>, _64>{});
        Tensor acc_dw = partition_fragment_C(tiled_mma, Shape<Int<kBT>, _64>{});
        clear(acc_dq);
        clear(acc_dk);
        clear(acc_dw);

        for (int i_v = 0; i_v < NV; ++i_v) {
            const int64_t v0 = i_v * kBV;
            T const* do_p = do_g + hvV_base + (int64_t)t0 * hvV_row + v0;
            stage_tile(sm.vl.do_, do_p, do_g, hvV_row, rows_valid, tid);
            T const* vn_p = v_new_g + hvV_base + (int64_t)t0 * hvV_row + v0;
            stage_tile(sm.vl.v_new, vn_p, v_new_g, hvV_row, rows_valid, tid);
            T const* dv_p = dv_g + hvV_base + (int64_t)t0 * hvV_row + v0;
            stage_tile(sm.vl.dv, dv_p, dv_g, hvV_row, rows_valid, tid);
            stage_state_tile(sm.vl.h, h_g + h_base, state_v_first ? K : V, i_k * kBK, v0, state_v_first, tid);
            stage_state_tile(sm.vl.dh, dh_g + h_base, state_v_first ? K : V, i_k * kBK, v0, state_v_first, tid);
            if (i_k == 0) {
                T const* v_p = v_g + hvV_base + (int64_t)t0 * hvV_row + v0;
                stage_tile(sm.vl.v, v_p, v_g, hvV_row, rows_valid, tid);
            }
            cp_async_commit();
            cp_async_wait<0>();
            __syncthreads();

            // dgk[k] += sum_v h[k][v] * dh[k][v] (fp32)
            {
                int kk = tid & 63, part = tid >> 6;
                constexpr int kSpan = 64 / (kThreads / 64);
                float s = 0.f;
                for (int vv = part * kSpan; vv < part * kSpan + kSpan; ++vv)
                    s += to_f32(sm.vl.h[kk * kCP + vv]) * to_f32(sm.vl.dh[kk * kCP + vv]);
                atomicAdd(&sm.dgk[kk], s);
            }

            gemm_sm(acc_dq, sm.vl.do_, sm.vl.h);    // dq += do @ h
            gemm_sm(acc_dk, sm.vl.v_new, sm.vl.dh); // dk += v_new @ dh
            gemm_sm(acc_dw, sm.vl.dv, sm.vl.h);     // dw += dv @ h

            if (i_k == 0) {
                gemm_sm(acc_dA, sm.vl.dv, sm.vl.v);  // dA += dv @ v^T
                Tensor acc_dvb = partition_fragment_C(tiled_mma, Shape<Int<kBT>, _64>{});
                clear(acc_dvb);
                gemm_sm_bt(acc_dvb, sm.A_t, sm.vl.dv);  // dvb = Akk^T @ dv
                T* dv2_p = dv2_g + hvV_base + (int64_t)t0 * hvV_row;
                // in-thread partial sums per fragment row half (e bit1): one
                // smem atomic per row instead of one per element
                float db_acc[2] = {0.f, 0.f};
                CUTE_UNROLL
                for (int e = 0; e < size(acc_dvb); e += 2) {
                    auto crd = tCcC(e);
                    int i = get<0>(crd), vv = get<1>(crd);
                    float dvb0 = acc_dvb(e), dvb1 = acc_dvb(e + 1);
                    db_acc[(e >> 1) & 1] += dvb0 * to_f32(sm.vl.v[i * kCP + vv])
                                          + dvb1 * to_f32(sm.vl.v[i * kCP + vv + 1]);
                    if (i < rows_valid) {
                        // adjacent-column pair -> one 4B store
                        T pair[2] = {T(dvb0 * sm.beta[i]), T(dvb1 * sm.beta[i])};
                        *reinterpret_cast<uint32_t*>(dv2_p + (int64_t)i * hvV_row + v0 + vv) =
                            *reinterpret_cast<uint32_t*>(pair);
                    }
                }
                CUTE_UNROLL
                for (int hh = 0; hh < 2; ++hh)
                    atomicAdd(&sm.db[get<0>(tCcC(2 * hh))], db_acc[hh]);
            }
            __syncthreads();  // tiles are reusable from the next i_v iteration
        }

        // decay + dw/kg tiles for the dA/dkgb GEMMs (g/k already staged)
        if (tid < 64) sm.dgk[tid] *= exp2f(sm.gn[tid]);
        {
            for (int e = 0; e < size(acc_dq); ++e) {
                auto crd = tCcC(e);
                int i = get<0>(crd), kk = get<1>(crd);
                bool rv = i < rows_valid;
                float g_ik = sm.sp.gk.g[i * 68 + kk];
                float k_ik = to_f32(sm.sp.gk.k[i * kCP + kk]);
                float e2g = exp2f(g_ik);
                acc_dq(e) *= e2g * scale;
                acc_dk(e) *= rv ? exp2f(sm.gn[kk] - g_ik) : 0.f;
                float dw_v = -acc_dw(e);
                sm.ep.dw[i * kCP + kk] = T(dw_v);
                sm.ep.dwT[kk * kCP + i] = T(dw_v);
                sm.ep.kg[i * kCP + kk] = T(k_ik * e2g);
            }
        }
        __syncthreads();
        gemm_sm(acc_dA, sm.ep.dw, sm.ep.kg);       // dA += (-dw) @ kg^T
        Tensor acc_dkgb = partition_fragment_C(tiled_mma, Shape<Int<kBT>, _64>{});
        clear(acc_dkgb);
        gemm_sm(acc_dkgb, sm.A_t, sm.ep.dwT);      // dkgb = Akk^T @ (-dw)
        __syncthreads();  // dw/dwT/kg consumed; dw slot is dead

        // stage q into the dead dw slot, overlapped with the db/dgk phase
        {
            T const* qsrc = q_g + qk_base + (int64_t)t0 * qk_row + i_k * kBK;
            stage_tile(sm.ep.dw, qsrc, q_g, qk_row, rows_valid, tid);
            cp_async_commit();
        }

        // db += sum_k dkgb * kg; dgk[k] += sum_i k * dk (dk before the dkgb term)
        // in-thread partial sums: e bit1 selects the row half (2 rows), e bit0
        // and bits 2+ select the column (8 cols) -> one atomic per row/col.
        {
            float db_acc[2] = {0.f, 0.f};
            float dgk_acc[8] = {};
            CUTE_UNROLL
            for (int e = 0; e < size(acc_dkgb); ++e) {
                auto crd = tCcC(e);
                int i = get<0>(crd), kk = get<1>(crd);
                float g_ik = sm.sp.gk.g[i * 68 + kk];
                float k_ik = to_f32(sm.sp.gk.k[i * kCP + kk]);
                db_acc[(e >> 1) & 1] += acc_dkgb(e) * (k_ik * exp2f(g_ik));
                dgk_acc[(e & 1) + 2 * (e >> 2)] += k_ik * acc_dk(e);
            }
            CUTE_UNROLL
            for (int hh = 0; hh < 2; ++hh)
                atomicAdd(&sm.db[get<0>(tCcC(2 * hh))], db_acc[hh]);
            CUTE_UNROLL
            for (int cc = 0; cc < 8; ++cc)
                atomicAdd(&sm.dgk[get<1>(tCcC((cc & 1) + 4 * (cc >> 1)))], dgk_acc[cc]);
        }
        cp_async_wait<0>();
        __syncthreads();

        // dg/dk composition, store dq/dk/dg (adjacent-column pairs as float2)
        {
            float* dq_p = dq_g + hvK_base + (int64_t)t0 * hvK_row + i_k * kBK;
            float* dk_p = dk_g + hvK_base + (int64_t)t0 * hvK_row + i_k * kBK;
            float* dg_p = dg_g + hvK_base + (int64_t)t0 * hvK_row + i_k * kBK;
            CUTE_UNROLL
            for (int e = 0; e < size(acc_dq); e += 2) {
                auto crd = tCcC(e);
                int i = get<0>(crd), kk = get<1>(crd);
                if (i >= rows_valid) continue;
                float2 fq, fk, fg;
                CUTE_UNROLL
                for (int j = 0; j < 2; ++j) {
                    float g_ik = sm.sp.gk.g[i * 68 + kk + j];
                    float e2g = exp2f(g_ik);
                    float k_ik = to_f32(sm.sp.gk.k[i * kCP + kk + j]);
                    float kg = k_ik * e2g;
                    float kdk = k_ik * acc_dk(e + j);
                    float dg_v = to_f32(sm.ep.dw[i * kCP + kk + j]) * acc_dq(e + j) - kdk
                               + (t0 + i == last ? sm.dgk[kk + j] : 0.f)
                               + kg * acc_dkgb(e + j) * sm.beta[i];
                    float dk_v = acc_dk(e + j) + acc_dkgb(e + j) * e2g * sm.beta[i];
                    (&fq.x)[j] = acc_dq(e + j);
                    (&fk.x)[j] = dk_v;
                    (&fg.x)[j] = dg_v;
                }
                int64_t off = (int64_t)i * hvK_row + kk;
                *reinterpret_cast<float2*>(dq_p + off) = fq;
                *reinterpret_cast<float2*>(dk_p + off) = fk;
                *reinterpret_cast<float2*>(dg_p + off) = fg;
            }
        }
    }

    // dA postprocess: strict lower mask, column beta, then dAkk = -A^T (dA . beta) A^T
    __syncthreads();
    for (int e = 0; e < size(acc_dA); ++e) {
        auto crd = tCcC(e);
        int i = get<0>(crd), j = get<1>(crd);
        bool m = (i > j) && (i < rows_valid) && (j < rows_valid);
        sm.pp.dAm[i * kCP + j] = T(m ? acc_dA(e) * sm.beta[j] : 0.f);
    }
    __syncthreads();
    Tensor acc1 = partition_fragment_C(tiled_mma, Shape<Int<kBT>, _64>{});
    clear(acc1);
    gemm_sm_bt(acc1, sm.pp.dAm, sm.A_t);  // dAm @ Akk^T (A_t[c][j] = Akk[j][c])
    __syncthreads();
    for (int e = 0; e < size(acc1); ++e) {
        auto crd = tCcC(e);
        int i = get<0>(crd), j = get<1>(crd);
        sm.pp.c1T[j * kCP + i] = T(acc1(e));
    }
    __syncthreads();
    Tensor acc2 = partition_fragment_C(tiled_mma, Shape<Int<kBT>, _64>{});
    clear(acc2);
    gemm_sm(acc2, sm.A_t, sm.pp.c1T);  // Akk^T @ (...)
    {
        float* dA_p = dA_g + A_base + (int64_t)t0 * A_row;
        for (int e = 0; e < size(acc2); ++e) {
            auto crd = tCcC(e);
            int i = get<0>(crd), j = get<1>(crd);
            if (i < rows_valid) {
                bool m = (i > j) && (j < rows_valid);
                dA_p[(int64_t)i * A_row + j] = m ? -acc2(e) : 0.f;
            }
        }
    }
    if (tid < kBT && tid < rows_valid)
        db_g[beta_base + (int64_t)(t0 + tid) * HV] = sm.db[tid];
}

template <typename T, int K, int V>
void launch_fused(
    T const* q, T const* k, T const* v, T const* v_new,
    float const* g, float const* beta, T const* A, T const* h,
    T const* do_, T const* dh, T const* dv,
    float* dq, float* dk, T* dv2, float* dg, float* db, float* dA,
    int64_t const* cu_seqlens, int64_t const* chunk_indices,
    float scale, int64_t B, int64_t T_len, int64_t H, int64_t HV, int64_t NT,
    bool state_v_first, cudaStream_t stream
) {
    auto* kern = chunk_kda_bwd_wy_dqkg_fused_kernel<T, K, V>;
    static bool configured = false;
    if (!configured) {
        cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize,
                             int(sizeof(FusedSmem<T>)));
        configured = true;
    }
    dim3 grid(unsigned(NT), unsigned(B * HV));
    kern<<<grid, kThreads, sizeof(FusedSmem<T>), stream>>>(
        q, k, v, v_new, g, beta, A, h, do_, dh, dv,
        dq, dk, dv2, dg, db, dA,
        cu_seqlens, chunk_indices, scale, int(T_len), int(H), int(HV),
        state_v_first, cu_seqlens != nullptr
    );
}

template <typename T>
void dispatch_kv(
    torch::Tensor const& q, torch::Tensor const& k, torch::Tensor const& v,
    torch::Tensor const& v_new, torch::Tensor const& g, torch::Tensor const& beta,
    torch::Tensor const& A, torch::Tensor const& h, torch::Tensor const& do_,
    torch::Tensor const& dh, torch::Tensor const& dv,
    torch::Tensor& dq, torch::Tensor& dk, torch::Tensor& dv2,
    torch::Tensor& dg, torch::Tensor& db, torch::Tensor& dA,
    int64_t const* cu_seqlens, int64_t const* chunk_indices,
    float scale, int64_t B, int64_t T_len, int64_t H, int64_t HV, int64_t NT,
    bool state_v_first, cudaStream_t stream
) {
    int64_t K = k.size(3), V = v.size(3);
    #define LAUNCH_KV(KK, VV) \
        launch_fused<T, KK, VV>( \
            reinterpret_cast<T const*>(q.data_ptr()), reinterpret_cast<T const*>(k.data_ptr()), \
            reinterpret_cast<T const*>(v.data_ptr()), reinterpret_cast<T const*>(v_new.data_ptr()), \
            g.data_ptr<float>(), beta.data_ptr<float>(), \
            reinterpret_cast<T const*>(A.data_ptr()), reinterpret_cast<T const*>(h.data_ptr()), \
            reinterpret_cast<T const*>(do_.data_ptr()), reinterpret_cast<T const*>(dh.data_ptr()), \
            reinterpret_cast<T const*>(dv.data_ptr()), \
            dq.data_ptr<float>(), dk.data_ptr<float>(), reinterpret_cast<T*>(dv2.data_ptr()), \
            dg.data_ptr<float>(), db.data_ptr<float>(), dA.data_ptr<float>(), \
            cu_seqlens, chunk_indices, scale, B, T_len, H, HV, NT, state_v_first, stream)
    if (K == 128 && V == 128) { LAUNCH_KV(128, 128); }
    else if (K == 64 && V == 64) { LAUNCH_KV(64, 64); }
    else if (K == 128 && V == 64) { LAUNCH_KV(128, 64); }
    else if (K == 64 && V == 128) { LAUNCH_KV(64, 128); }
    else { TORCH_CHECK(false, "unsupported K/V: ", K, "/", V, " (must be 64 or 128)"); }
    #undef LAUNCH_KV
}

}  // namespace

// fla/ops/kda/chunk_bwd.py::chunk_kda_bwd_wy_dqkg_fused host wrapper.
// Returns (dq, dk, dv2, db, dg, dAkk) in the same order as fla.
std::vector<torch::Tensor> chunk_kda_bwd_wy_dqkg_fused(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor v_new,
    torch::Tensor g,
    torch::Tensor beta,
    torch::Tensor A,
    torch::Tensor h,
    torch::Tensor do_,
    torch::Tensor dh,
    torch::Tensor dv,
    double scale,
    bool state_v_first,
    std::optional<torch::Tensor> cu_seqlens,
    std::optional<torch::Tensor> chunk_indices,
    int64_t chunk_size
) {
    TORCH_CHECK(k.is_cuda() && k.is_contiguous(), "k must be contiguous CUDA tensor");
    TORCH_CHECK(k.dim() == 4, "k must be [B, T, H, K]");
    TORCH_CHECK(chunk_size == kBT, "only chunk_size=64 is supported");
    int64_t B = k.size(0), T_len = k.size(1), H = k.size(2), K = k.size(3);
    int64_t HV = v.size(2), V = v.size(3);
    TORCH_CHECK(K % kBK == 0 && V % kBV == 0, "K and V must be multiples of 64");
    TORCH_CHECK(HV % H == 0, "HV must be a multiple of H");
    TORCH_CHECK(q.scalar_type() == k.scalar_type() && v.scalar_type() == k.scalar_type() &&
                v_new.scalar_type() == k.scalar_type() && do_.scalar_type() == k.scalar_type() &&
                dh.scalar_type() == k.scalar_type() && dv.scalar_type() == k.scalar_type() &&
                A.scalar_type() == k.scalar_type() && h.scalar_type() == k.scalar_type(),
                "q/k/v/v_new/do/dh/dv/A/h must share the same dtype");
    TORCH_CHECK(k.scalar_type() == at::kBFloat16 || k.scalar_type() == at::kHalf,
                "only bf16/fp16 are supported");
    TORCH_CHECK(g.is_cuda() && g.is_contiguous() && g.scalar_type() == at::kFloat, "g must be fp32");
    TORCH_CHECK(beta.is_cuda() && beta.is_contiguous() && beta.scalar_type() == at::kFloat, "beta must be fp32");
    for (auto const* t : {&q, &v, &v_new, &A, &h, &do_, &dh, &dv}) {
        TORCH_CHECK(t->is_cuda() && t->is_contiguous(), "all inputs must be contiguous CUDA tensors");
    }

    bool is_varlen = cu_seqlens.has_value();
    int64_t NT;
    int64_t const* cu_ptr = nullptr;
    int64_t const* ci_ptr = nullptr;
    if (is_varlen) {
        TORCH_CHECK(B == 1, "B must be 1 when cu_seqlens is provided");
        TORCH_CHECK(chunk_indices.has_value(), "chunk_indices must be provided with cu_seqlens");
        auto const& cu = cu_seqlens.value();
        auto const& ci = chunk_indices.value();
        TORCH_CHECK(cu.scalar_type() == torch::kLong && cu.is_cuda() && cu.is_contiguous());
        TORCH_CHECK(ci.scalar_type() == torch::kLong && ci.is_cuda() && ci.is_contiguous());
        TORCH_CHECK(ci.dim() == 2 && ci.size(1) == 2);
        cu_ptr = cu.data_ptr<int64_t>();
        ci_ptr = ci.data_ptr<int64_t>();
        NT = ci.size(0);
    } else {
        NT = (T_len + kBT - 1) / kBT;
    }

    auto opts_f = g.options().dtype(at::kFloat);
    torch::Tensor dq = torch::empty({B, T_len, HV, K}, opts_f);
    torch::Tensor dk = torch::empty({B, T_len, HV, K}, opts_f);
    torch::Tensor dv2 = torch::empty_like(v);
    torch::Tensor dg = torch::empty_like(g, opts_f);
    torch::Tensor db = torch::empty_like(beta, opts_f);
    torch::Tensor dA = torch::empty_like(A, opts_f);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    if (k.scalar_type() == at::kBFloat16) {
        dispatch_kv<cutlass::bfloat16_t>(
            q, k, v, v_new, g, beta, A, h, do_, dh, dv, dq, dk, dv2, dg, db, dA,
            cu_ptr, ci_ptr, float(scale), B, T_len, H, HV, NT, state_v_first, stream);
    } else {
        dispatch_kv<cutlass::half_t>(
            q, k, v, v_new, g, beta, A, h, do_, dh, dv, dq, dk, dv2, dg, db, dA,
            cu_ptr, ci_ptr, float(scale), B, T_len, H, HV, NT, state_v_first, stream);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {dq, dk, dv2, db, dg, dA};
}
