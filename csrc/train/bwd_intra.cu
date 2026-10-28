// KDA backward: intra-chunk (sub-chunk) gradients accumulated into dq/dk/db/dg.
// Replicates fla/ops/kda/chunk_intra.py::chunk_kda_bwd_kernel_intra.
//
// Each block handles one (k-slice i_k, sub-chunk i_i, chunk i_t, batch*head) tile of
// shape [BC=16, BK=64]. The kernel has three parts, matching the Triton source:
//   (a) contributions from previous sub-chunks of the chunk (query/key side)
//   (b) the diagonal sub-chunk, i<=j masked side
//   (c) the reverse (key side) contributions, incl. the i>=j masked diagonal side
// All MMAs run on tensor cores (SM80 16x8x8 tf32 atoms, fp32 accumulators); the Triton
// reference computes these dots from fp32 inputs with tf32 precision.
//
// BK=64 with 256 threads (8 warps, one 1x8 tiled mma) halves the dAqk/dAkk gmem
// traffic per output element versus BK=32 and keeps all staging loads/stores on
// 16B vectors. Per-element math and accumulation order are unchanged from the
// BK=32 version (each output element is computed by the identical op sequence).

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>

#include <cutlass/tfloat32.h>
#include <cute/tensor.hpp>

#include "common.cuh"

namespace kda_impl {

using namespace cute;

constexpr int kBT = 64;  // chunk size (KDA bwd is only ever run with 64)
constexpr int kBC = 16;  // sub-chunk size
constexpr int kBK = 64;  // K tile
constexpr int kThreads = 256;

__device__ __forceinline__ cutlass::tfloat32_t to_tf32(float x) {
    return cutlass::tfloat32_t(x);
}

__device__ __forceinline__ uint4 pack_tf32(float a, float b, float c, float d) {
    return make_uint4(to_tf32(a).storage, to_tf32(b).storage, to_tf32(c).storage, to_tf32(d).storage);
}

// ---------------------------------------------------------------------------
// staging helpers (vector path requires K % kBK == 0 so every row segment is
// in-bounds and 16B-aligned; the scalar fallbacks keep arbitrary K working)

// fp32 gmem tile [kBC][kBK] -> fp32 smem, rows past rows_valid / cols past
// cols_valid zero-filled (vec path requires the full row segment in-bounds).
__device__ __forceinline__ void stage_f32(float* s, float const* g, int64_t row_stride,
                                          int rows_valid, int cols_valid, bool vec_ok, int tid) {
    if (vec_ok) {
        constexpr int kCG = kBK / 4;
        CUTE_UNROLL
        for (int idx = tid; idx < kBC * kCG; idx += kThreads) {
            int r = idx / kCG;
            float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
            if (r < rows_valid) v = *reinterpret_cast<float4 const*>(g + (int64_t)r * row_stride + (idx % kCG) * 4);
            *reinterpret_cast<float4*>(s + idx * 4) = v;
        }
    } else {
        for (int idx = tid; idx < kBC * kBK; idx += kThreads) {
            int r = idx / kBK, c = idx % kBK;
            s[idx] = (r < rows_valid && c < cols_valid) ? g[(int64_t)r * row_stride + c] : 0.f;
        }
    }
}

// bf16/fp16 gmem tile [kBC][kBK] -> fp32 smem (converted), zero-filled past
// rows_valid / cols_valid.
template <typename T>
__device__ __forceinline__ void stage_t_f32(float* s, T const* g, int64_t row_stride,
                                            int rows_valid, int cols_valid, bool vec_ok, int tid) {
    if (vec_ok) {
        constexpr int kCG = kBK / 8;
        CUTE_UNROLL
        for (int idx = tid; idx < kBC * kCG; idx += kThreads) {
            int r = idx / kCG, c = (idx % kCG) * 8;
            float vals[8] = {};
            if (r < rows_valid) {
                uint4 raw = *reinterpret_cast<uint4 const*>(g + (int64_t)r * row_stride + c);
                T const* h = reinterpret_cast<T const*>(&raw);
                CUTE_UNROLL
                for (int j = 0; j < 8; ++j) vals[j] = to_f32(h[j]);
            }
            float4* d = reinterpret_cast<float4*>(s + r * kBK + c);
            d[0] = make_float4(vals[0], vals[1], vals[2], vals[3]);
            d[1] = make_float4(vals[4], vals[5], vals[6], vals[7]);
        }
    } else {
        for (int idx = tid; idx < kBC * kBK; idx += kThreads) {
            int r = idx / kBK, c = idx % kBK;
            s[idx] = (r < rows_valid && c < cols_valid) ? to_f32(g[(int64_t)r * row_stride + c]) : 0.f;
        }
    }
}

// dA tile [kBC][kBC] fp32 gmem -> tf32 smem, natural orientation:
// s[r][jj] = dA[r * row_stride + col0 + jj], rows past rows_valid zero-filled.
__device__ __forceinline__ void stage_dA(cutlass::tfloat32_t* sA, float const* dA,
                                         int64_t row_stride, int col0, int rows_valid, int tid) {
    CUTE_UNROLL
    for (int idx = tid; idx < kBC * (kBC / 4); idx += kThreads) {
        int r = idx / (kBC / 4), c = (idx % (kBC / 4)) * 4;
        float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
        if (r < rows_valid) v = *reinterpret_cast<float4 const*>(dA + (int64_t)r * row_stride + col0 + c);
        *reinterpret_cast<uint4*>(sA + r * kBC + c) = pack_tf32(v.x, v.y, v.z, v.w);
    }
}

// dA tile transposed: s[r][jj] = dA[jj * row_stride + col0 + r], so the gmem
// reads coalesce along r (the tile's column dim). Rows (jj) past rows_valid
// are zero-filled.
__device__ __forceinline__ void stage_dA_T(cutlass::tfloat32_t* sA, float const* dA,
                                           int64_t row_stride, int col0, int rows_valid, int tid) {
    CUTE_UNROLL
    for (int idx = tid; idx < kBC * (kBC / 4); idx += kThreads) {
        int jj = idx / (kBC / 4), r = (idx % (kBC / 4)) * 4;
        float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
        if (jj < rows_valid) v = *reinterpret_cast<float4 const*>(dA + (int64_t)jj * row_stride + col0 + r);
        sA[r * kBC + jj] = to_tf32(v.x);
        sA[(r + 1) * kBC + jj] = to_tf32(v.y);
        sA[(r + 2) * kBC + jj] = to_tf32(v.z);
        sA[(r + 3) * kBC + jj] = to_tf32(v.w);
    }
}

// Grid: (NK*NC, NT, B*HV).
template <typename T, bool IS_VARLEN, bool SAFE_GATE>
__global__ void __launch_bounds__(kThreads, 3) chunk_kda_bwd_intra_kernel(
    T const* __restrict__ q,
    T const* __restrict__ k,
    float const* __restrict__ g,
    float const* __restrict__ beta,
    float const* __restrict__ dAqk,
    float const* __restrict__ dAkk,
    float const* __restrict__ dq,
    float* __restrict__ dq2,
    float const* __restrict__ dk,
    float* __restrict__ dk2,
    float const* __restrict__ dg,
    float* __restrict__ dg2,
    float* __restrict__ db2,
    int64_t const* __restrict__ cu_seqlens,
    int64_t const* __restrict__ chunk_indices,
    int64_t allT,
    int T_len, int H, int HV, int K, int NC
) {
    int const i_kc = blockIdx.x;
    int64_t const i_t = blockIdx.y;
    int64_t const i_bh = blockIdx.z;
    int const i_k = i_kc / NC, i_i = i_kc % NC;
    int64_t const i_hv = i_bh % HV;
    int64_t const i_h = i_hv / (HV / H);

    int64_t bos, t_chunk;
    int64_t seq_len = T_len;
    if (IS_VARLEN) {
        int64_t const i_n = chunk_indices[i_t * 2];
        int64_t const i_tl = chunk_indices[i_t * 2 + 1];
        bos = cu_seqlens[i_n];
        seq_len = cu_seqlens[i_n + 1] - bos;
        t_chunk = i_tl * kBT;
    } else {
        bos = (i_bh / HV) * (int64_t)T_len;
        t_chunk = i_t * kBT;
    }
    int const seq = (int)seq_len;

    int const i_ti = (int)t_chunk + i_i * kBC;
    if (i_ti >= seq) return;

    int64_t const HK = (int64_t)H * K;
    int64_t const HVK = (int64_t)HV * K;
    int64_t const HVBT = (int64_t)HV * kBT;

    T const* q_base = q + (bos * H + i_h) * K;
    T const* k_base = k + (bos * H + i_h) * K;
    float const* g_base = g + (bos * HV + i_hv) * K;
    float const* beta_base = beta + bos * HV + i_hv;
    float const* dAqk_base = dAqk + (bos * HV + i_hv) * kBT;
    float const* dAkk_base = dAkk + (bos * HV + i_hv) * kBT;
    float const* dq_base = dq + (bos * HV + i_hv) * K;
    float* dq2_base = dq2 + (bos * HV + i_hv) * K;
    float const* dk_base = dk + (bos * HV + i_hv) * K;
    float* dk2_base = dk2 + (bos * HV + i_hv) * K;
    float const* dg_base = dg + (bos * HV + i_hv) * K;
    float* dg2_base = dg2 + (bos * HV + i_hv) * K;
    float* db2_base = db2 + (i_kc / NC * allT + bos) * HV + i_hv;

    __shared__ __align__(16) float s_g[kBC * kBK];    // this sub-chunk's g rows (fp32)
    __shared__ __align__(16) float s_q[kBC * kBK];
    __shared__ __align__(16) float s_k[kBC * kBK];
    __shared__ float s_beta[kBC];
    __shared__ __align__(16) float s_dq2[kBC * kBK];  // fp32 staging tiles for elementwise phases
    __shared__ __align__(16) float s_dk2[kBC * kBK];
    __shared__ __align__(16) float s_dg2[kBC * kBK];
    // s_dq2 is dead after the dq2/beta-scale phase below, so the (c)-loop B2
    // operand and the final dkt staging tile overlap it (saves 8KB -> 3 CTAs/SM).
    __shared__ __align__(16) cutlass::tfloat32_t s_A[kBC * kBC];   // MMA A operand (M=c, K=j)
    __shared__ __align__(16) cutlass::tfloat32_t s_A2[kBC * kBC];
    __shared__ __align__(16) cutlass::tfloat32_t s_B[kBC * kBK];   // MMA B operand, stored [j][n], viewed (N=n, K=j)
    float* s_dkt = s_dq2;
    cutlass::tfloat32_t* s_B2 = reinterpret_cast<cutlass::tfloat32_t*>(s_dq2);
    __shared__ float s_colA[kBC], s_colB[kBC];       // scalar-loop staging (non-safe paths)
    __shared__ float s_rowq[kBK], s_rowk[kBK], s_rowg[kBK];

    int const tid = threadIdx.x;
    int const col0 = i_k * kBK;
    bool const vec_ok = (K % kBK) == 0;
    int const rows_valid_c = min(kBC, seq - i_ti);  // i_ti < seq, so >= 1
    int const cols_valid = min(kBK, K - col0);

    stage_f32(s_g, g_base + (int64_t)i_ti * HVK + col0, HVK, rows_valid_c, cols_valid, vec_ok, tid);
    stage_t_f32(s_q, q_base + (int64_t)i_ti * HK + col0, HK, rows_valid_c, cols_valid, vec_ok, tid);
    stage_t_f32(s_k, k_base + (int64_t)i_ti * HK + col0, HK, rows_valid_c, cols_valid, vec_ok, tid);
    if (tid < kBC) {
        int t = i_ti + tid;
        s_beta[tid] = (t < seq) ? beta_base[(int64_t)t * HV] : 0.0f;
    }
    __syncthreads();

    // 8 warps split the N=64 dimension of the [16,64] output tile.
    auto mma = make_tiled_mma(SM80_16x8x8_F32TF32TF32F32_TN{}, Layout<Shape<_1, _8, _1>>{});
    auto thr_mma = mma.get_thread_slice(tid);

    Tensor sA_t = make_tensor(make_smem_ptr(s_A), Layout<Shape<_16, _16>, Stride<_16, _1>>{});
    Tensor sA2_t = make_tensor(make_smem_ptr(s_A2), Layout<Shape<_16, _16>, Stride<_16, _1>>{});
    Tensor sB_t = make_tensor(make_smem_ptr(s_B), Layout<Shape<Int<kBK>, _16>, Stride<_1, Int<kBK>>>{});
    Tensor sB2_t = make_tensor(make_smem_ptr(s_B2), Layout<Shape<Int<kBK>, _16>, Stride<_1, Int<kBK>>>{});

    // Identity tensors give the (m,k)/(n,k)/(m,n) coordinate of each fragment element;
    // fragments themselves are shaped from the fp32 smem staging tiles.
    Tensor cA = make_identity_tensor(Shape<_16, _16>{});
    Tensor cB = make_identity_tensor(Shape<Int<kBK>, _16>{});
    Tensor cC = make_identity_tensor(Shape<_16, Int<kBK>>{});
    Tensor tCcA = thr_mma.partition_A(cA);
    Tensor tCcB = thr_mma.partition_B(cB);
    Tensor tCcC = thr_mma.partition_C(cC);

    Tensor sC_t = make_tensor(make_smem_ptr(s_dq2), Layout<Shape<_16, Int<kBK>>, Stride<Int<kBK>, _1>>{});

    Tensor tCrA = thr_mma.partition_fragment_A(sA_t);
    Tensor tCrA2 = thr_mma.partition_fragment_A(sA2_t);
    Tensor tCrB = thr_mma.partition_fragment_B(sB_t);
    Tensor tCrB2 = thr_mma.partition_fragment_B(sB2_t);

    Tensor fdq2 = thr_mma.partition_fragment_C(sC_t);
    Tensor fdk2 = thr_mma.partition_fragment_C(sC_t);
    Tensor fdkt = thr_mma.partition_fragment_C(sC_t);
    Tensor ftmp = thr_mma.partition_fragment_C(sC_t);
    Tensor ftmp2 = thr_mma.partition_fragment_C(sC_t);
    clear(fdq2);
    clear(fdk2);
    clear(fdkt);

    auto load_frag = [](auto& frag, auto const& coords, auto const& s_t) {
        for (int i = 0; i < size(frag); ++i)
            frag(i) = s_t(get<0>(coords(i)), get<1>(coords(i)));
    };

    // (a) contributions from previous sub-chunks: dq2/dk2 += dA[c,j] @ (k_j * 2^(gn-g_j))
    if (i_i > 0) {
        for (int i_j = 0; i_j < i_i; ++i_j) {
            int const j0 = (int)t_chunk + i_j * kBC;
            int const rows_valid_j = min(kBC, seq - j0);
            stage_dA(s_A, dAqk_base + (int64_t)i_ti * HVBT + i_j * kBC, HVBT, 0, rows_valid_c, tid);
            stage_dA(s_A2, dAkk_base + (int64_t)i_ti * HVBT + i_j * kBC, HVBT, 0, rows_valid_c, tid);
            if (vec_ok) {
                constexpr int kCG = kBK / 4;
                CUTE_UNROLL
                for (int idx = tid; idx < kBC * kCG; idx += kThreads) {
                    int jj = idx / kCG, n = (idx % kCG) * 4;
                    int tj = j0 + jj;
                    float kv[4] = {}, gv[4] = {};
                    if (tj < seq) {
                        uint2 rawk = *reinterpret_cast<uint2 const*>(k_base + (int64_t)tj * HK + col0 + n);
                        T const* hk = reinterpret_cast<T const*>(&rawk);
                        float4 g4 = *reinterpret_cast<float4 const*>(g_base + (int64_t)tj * HVK + col0 + n);
                        kv[0] = to_f32(hk[0]); kv[1] = to_f32(hk[1]); kv[2] = to_f32(hk[2]); kv[3] = to_f32(hk[3]);
                        gv[0] = g4.x; gv[1] = g4.y; gv[2] = g4.z; gv[3] = g4.w;
                    }
                    float4 gn = *reinterpret_cast<float4 const*>(s_g + n);  // s_g row 0 is g[i_ti] (gn)
                    *reinterpret_cast<uint4*>(s_B + jj * kBK + n) = pack_tf32(
                        kv[0] * exp2f(gn.x - gv[0]), kv[1] * exp2f(gn.y - gv[1]),
                        kv[2] * exp2f(gn.z - gv[2]), kv[3] * exp2f(gn.w - gv[3]));
                }
            } else {
                for (int idx = tid; idx < kBC * kBK; idx += kThreads) {
                    int jj = idx / kBK, c = idx % kBK;
                    int tj = j0 + jj, col = col0 + c;
                    bool valid = (tj < seq) && (col < K);
                    float kv = valid ? to_f32(k_base[(int64_t)tj * HK + col]) : 0.0f;
                    float gv = valid ? g_base[(int64_t)tj * HVK + col] : 0.0f;
                    s_B[idx] = to_tf32(kv * exp2f(s_g[c] - gv));
                }
            }
            __syncthreads();
            load_frag(tCrA, tCcA, sA_t);
            load_frag(tCrA2, tCcA, sA2_t);
            load_frag(tCrB, tCcB, sB_t);
            cute::gemm(mma, tCrA, tCrB, fdq2);
            cute::gemm(mma, tCrA2, tCrB, fdk2);
            __syncthreads();
        }
        for (int i = 0; i < size(fdq2); ++i) {
            int m = get<0>(tCcC(i)), n = get<1>(tCcC(i));
            float e = exp2f(s_g[m * kBK + n] - s_g[n]);
            fdq2(i) *= e;
            fdk2(i) *= e;
        }
    }

    // (b) diagonal sub-chunk, i <= j masked side
    int const mid = min(kBC / 2, seq - i_ti - 1);  // safe_gate midpoint reference row
    if (SAFE_GATE) {
        {
            // masked dA loads: keep = (r >= jj) && both tokens valid
            CUTE_UNROLL
            for (int idx = tid; idx < kBC * (kBC / 4); idx += kThreads) {
                int r = idx / (kBC / 4), c = (idx % (kBC / 4)) * 4;
                float4 v1 = make_float4(0.f, 0.f, 0.f, 0.f), v2 = make_float4(0.f, 0.f, 0.f, 0.f);
                if (i_ti + r < seq) {
                    v1 = *reinterpret_cast<float4 const*>(dAqk_base + (int64_t)(i_ti + r) * HVBT + i_i * kBC + c);
                    v2 = *reinterpret_cast<float4 const*>(dAkk_base + (int64_t)(i_ti + r) * HVBT + i_i * kBC + c);
                }
                float va1[4] = {v1.x, v1.y, v1.z, v1.w}, va2[4] = {v2.x, v2.y, v2.z, v2.w};
                CUTE_UNROLL
                for (int j = 0; j < 4; ++j) {
                    bool keep = (r >= c + j) && (i_ti + r < seq) && (i_ti + c + j < seq);
                    va1[j] = keep ? va1[j] : 0.f;
                    va2[j] = keep ? va2[j] : 0.f;
                }
                *reinterpret_cast<uint4*>(s_A + r * kBC + c) = pack_tf32(va1[0], va1[1], va1[2], va1[3]);
                *reinterpret_cast<uint4*>(s_A2 + r * kBC + c) = pack_tf32(va2[0], va2[1], va2[2], va2[3]);
            }
            constexpr int kCG = kBK / 4;
            CUTE_UNROLL
            for (int idx = tid; idx < kBC * kCG; idx += kThreads) {
                int j = idx / kCG, c = (idx % kCG) * 4;
                float4 kj = *reinterpret_cast<float4 const*>(s_k + j * kBK + c);
                float4 gj = *reinterpret_cast<float4 const*>(s_g + j * kBK + c);
                float4 gm = *reinterpret_cast<float4 const*>(s_g + mid * kBK + c);
                float o[4] = {0.f, 0.f, 0.f, 0.f};
                if (i_ti + j < seq) {
                    o[0] = kj.x * exp2f(-(gj.x - gm.x));
                    o[1] = kj.y * exp2f(-(gj.y - gm.y));
                    o[2] = kj.z * exp2f(-(gj.z - gm.z));
                    o[3] = kj.w * exp2f(-(gj.w - gm.w));
                }
                *reinterpret_cast<uint4*>(s_B + j * kBK + c) = pack_tf32(o[0], o[1], o[2], o[3]);
            }
        }
        __syncthreads();
        load_frag(tCrA, tCcA, sA_t);
        load_frag(tCrA2, tCcA, sA2_t);
        load_frag(tCrB, tCcB, sB_t);
        clear(ftmp);
        clear(ftmp2);
        cute::gemm(mma, tCrA, tCrB, ftmp);
        cute::gemm(mma, tCrA2, tCrB, ftmp2);
        for (int i = 0; i < size(fdq2); ++i) {
            int m = get<0>(tCcC(i)), n = get<1>(tCcC(i));
            float e = (i_ti + m < seq) ? exp2f(s_g[m * kBK + n] - s_g[mid * kBK + n]) : 0.0f;
            fdq2(i) += ftmp(i) * e;
            fdk2(i) += ftmp2(i) * e;
        }
        __syncthreads();
    } else {
        int const jmax = min(kBC, seq - i_ti);
        for (int j = 0; j < jmax; ++j) {
            if (tid < kBC) {
                int t = i_ti + tid;
                int64_t off = (int64_t)t * HVBT + i_i * kBC + j;
                s_colA[tid] = (t < seq) ? dAqk_base[off] : 0.0f;
                s_colB[tid] = (t < seq) ? dAkk_base[off] : 0.0f;
            } else if (tid < kBC + kBK) {
                int c = tid - kBC;
                int col = col0 + c;
                s_rowk[c] = (col < K) ? to_f32(k_base[(int64_t)(i_ti + j) * HK + col]) : 0.0f;
                s_rowg[c] = (col < K) ? g_base[(int64_t)(i_ti + j) * HVK + col] : 0.0f;
            }
            __syncthreads();
            for (int i = 0; i < size(fdq2); ++i) {
                int m = get<0>(tCcC(i)), n = get<1>(tCcC(i));
                if (m >= j) {
                    float e = exp2f(s_g[m * kBK + n] - s_rowg[n]);
                    fdq2(i) += s_colA[m] * s_rowk[n] * e;
                    fdk2(i) += s_colB[m] * s_rowk[n] * e;
                }
            }
            __syncthreads();
        }
    }

    // db is the row-sum of dk2*k BEFORE the beta scaling; dq2/dg2 use pre-add values.
    for (int i = 0; i < size(fdq2); ++i) {
        int m = get<0>(tCcC(i)), n = get<1>(tCcC(i));
        s_dq2[m * kBK + n] = fdq2(i);
        s_dk2[m * kBK + n] = fdk2(i);
    }
    __syncthreads();
    if (tid < kBC) {
        float acc = 0.0f;
        CUTE_UNROLL
        for (int c4 = 0; c4 < kBK / 4; ++c4) {
            float4 dv = *reinterpret_cast<float4 const*>(s_dk2 + tid * kBK + c4 * 4);
            float4 kv = *reinterpret_cast<float4 const*>(s_k + tid * kBK + c4 * 4);
            acc += dv.x * kv.x;
            acc += dv.y * kv.y;
            acc += dv.z * kv.z;
            acc += dv.w * kv.w;
        }
        int t = i_ti + tid;
        if (t < seq) db2_base[(int64_t)t * HV] = acc;
    }
    __syncthreads();  // db must read dk2 before the beta scaling below overwrites it
    if (vec_ok) {
        constexpr int kCG = kBK / 4;
        CUTE_UNROLL
        for (int idx = tid; idx < kBC * kCG; idx += kThreads) {
            int r = idx / kCG, c = (idx % kCG) * 4;
            int t = i_ti + r;
            float4 dq2v = *reinterpret_cast<float4 const*>(s_dq2 + r * kBK + c);
            float4 qv = *reinterpret_cast<float4 const*>(s_q + r * kBK + c);
            float4 dk2v = *reinterpret_cast<float4 const*>(s_dk2 + r * kBK + c);
            *reinterpret_cast<float4*>(s_dg2 + r * kBK + c) =
                make_float4(qv.x * dq2v.x, qv.y * dq2v.y, qv.z * dq2v.z, qv.w * dq2v.w);
            if (t < seq) {
                int64_t off = (int64_t)t * HVK + col0 + c;
                float4 dqv = *reinterpret_cast<float4 const*>(dq_base + off);
                *reinterpret_cast<float4*>(dq2_base + off) = make_float4(
                    dq2v.x + dqv.x, dq2v.y + dqv.y, dq2v.z + dqv.z, dq2v.w + dqv.w);
            }
            float b = s_beta[r];
            *reinterpret_cast<float4*>(s_dk2 + r * kBK + c) =
                make_float4(dk2v.x * b, dk2v.y * b, dk2v.z * b, dk2v.w * b);
        }
    } else {
        for (int idx = tid; idx < kBC * kBK; idx += kThreads) {
            int r = idx / kBK, c = idx % kBK;
            int t = i_ti + r, col = col0 + c;
            bool valid = (t < seq) && (col < K);
            s_dg2[idx] = s_q[idx] * s_dq2[idx];
            if (valid) dq2_base[(int64_t)t * HVK + col] = s_dq2[idx] + dq_base[(int64_t)t * HVK + col];
            s_dk2[idx] *= s_beta[r];
        }
    }
    __syncthreads();

    // (c) reverse (key side) contributions.
    int const NC_eff = min(NC, (seq - (int)t_chunk + kBC - 1) / kBC);
    if (i_i < NC_eff - 1) {
        int const r3 = min(i_ti + kBC, seq) - 1 - i_ti;  // row of this sub-chunk's last valid token
        for (int i_j = i_i + 1; i_j < NC_eff; ++i_j) {
            int const j0 = (int)t_chunk + i_j * kBC;
            int const rows_valid_j = min(kBC, seq - j0);
            stage_dA_T(s_A, dAqk_base + (int64_t)j0 * HVBT + i_i * kBC, HVBT, 0, rows_valid_j, tid);
            stage_dA_T(s_A2, dAkk_base + (int64_t)j0 * HVBT + i_i * kBC, HVBT, 0, rows_valid_j, tid);
            if (vec_ok) {
                constexpr int kCG = kBK / 4;
                CUTE_UNROLL
                for (int idx = tid; idx < kBC * kCG; idx += kThreads) {
                    int jj = idx / kCG, n = (idx % kCG) * 4;
                    int tj = j0 + jj;
                    float e[4] = {}, qv[4] = {}, kv[4] = {}, bv = 0.f;
                    if (tj < seq) {
                        float4 g4 = *reinterpret_cast<float4 const*>(g_base + (int64_t)tj * HVK + col0 + n);
                        uint2 rawq = *reinterpret_cast<uint2 const*>(q_base + (int64_t)tj * HK + col0 + n);
                        uint2 rawk = *reinterpret_cast<uint2 const*>(k_base + (int64_t)tj * HK + col0 + n);
                        T const* hq = reinterpret_cast<T const*>(&rawq);
                        T const* hk = reinterpret_cast<T const*>(&rawk);
                        float4 gr3 = *reinterpret_cast<float4 const*>(s_g + r3 * kBK + n);
                        e[0] = exp2f(g4.x - gr3.x); e[1] = exp2f(g4.y - gr3.y);
                        e[2] = exp2f(g4.z - gr3.z); e[3] = exp2f(g4.w - gr3.w);
                        qv[0] = to_f32(hq[0]); qv[1] = to_f32(hq[1]); qv[2] = to_f32(hq[2]); qv[3] = to_f32(hq[3]);
                        kv[0] = to_f32(hk[0]); kv[1] = to_f32(hk[1]); kv[2] = to_f32(hk[2]); kv[3] = to_f32(hk[3]);
                        bv = beta_base[(int64_t)tj * HV];
                    }
                    *reinterpret_cast<uint4*>(s_B + jj * kBK + n) = pack_tf32(
                        qv[0] * e[0], qv[1] * e[1], qv[2] * e[2], qv[3] * e[3]);
                    *reinterpret_cast<uint4*>(s_B2 + jj * kBK + n) = pack_tf32(
                        kv[0] * bv * e[0], kv[1] * bv * e[1], kv[2] * bv * e[2], kv[3] * bv * e[3]);
                }
            } else {
                for (int idx = tid; idx < kBC * kBK; idx += kThreads) {
                    int jj = idx / kBK, c = idx % kBK;
                    int tj = j0 + jj, col = col0 + c;
                    bool valid = (tj < seq) && (col < K);
                    float e = 0.0f, qv = 0.0f, kv = 0.0f, bv = 0.0f;
                    if (valid) {
                        e = exp2f(g_base[(int64_t)tj * HVK + col] - s_g[r3 * kBK + c]);
                        qv = to_f32(q_base[(int64_t)tj * HK + col]);
                        kv = to_f32(k_base[(int64_t)tj * HK + col]);
                        bv = beta_base[(int64_t)tj * HV];
                    }
                    s_B[idx] = to_tf32(qv * e);
                    s_B2[idx] = to_tf32(kv * bv * e);
                }
            }
            __syncthreads();
            load_frag(tCrA, tCcA, sA_t);
            load_frag(tCrA2, tCcA, sA2_t);
            load_frag(tCrB, tCcB, sB_t);
            load_frag(tCrB2, tCcB, sB2_t);
            cute::gemm(mma, tCrA, tCrB, fdkt);
            cute::gemm(mma, tCrA2, tCrB2, fdkt);
            __syncthreads();
        }
        for (int i = 0; i < size(fdkt); ++i) {
            int m = get<0>(tCcC(i)), n = get<1>(tCcC(i));
            fdkt(i) *= exp2f(s_g[r3 * kBK + n] - s_g[m * kBK + n]);
        }
    }

    // (c) diagonal sub-chunk, i >= j masked side
    if (SAFE_GATE) {
        {
            // masked transposed dA loads: keep = (r <= jj) && both tokens valid;
            // gmem coalesces along r
            CUTE_UNROLL
            for (int idx = tid; idx < kBC * (kBC / 4); idx += kThreads) {
                int jj = idx / (kBC / 4), r = (idx % (kBC / 4)) * 4;
                float4 v1 = make_float4(0.f, 0.f, 0.f, 0.f), v2 = make_float4(0.f, 0.f, 0.f, 0.f);
                if (i_ti + jj < seq) {
                    v1 = *reinterpret_cast<float4 const*>(dAqk_base + (int64_t)(i_ti + jj) * HVBT + i_i * kBC + r);
                    v2 = *reinterpret_cast<float4 const*>(dAkk_base + (int64_t)(i_ti + jj) * HVBT + i_i * kBC + r);
                }
                float va1[4] = {v1.x, v1.y, v1.z, v1.w}, va2[4] = {v2.x, v2.y, v2.z, v2.w};
                CUTE_UNROLL
                for (int j = 0; j < 4; ++j) {
                    bool keep = (r + j <= jj) && (i_ti + r + j < seq) && (i_ti + jj < seq);
                    va1[j] = keep ? va1[j] : 0.f;
                    va2[j] = keep ? va2[j] : 0.f;
                }
                s_A[r * kBC + jj] = to_tf32(va1[0]);
                s_A[(r + 1) * kBC + jj] = to_tf32(va1[1]);
                s_A[(r + 2) * kBC + jj] = to_tf32(va1[2]);
                s_A[(r + 3) * kBC + jj] = to_tf32(va1[3]);
                s_A2[r * kBC + jj] = to_tf32(va2[0]);
                s_A2[(r + 1) * kBC + jj] = to_tf32(va2[1]);
                s_A2[(r + 2) * kBC + jj] = to_tf32(va2[2]);
                s_A2[(r + 3) * kBC + jj] = to_tf32(va2[3]);
            }
            constexpr int kCG = kBK / 4;
            CUTE_UNROLL
            for (int idx = tid; idx < kBC * kCG; idx += kThreads) {
                int j = idx / kCG, c = (idx % kCG) * 4;
                float4 qv = *reinterpret_cast<float4 const*>(s_q + j * kBK + c);
                float4 kv = *reinterpret_cast<float4 const*>(s_k + j * kBK + c);
                float4 gj = *reinterpret_cast<float4 const*>(s_g + j * kBK + c);
                float4 gm = *reinterpret_cast<float4 const*>(s_g + mid * kBK + c);
                float e[4] = {0.f, 0.f, 0.f, 0.f};
                if (i_ti + j < seq) {
                    e[0] = exp2f(gj.x - gm.x);
                    e[1] = exp2f(gj.y - gm.y);
                    e[2] = exp2f(gj.z - gm.z);
                    e[3] = exp2f(gj.w - gm.w);
                }
                float b = s_beta[j];
                *reinterpret_cast<uint4*>(s_B + j * kBK + c) = pack_tf32(
                    qv.x * e[0], qv.y * e[1], qv.z * e[2], qv.w * e[3]);
                *reinterpret_cast<uint4*>(s_B2 + j * kBK + c) = pack_tf32(
                    kv.x * b * e[0], kv.y * b * e[1], kv.z * b * e[2], kv.w * b * e[3]);
            }
        }
        __syncthreads();
        load_frag(tCrA, tCcA, sA_t);
        load_frag(tCrA2, tCcA, sA2_t);
        load_frag(tCrB, tCcB, sB_t);
        load_frag(tCrB2, tCcB, sB2_t);
        clear(ftmp);
        clear(ftmp2);
        cute::gemm(mma, tCrA, tCrB, ftmp);
        cute::gemm(mma, tCrA2, tCrB2, ftmp2);
        for (int i = 0; i < size(fdkt); ++i) {
            int m = get<0>(tCcC(i)), n = get<1>(tCcC(i));
            float en = (i_ti + m < seq) ? exp2f(-(s_g[m * kBK + n] - s_g[mid * kBK + n])) : 0.0f;
            fdkt(i) += (ftmp(i) + ftmp2(i)) * en;
        }
        __syncthreads();
    } else {
        int const jmax = min(kBC, seq - i_ti);
        for (int j = 0; j < jmax; ++j) {
            if (tid < kBC) {
                int64_t off = (int64_t)(i_ti + j) * HVBT + i_i * kBC + tid;
                s_colA[tid] = dAqk_base[off];
                s_colB[tid] = dAkk_base[off];
            } else if (tid < kBC + kBK) {
                int c = tid - kBC;
                int col = col0 + c;
                s_rowq[c] = (col < K) ? to_f32(q_base[(int64_t)(i_ti + j) * HK + col]) : 0.0f;
                s_rowk[c] = (col < K) ? to_f32(k_base[(int64_t)(i_ti + j) * HK + col]) : 0.0f;
                s_rowg[c] = (col < K) ? g_base[(int64_t)(i_ti + j) * HVK + col] : 0.0f;
            }
            __syncthreads();
            float const bj = s_beta[j];
            for (int i = 0; i < size(fdkt); ++i) {
                int m = get<0>(tCcC(i)), n = get<1>(tCcC(i));
                if (m <= j) {
                    float e = exp2f(s_rowg[n] - s_g[m * kBK + n]);
                    fdkt(i) += s_colA[m] * s_rowq[n] * e + s_colB[m] * s_rowk[n] * bj * e;
                }
            }
            __syncthreads();
        }
    }

    // Epilogue: dk2 = beta*dk2 + dk_in + dkt; dg2 = q*dq2 + (beta*dk2 - dkt)*k + dg_in
    for (int i = 0; i < size(fdkt); ++i) {
        int m = get<0>(tCcC(i)), n = get<1>(tCcC(i));
        s_dkt[m * kBK + n] = fdkt(i);
    }
    __syncthreads();
    if (vec_ok) {
        constexpr int kCG = kBK / 4;
        CUTE_UNROLL
        for (int idx = tid; idx < kBC * kCG; idx += kThreads) {
            int r = idx / kCG, c = (idx % kCG) * 4;
            int t = i_ti + r;
            if (t >= seq) continue;
            int64_t off = (int64_t)t * HVK + col0 + c;
            float4 dg2v = *reinterpret_cast<float4 const*>(s_dg2 + r * kBK + c);
            float4 dk2v = *reinterpret_cast<float4 const*>(s_dk2 + r * kBK + c);
            float4 dktv = *reinterpret_cast<float4 const*>(s_dkt + r * kBK + c);
            float4 kv = *reinterpret_cast<float4 const*>(s_k + r * kBK + c);
            float4 dgv_in = *reinterpret_cast<float4 const*>(dg_base + off);
            float4 dkv_in = *reinterpret_cast<float4 const*>(dk_base + off);
            *reinterpret_cast<float4*>(dg2_base + off) = make_float4(
                dg2v.x + (dk2v.x - dktv.x) * kv.x + dgv_in.x,
                dg2v.y + (dk2v.y - dktv.y) * kv.y + dgv_in.y,
                dg2v.z + (dk2v.z - dktv.z) * kv.z + dgv_in.z,
                dg2v.w + (dk2v.w - dktv.w) * kv.w + dgv_in.w);
            *reinterpret_cast<float4*>(dk2_base + off) = make_float4(
                dk2v.x + dkv_in.x + dktv.x, dk2v.y + dkv_in.y + dktv.y,
                dk2v.z + dkv_in.z + dktv.z, dk2v.w + dkv_in.w + dktv.w);
        }
    } else {
        for (int idx = tid; idx < kBC * kBK; idx += kThreads) {
            int r = idx / kBK, c = idx % kBK;
            int t = i_ti + r, col = col0 + c;
            bool valid = (t < seq) && (col < K);
            if (!valid) continue;
            int64_t off = (int64_t)t * HVK + col;
            float dgv = s_dg2[idx] + (s_dk2[idx] - s_dkt[idx]) * s_k[idx] + dg_base[off];
            float dkv = s_dk2[idx] + dk_base[off] + s_dkt[idx];
            dg2_base[off] = dgv;
            dk2_base[off] = dkv;
        }
    }
}

template <typename T>
void launch_intra(
    T const* q, T const* k, float const* g, float const* beta,
    float const* dAqk, float const* dAkk,
    float const* dq, float* dq2, float const* dk, float* dk2,
    float const* dg, float* dg2, float* db2,
    int64_t const* cu_seqlens, int64_t const* chunk_indices,
    int64_t allT, int64_t NT, int64_t B, int64_t T_len,
    int H, int HV, int K, int NC, bool safe_gate,
    cudaStream_t stream
) {
    int const NK = (K + kBK - 1) / kBK;
    dim3 grid((unsigned)(NK * NC), (unsigned)NT, (unsigned)(B * HV));
    dim3 block(kThreads);

    #define LAUNCH_INTRA(IS_VARLEN, SAFE_GATE) \
        chunk_kda_bwd_intra_kernel<T, IS_VARLEN, SAFE_GATE><<<grid, block, 0, stream>>>( \
            q, k, g, beta, dAqk, dAkk, dq, dq2, dk, dk2, dg, dg2, db2, \
            cu_seqlens, chunk_indices, allT, (int)T_len, H, HV, K, NC)

    if (cu_seqlens) {
        if (safe_gate) { LAUNCH_INTRA(true, true); } else { LAUNCH_INTRA(true, false); }
    } else {
        if (safe_gate) { LAUNCH_INTRA(false, true); } else { LAUNCH_INTRA(false, false); }
    }
    #undef LAUNCH_INTRA
}

}  // namespace kda_impl

using kda_impl::launch_intra;
using kda_impl::kBT;
using kda_impl::kBC;
using kda_impl::kBK;

// fla/ops/kda/chunk_intra.py::chunk_kda_bwd_intra host wrapper.
// dq/dk/db/dg are the upstream (fused kernel) gradients; the kernel accumulates the
// intra-chunk parts and the host wrapper reduces db2 over the NK dim and adds db,
// exactly like the Triton host. Returns (dq2, dk2, db_out, dg2), all fp32.
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> chunk_kda_bwd_intra(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor g,
    torch::Tensor beta,
    torch::Tensor dAqk,
    torch::Tensor dAkk,
    torch::Tensor dq,
    torch::Tensor dk,
    torch::Tensor db,
    torch::Tensor dg,
    bool safe_gate,
    std::optional<torch::Tensor> cu_seqlens,
    std::optional<torch::Tensor> chunk_indices,
    int64_t chunk_size
) {
    TORCH_CHECK(chunk_size == kBT, "chunk_kda_bwd_intra only supports chunk_size 64");
    TORCH_CHECK(q.is_cuda() && q.is_contiguous() && q.dim() == 4, "q must be [B, T, H, K]");
    TORCH_CHECK(k.is_cuda() && k.is_contiguous() && k.sizes() == q.sizes());
    TORCH_CHECK(k.scalar_type() == q.scalar_type());

    int64_t B = k.size(0), T = k.size(1), H = k.size(2), K = k.size(3);
    int64_t HV = g.size(2);
    TORCH_CHECK(HV % H == 0, "HV must be a multiple of H");
    TORCH_CHECK(g.is_cuda() && g.is_contiguous() && g.dtype() == torch::kFloat32);
    TORCH_CHECK(g.dim() == 4 && g.size(0) == B && g.size(1) == T && g.size(3) == K);
    for (auto const& t : {beta, db}) {
        TORCH_CHECK(t.is_cuda() && t.is_contiguous() && t.dtype() == torch::kFloat32);
        TORCH_CHECK(t.dim() == 3 && t.size(0) == B && t.size(1) == T && t.size(2) == HV);
    }
    for (auto const& t : {dAqk, dAkk}) {
        TORCH_CHECK(t.is_cuda() && t.is_contiguous() && t.dtype() == torch::kFloat32);
        TORCH_CHECK(t.dim() == 4 && t.size(0) == B && t.size(1) == T && t.size(2) == HV && t.size(3) == kBT);
    }
    for (auto const& t : {dq, dk, dg}) {
        TORCH_CHECK(t.is_cuda() && t.is_contiguous() && t.dtype() == torch::kFloat32);
        TORCH_CHECK(t.sizes() == g.sizes());
    }

    int64_t const* cu_ptr = nullptr;
    int64_t const* ci_ptr = nullptr;
    int64_t NT = (T + kBT - 1) / kBT;
    if (cu_seqlens.has_value()) {
        TORCH_CHECK(B == 1, "B must be 1 when cu_seqlens is provided");
        TORCH_CHECK(chunk_indices.has_value(), "chunk_indices must be provided with cu_seqlens");
        auto const& cu = cu_seqlens.value();
        auto const& ci = chunk_indices.value();
        TORCH_CHECK(cu.is_cuda() && cu.is_contiguous() && cu.dtype() == torch::kLong);
        TORCH_CHECK(ci.is_cuda() && ci.is_contiguous() && ci.dtype() == torch::kLong);
        TORCH_CHECK(ci.dim() == 2 && ci.size(1) == 2);
        cu_ptr = cu.data_ptr<int64_t>();
        ci_ptr = ci.data_ptr<int64_t>();
        NT = ci.size(0);
    }

    int64_t const NK = (K + kBK - 1) / kBK;
    int64_t const NC = kBT / kBC;
    auto dq2 = torch::empty_like(dq);
    auto dk2 = torch::empty_like(dk);
    auto dg2 = torch::empty_like(dg);
    auto db2 = torch::empty({NK, B, T, HV}, dq.options().dtype(torch::kFloat32));

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    if (k.scalar_type() == at::kBFloat16) {
        launch_intra<cutlass::bfloat16_t>(
            reinterpret_cast<cutlass::bfloat16_t const*>(q.data_ptr()),
            reinterpret_cast<cutlass::bfloat16_t const*>(k.data_ptr()),
            g.data_ptr<float>(), beta.data_ptr<float>(), dAqk.data_ptr<float>(), dAkk.data_ptr<float>(),
            dq.data_ptr<float>(), dq2.data_ptr<float>(), dk.data_ptr<float>(), dk2.data_ptr<float>(),
            dg.data_ptr<float>(), dg2.data_ptr<float>(), db2.data_ptr<float>(),
            cu_ptr, ci_ptr, B * T, NT, B, T, (int)H, (int)HV, (int)K, (int)NC, safe_gate, stream);
    } else if (k.scalar_type() == at::kHalf) {
        launch_intra<cutlass::half_t>(
            reinterpret_cast<cutlass::half_t const*>(q.data_ptr()),
            reinterpret_cast<cutlass::half_t const*>(k.data_ptr()),
            g.data_ptr<float>(), beta.data_ptr<float>(), dAqk.data_ptr<float>(), dAkk.data_ptr<float>(),
            dq.data_ptr<float>(), dq2.data_ptr<float>(), dk.data_ptr<float>(), dk2.data_ptr<float>(),
            dg.data_ptr<float>(), dg2.data_ptr<float>(), db2.data_ptr<float>(),
            cu_ptr, ci_ptr, B * T, NT, B, T, (int)H, (int)HV, (int)K, (int)NC, safe_gate, stream);
    } else {
        TORCH_CHECK(false, "chunk_kda_bwd_intra: unsupported dtype ", k.scalar_type());
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    auto db_out = db2.sum(0).add_(db);
    return {dq2, dk2, db_out, dg2};
}
