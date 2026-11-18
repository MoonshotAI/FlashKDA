// Output kernel of the chunked GLA forward, reused by KDA.
// Replicates fla/ops/gla/chunk.py::chunk_gla_fwd_kernel_o (BT=64, BK=BV=64).
//
// Per (v-tile, chunk, hv-head) block:
//   o = scale * (q * exp2(g)) @ h  +  tril(Aqk, 0) @ v_new
// q*exp2(g) is computed in fp32 and rounded to the input dtype before the dot,
// as the Triton kernel does. h is bf16, [B, NT, HV, K, V] or [B, NT, HV, V, K]
// when state_v_first. GEMMs run on tensor cores (SM80 16x8x16, fp32 accum).
//
// Data movement: pure-copy tiles (h, v) are staged with 16B cp.async (masked
// rows zero-filled via src-size 0), computed tiles (q*exp2(g), tril(A)) use
// 16B vector loads/stores, smem rows are padded +8 halves against bank
// conflicts, MMA fragments load with ldmatrix, and the output goes through smem
// staging for 16B coalesced stores. GEMM/accumulation order and rounding
// points are unchanged from v1.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>

#include <cute/tensor.hpp>

#include "common.cuh"

// NOTE: named namespace, not anonymous — nvcc's launch stub generation
// mis-resolves anonymous-namespace kernels when cute headers are included.
namespace chunk_o_impl {

using namespace cute;

template <typename T> struct MmaAtom;
template <> struct MmaAtom<cutlass::bfloat16_t> { using type = SM80_16x8x16_F32BF16BF16F32_TN; };
template <> struct MmaAtom<cutlass::half_t> { using type = SM80_16x8x16_F32F16F16F32_TN; };

constexpr int kBT = 64;
constexpr int kThreads = 128;
constexpr int kPad = 8;          // smem row padding in halves (16B)
constexpr int kCP = kBT + kPad;  // padded row stride of every 64-wide tile

__device__ __forceinline__ void cp_async16(void* dst, void const* src, bool full) {
    uint32_t s = cute::cast_smem_ptr_to_uint(dst);
    int sz = full ? 16 : 0;  // src-size 0 zero-fills, used for masked rows/cols
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

// s[r][c] = g[r*row_stride + c] via 16B cp.async; rows past rows_valid and col
// chunks at/past cols_valid zero-filled. Requires row_stride and cols_valid to
// be multiples of 8 halves (16B).
template <typename T>
__device__ __forceinline__ void stage_tile(T* s, T const* g, int64_t row_stride,
                                           int rows_valid, int cols_valid, int tid) {
    constexpr int kCG = kBT / 8;
    CUTE_UNROLL
    for (int idx = tid; idx < kBT * kCG; idx += kThreads) {
        int r = idx / kCG, c = (idx % kCG) * 8;
        bool full = (r < rows_valid) && (c < cols_valid);
        cp_async16(s + r * kCP + c, full ? g + (int64_t)r * row_stride + c : g, full);
    }
}

template <typename T>
__device__ __forceinline__ void stage_tile_scalar(T* s, T const* g, int64_t row_stride,
                                                  int rows_valid, int cols_valid, int tid) {
    for (int idx = tid; idx < kBT * kBT; idx += kThreads) {
        int r = idx / kBT, c = idx % kBT;
        s[r * kCP + c] = (r < rows_valid && c < cols_valid) ? g[(int64_t)r * row_stride + c] : T(0.0f);
    }
}

// Grid: (cdiv(V, 64), NT, B*HV). Block: 128 threads.
template <typename T, bool STATE_V_FIRST, bool IS_VARLEN>
__global__ void __launch_bounds__(kThreads) chunk_gla_fwd_o_gk_kernel(
    T const* __restrict__ q,
    T const* __restrict__ v,
    float const* __restrict__ g,
    T const* __restrict__ h,
    T const* __restrict__ A,
    T* __restrict__ o,
    float scale,
    int64_t const* __restrict__ cu_seqlens,
    int64_t const* __restrict__ chunk_indices,
    int64_t T_len,
    int H,
    int HV,
    int K,
    int V
) {
    int i_v = blockIdx.x;
    int64_t i_t = blockIdx.y;
    int64_t i_bh = blockIdx.z;
    int64_t i_b = i_bh / HV, i_hv = i_bh % HV;
    int i_h = int(i_hv / (HV / H));

    int64_t bos, t0, seq_len, i_tg;
    if (IS_VARLEN) {
        i_tg = i_t;  // the grid chunk index is already the global h chunk index
        int64_t i_n = chunk_indices[i_t * 2];
        int64_t i_tl = chunk_indices[i_t * 2 + 1];
        bos = cu_seqlens[i_n];
        seq_len = cu_seqlens[i_n + 1] - bos;
        t0 = i_tl * kBT;
    } else {
        int64_t NT = (T_len + kBT - 1) / kBT;
        i_tg = i_b * NT + i_t;
        bos = i_b * T_len;
        seq_len = T_len;
        t0 = i_t * kBT;
    }
    int64_t rem = seq_len - t0;
    int rows_valid = int(rem < (int64_t)kBT ? rem : (int64_t)kBT);
    int v0 = i_v * kBT;
    int cols_valid = V - v0 < kBT ? V - v0 : kBT;
    if (rows_valid <= 0 || cols_valid <= 0) return;

    const int64_t tok = bos + t0;
    T const* qp = q + (tok * H + i_h) * (int64_t)K;
    float const* gp = g + (tok * HV + i_hv) * (int64_t)K;
    T const* vp = v + (tok * HV + i_hv) * (int64_t)V;
    T* op = o + (tok * HV + i_hv) * (int64_t)V;
    T const* hp = h + (i_tg * HV + i_hv) * (int64_t)K * V;
    T const* Ap = A + (tok * HV + i_hv) * (int64_t)kBT;

    __shared__ T sA[kBT * kCP];
    __shared__ T sB[kBT * kCP];

    int const tid = threadIdx.x;
    bool const vec_ok = (K % 8) == 0 && (V % 8) == 0;

    using Atom = typename MmaAtom<T>::type;
    auto mma = make_tiled_mma(Atom{}, Layout<Shape<_4, _1, _1>>{}, Tile<_64, _64, _16>{});
    auto thr_mma = mma.get_thread_slice(tid);

    Copy_Atom<SM75_U32x4_LDSM_N, T> ldsm_n;
    Copy_Atom<SM75_U16x4_LDSM_T, T> ldsm_t;
    auto s2r_a = make_tiled_copy_A(ldsm_n, mma);   // row-major (M,K)
    auto s2r_b = make_tiled_copy_B(ldsm_n, mma);   // row-major (N,K)
    auto s2r_bt = make_tiled_copy_B(ldsm_t, mma);  // strided (N,K) view
    auto thr_s2r_a = s2r_a.get_thread_slice(tid);
    auto thr_s2r_b = s2r_b.get_thread_slice(tid);
    auto thr_s2r_bt = s2r_bt.get_thread_slice(tid);

    Tensor sA_rm = make_tensor(make_smem_ptr(sA), Layout<Shape<_64, _64>, Stride<Int<kCP>, _1>>{});
    Tensor sB_rm = make_tensor(make_smem_ptr(sB), Layout<Shape<_64, _64>, Stride<Int<kCP>, _1>>{});
    Tensor sB_st = make_tensor(make_smem_ptr(sB), Layout<Shape<_64, _64>, Stride<_1, Int<kCP>>>{});

    Tensor cC = make_identity_tensor(Shape<_64, _64>{});
    Tensor tCcC = thr_mma.partition_C(cC);
    Tensor rC = thr_mma.make_fragment_C(tCcC);
    clear(rC);

    // rC += sA[64(M),64(K)] @ sB[64(N),64(K)]^T, K sliced 16-wide, ascending.
    auto gemm_acc = [&](auto const& sA_t, auto const& sB_t, auto b_strided) {
        constexpr bool kBStrided = decltype(b_strided)::value;
        Tensor tCrA = thr_mma.partition_fragment_A(sA_t);
        Tensor tCrB = thr_mma.partition_fragment_B(sB_t);
        Tensor tXsA = thr_s2r_a.partition_S(sA_t);
        Tensor tXrA = thr_s2r_a.retile_D(tCrA);
        constexpr int KB = decltype(size<2>(tXsA))::value;
        if constexpr (kBStrided) {
            Tensor tXsB = thr_s2r_bt.partition_S(sB_t);
            Tensor tXrB = thr_s2r_bt.retile_D(tCrB);
            CUTE_UNROLL
            for (int kb = 0; kb < KB; ++kb) {
                copy(s2r_a, tXsA(_, _, kb), tXrA(_, _, kb));
                copy(s2r_bt, tXsB(_, _, kb), tXrB(_, _, kb));
                gemm(mma, tCrA(_, _, kb), tCrB(_, _, kb), rC);
            }
        } else {
            Tensor tXsB = thr_s2r_b.partition_S(sB_t);
            Tensor tXrB = thr_s2r_b.retile_D(tCrB);
            CUTE_UNROLL
            for (int kb = 0; kb < KB; ++kb) {
                copy(s2r_a, tXsA(_, _, kb), tXrA(_, _, kb));
                copy(s2r_b, tXsB(_, _, kb), tXrB(_, _, kb));
                gemm(mma, tCrA(_, _, kb), tCrB(_, _, kb), rC);
            }
        }
    };

    // inter part: sum over K tiles of (q * exp2(g)) @ h
    for (int k0 = 0; k0 < K; k0 += kBT) {
        int kcols = K - k0 < kBT ? K - k0 : kBT;
        if (vec_ok) {
            // sA[r][c] = T(q[r, k0+c] * exp2f(g[r, k0+c])), 8 cols per thread
            constexpr int kCG = kBT / 8;
            CUTE_UNROLL
            for (int idx = tid; idx < kBT * kCG; idx += kThreads) {
                int r = idx / kCG, c = (idx % kCG) * 8;
                T vals[8] = {};
                if (r < rows_valid && c < kcols) {
                    uint4 rq = *reinterpret_cast<uint4 const*>(qp + r * (int64_t)H * K + k0 + c);
                    T const* hq = reinterpret_cast<T const*>(&rq);
                    float4 const* pg = reinterpret_cast<float4 const*>(gp + r * (int64_t)HV * K + k0 + c);
                    float4 g0 = pg[0], g1 = pg[1];
                    vals[0] = T(to_f32(hq[0]) * exp2f(g0.x));
                    vals[1] = T(to_f32(hq[1]) * exp2f(g0.y));
                    vals[2] = T(to_f32(hq[2]) * exp2f(g0.z));
                    vals[3] = T(to_f32(hq[3]) * exp2f(g0.w));
                    vals[4] = T(to_f32(hq[4]) * exp2f(g1.x));
                    vals[5] = T(to_f32(hq[5]) * exp2f(g1.y));
                    vals[6] = T(to_f32(hq[6]) * exp2f(g1.z));
                    vals[7] = T(to_f32(hq[7]) * exp2f(g1.w));
                }
                *reinterpret_cast<uint4*>(sA + r * kCP + c) = *reinterpret_cast<uint4 const*>(vals);
            }
            // sB: h tile, pure copy
            if (!STATE_V_FIRST) {
                // h is [K, V]: rows k0+r, cols v0+c; row-dim validity is kcols
                stage_tile(sB, hp + (int64_t)k0 * V + v0, V, kcols, cols_valid, tid);
            } else {
                // h is [V, K]: rows v0+r, cols k0+c
                stage_tile(sB, hp + (int64_t)v0 * K + k0, K, cols_valid, kcols, tid);
            }
            cp_async_commit();
            cp_async_wait<0>();
        } else {
            for (int idx = tid; idx < kBT * kBT; idx += kThreads) {
                int r = idx >> 6, c = idx & 63;
                float val = 0.0f;
                if (r < rows_valid && c < kcols) {
                    val = to_f32(qp[r * (int64_t)H * K + k0 + c]) * exp2f(gp[r * (int64_t)HV * K + k0 + c]);
                }
                sA[r * kCP + c] = T(val);
            }
            if (!STATE_V_FIRST) {
                stage_tile_scalar(sB, hp + (int64_t)k0 * V + v0, V, kcols, cols_valid, tid);
            } else {
                stage_tile_scalar(sB, hp + (int64_t)v0 * K + k0, K, cols_valid, kcols, tid);
            }
        }
        __syncthreads();
        if constexpr (STATE_V_FIRST) { gemm_acc(sA_rm, sB_rm, cute::false_type{}); }
        else { gemm_acc(sA_rm, sB_st, cute::true_type{}); }
        __syncthreads();
    }

    for (int i = 0; i < size(rC); ++i) rC(i) *= scale;

    // intra part: tril(Aqk, 0) @ v_new (Aqk already carries the scale)
    if (vec_ok) {
        constexpr int kCG = kBT / 8;
        CUTE_UNROLL
        for (int idx = tid; idx < kBT * kCG; idx += kThreads) {
            int r = idx / kCG, c = (idx % kCG) * 8;
            uint4 raw = make_uint4(0, 0, 0, 0);
            if (r < rows_valid && r >= c) {
                // tril mask within the 8-col group: keep cols c+j <= r
                uint4 full = *reinterpret_cast<uint4 const*>(Ap + r * (int64_t)HV * kBT + c);
                T const* fa = reinterpret_cast<T const*>(&full);
                T vals[8];
                CUTE_UNROLL
                for (int j = 0; j < 8; ++j) vals[j] = (c + j <= r) ? fa[j] : T(0.0f);
                raw = *reinterpret_cast<uint4 const*>(vals);
            }
            *reinterpret_cast<uint4*>(sA + r * kCP + c) = raw;
        }
        stage_tile(sB, vp + v0, (int64_t)HV * V, rows_valid, cols_valid, tid);
        cp_async_commit();
        cp_async_wait<0>();
    } else {
        for (int idx = tid; idx < kBT * kBT; idx += kThreads) {
            int r = idx >> 6, c = idx & 63;
            sA[r * kCP + c] = (r < rows_valid && r >= c) ? Ap[r * (int64_t)HV * kBT + c] : T(0.0f);
        }
        stage_tile_scalar(sB, vp + v0, (int64_t)HV * V, rows_valid, cols_valid, tid);
    }
    __syncthreads();
    gemm_acc(sA_rm, sB_st, cute::true_type{});
    __syncthreads();  // sA is dead; reuse it as the o staging tile

    for (int i = 0; i < size(rC); ++i) {
        sA[get<0>(tCcC(i)) * kCP + get<1>(tCcC(i))] = T(rC(i));
    }
    __syncthreads();
    if (vec_ok) {
        constexpr int kCG = kBT / 8;
        CUTE_UNROLL
        for (int idx = tid; idx < kBT * kCG; idx += kThreads) {
            int r = idx / kCG, c = (idx % kCG) * 8;
            if (r < rows_valid && c < cols_valid) {
                *reinterpret_cast<uint4*>(op + r * (int64_t)HV * V + v0 + c) =
                    *reinterpret_cast<uint4 const*>(sA + r * kCP + c);
            }
        }
    } else {
        for (int idx = tid; idx < kBT * kBT; idx += kThreads) {
            int r = idx >> 6, c = idx & 63;
            if (r < rows_valid && c < cols_valid) {
                op[r * (int64_t)HV * V + v0 + c] = sA[r * kCP + c];
            }
        }
    }
}

template <typename T>
void launch_chunk_gla_fwd_o_gk(
    torch::Tensor const& q,
    torch::Tensor const& v,
    torch::Tensor const& g,
    torch::Tensor const& A,
    torch::Tensor const& h,
    torch::Tensor& o,
    float scale,
    bool state_v_first,
    int64_t const* cu_seqlens,
    int64_t const* chunk_indices,
    int64_t NT,
    int64_t B,
    int64_t T_len,
    int H,
    int HV,
    int K,
    int V,
    cudaStream_t stream
) {
    dim3 grid((V + kBT - 1) / kBT, NT, B * HV);
    dim3 block(kThreads);

    #define LAUNCH_O(SVF, IS_VARLEN) \
        chunk_gla_fwd_o_gk_kernel<T, SVF, IS_VARLEN><<<grid, block, 0, stream>>>( \
            reinterpret_cast<T const*>(q.data_ptr()), \
            reinterpret_cast<T const*>(v.data_ptr()), \
            g.data_ptr<float>(), \
            reinterpret_cast<T const*>(h.data_ptr()), \
            reinterpret_cast<T const*>(A.data_ptr()), \
            reinterpret_cast<T*>(o.data_ptr()), \
            scale, cu_seqlens, chunk_indices, T_len, H, HV, K, V)

    if (state_v_first) {
        if (cu_seqlens) { LAUNCH_O(true, true); } else { LAUNCH_O(true, false); }
    } else {
        if (cu_seqlens) { LAUNCH_O(false, true); } else { LAUNCH_O(false, false); }
    }
    #undef LAUNCH_O
}

}  // namespace chunk_o_impl

using chunk_o_impl::kBT;
using chunk_o_impl::launch_chunk_gla_fwd_o_gk;

torch::Tensor chunk_gla_fwd_o_gk(
    torch::Tensor q,
    torch::Tensor v,
    torch::Tensor g,
    torch::Tensor A,
    torch::Tensor h,
    double scale,
    bool state_v_first,
    std::optional<torch::Tensor> cu_seqlens,
    int64_t chunk_size,
    std::optional<torch::Tensor> chunk_indices
) {
    TORCH_CHECK(q.is_cuda() && q.is_contiguous() && q.dim() == 4, "q must be [B, T, H, K] contiguous CUDA");
    TORCH_CHECK(v.is_cuda() && v.is_contiguous() && v.dim() == 4, "v must be [B, T, HV, V] contiguous CUDA");
    TORCH_CHECK(v.scalar_type() == q.scalar_type(), "v and q must share dtype");
    TORCH_CHECK(g.is_cuda() && g.is_contiguous() && g.scalar_type() == torch::kFloat32,
                "g must be fp32 [B, T, HV, K] contiguous CUDA");
    TORCH_CHECK(A.is_cuda() && A.is_contiguous() && A.dim() == 4, "A must be [B, T, HV, BT] contiguous CUDA");
    TORCH_CHECK(A.scalar_type() == q.scalar_type(), "A and q must share dtype");
    TORCH_CHECK(h.is_cuda() && h.is_contiguous() && h.dim() == 5, "h must be 5D contiguous CUDA");
    TORCH_CHECK(h.scalar_type() == q.scalar_type(), "h and q must share dtype");
    TORCH_CHECK(chunk_size == kBT, "only chunk_size 64 is supported");

    int64_t B = q.size(0), T_len = q.size(1);
    int H = int(q.size(2)), K = int(q.size(3));
    int HV = int(v.size(2)), V = int(v.size(3));
    TORCH_CHECK(HV % H == 0, "HV must be a multiple of H");
    if (state_v_first) {
        TORCH_CHECK(h.size(3) == V && h.size(4) == K, "h must be [B, NT, HV, V, K]");
    } else {
        TORCH_CHECK(h.size(3) == K && h.size(4) == V, "h must be [B, NT, HV, K, V]");
    }

    int64_t const* cu_ptr = nullptr;
    int64_t const* ci_ptr = nullptr;
    int64_t NT;
    if (cu_seqlens.has_value()) {
        TORCH_CHECK(B == 1, "B must be 1 when cu_seqlens is provided");
        TORCH_CHECK(chunk_indices.has_value(), "chunk_indices required with cu_seqlens");
        auto const& cu = cu_seqlens.value();
        auto const& ci = chunk_indices.value();
        TORCH_CHECK(cu.dtype() == torch::kLong && cu.is_cuda() && cu.is_contiguous());
        TORCH_CHECK(ci.is_cuda() && ci.is_contiguous() && ci.dtype() == torch::kLong);
        TORCH_CHECK(ci.dim() == 2 && ci.size(1) == 2);
        cu_ptr = cu.data_ptr<int64_t>();
        ci_ptr = ci.data_ptr<int64_t>();
        NT = ci.size(0);
    } else {
        NT = (T_len + kBT - 1) / kBT;
    }

    auto o = torch::zeros_like(v);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    if (q.scalar_type() == at::kBFloat16) {
        launch_chunk_gla_fwd_o_gk<cutlass::bfloat16_t>(
            q, v, g, A, h, o, float(scale), state_v_first,
            cu_ptr, ci_ptr, NT, B, T_len, H, HV, K, V, stream);
    } else if (q.scalar_type() == at::kHalf) {
        launch_chunk_gla_fwd_o_gk<cutlass::half_t>(
            q, v, g, A, h, o, float(scale), state_v_first,
            cu_ptr, ci_ptr, NT, B, T_len, H, HV, K, V, stream);
    } else {
        TORCH_CHECK(false, "chunk_gla_fwd_o_gk supports bf16/fp16 only");
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return o;
}
