// KDA backward: dAqk (attention gradient matrix) and the intra-chunk part of dv.
// Replicates fla/ops/kda/chunk_bwd.py::chunk_kda_bwd_kernel_dAv.
//
// Per (chunk, head) block:
//   dAqk[i,j] = scale * sum_v do[i,v] * v_new[j,v]   for i >= j (fp32)
//   dv[i,:]   = sum_j tril(Aqk)[j,i] * do[j,:]       (stored in do dtype)
// All MMAs run on tensor cores (SM80 16x8x16 bf16/fp16 atoms, fp32 accumulators).
//
// Data movement: gmem tiles are staged with 16B cp.async (masked rows zero-filled
// via src-size 0), smem rows padded +8 halves against bank conflicts, MMA operand
// fragments load with ldmatrix (x4 for row-major tiles, .trans for strided
// views), and gmem stores go through smem staging for 16B coalesced writes.
// GEMM/accumulation order and rounding points are unchanged from v1.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>

#include <cute/tensor.hpp>

#include "common.cuh"

namespace kda_impl {

using namespace cute;

constexpr int kBT = 64;  // chunk size (KDA bwd is only ever run with 64)
constexpr int kBV = 64;  // V tile, matches fla's CONST_TILING on non-Hopper
constexpr int kThreads = 128;
constexpr int kPad = 8;            // smem row padding in halves (16B)
constexpr int kCP = kBT + kPad;    // padded row stride of every 64-wide tile

template <typename T> struct MmaAtom;
template <> struct MmaAtom<cutlass::bfloat16_t> { using type = SM80_16x8x16_F32BF16BF16F32_TN; };
template <> struct MmaAtom<cutlass::half_t> { using type = SM80_16x8x16_F32F16F16F32_TN; };

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
// chunks at/past cols_valid zero-filled. Requires 16B-aligned rows (row_stride
// and cols_valid multiples of 8 halves).
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

// scalar fallback for irregular V (not a multiple of 8)
template <typename T>
__device__ __forceinline__ void stage_tile_scalar(T* s, T const* g, int64_t row_stride,
                                                  int rows_valid, int cols_valid, int tid) {
    for (int idx = tid; idx < kBT * kBT; idx += kThreads) {
        int r = idx / kBT, c = idx % kBT;
        s[r * kCP + c] = (r < rows_valid && c < cols_valid) ? g[(int64_t)r * row_stride + c] : T(0.0f);
    }
}

// Grid: (NT, B*HV).
template <typename T, bool IS_VARLEN>
__global__ void __launch_bounds__(kThreads) chunk_kda_bwd_dav_kernel(
    T const* __restrict__ A,
    T const* __restrict__ v,
    T const* __restrict__ do_,
    T* __restrict__ dv,
    float* __restrict__ dA,
    float scale,
    int64_t const* __restrict__ cu_seqlens,
    int64_t const* __restrict__ chunk_indices,
    int T_len, int HV, int V
) {
    int64_t const i_t = blockIdx.x;
    int64_t const i_bh = blockIdx.y;
    int64_t const i_hv = i_bh % HV;

    int64_t bos, t0;
    int64_t seq_len = T_len;
    if (IS_VARLEN) {
        int64_t const i_n = chunk_indices[i_t * 2];
        int64_t const i_tl = chunk_indices[i_t * 2 + 1];
        bos = cu_seqlens[i_n];
        seq_len = cu_seqlens[i_n + 1] - bos;
        t0 = i_tl * kBT;
    } else {
        bos = (i_bh / HV) * (int64_t)T_len;
        t0 = i_t * kBT;
    }
    int64_t const rem = seq_len - t0;
    int const rows_valid = int(rem < (int64_t)kBT ? rem : (int64_t)kBT);
    if (rows_valid <= 0) return;

    T const* A_base = A + (bos * HV + i_hv) * kBT;
    T const* v_base = v + (bos * HV + i_hv) * V;
    T const* do_base = do_ + (bos * HV + i_hv) * V;
    T* dv_base = dv + (bos * HV + i_hv) * V;
    float* dA_base = dA + (bos * HV + i_hv) * kBT;

    // dv fragments are staged through sV (dead by then); the fp32 dA tile
    // overlays all three tiles after the V loop.
    __shared__ union {
        struct {
            T sDo[kBT * kCP];  // do tile: [r][v]
            T sV[kBV * kCP];   // v_new tile: [j][v]
            T sA[kBT * kCP];   // Aqk tile: [r][a], tril-masked in smem
        } t;
        float sDA[kBT * kBT];
    } sm;

    int const tid = threadIdx.x;
    bool const vec_ok = (V % 8) == 0;

    // A tile: natural [r][a] staging, then apply the tril/seq column mask in smem
    // (v1 staged the transposed view with strided scalar gmem reads instead).
    if (vec_ok) {
        stage_tile(sm.t.sA, A_base + t0 * (int64_t)HV * kBT, (int64_t)HV * kBT, rows_valid, kBT, tid);
        cp_async_commit();
        cp_async_wait<0>();
    } else {
        stage_tile_scalar(sm.t.sA, A_base + t0 * (int64_t)HV * kBT, (int64_t)HV * kBT, rows_valid, kBT, tid);
    }
    __syncthreads();
    for (int idx = tid; idx < kBT * kBT; idx += kThreads) {
        int r = idx / kBT, a = idx % kBT;
        if (a > r || t0 + a >= seq_len) sm.t.sA[r * kCP + a] = T(0.0f);
    }

    using Atom = typename MmaAtom<T>::type;
    auto mma = make_tiled_mma(Atom{}, Layout<Shape<_4, _1, _1>>{}, Tile<_64, _64, _16>{});
    auto thr_mma = mma.get_thread_slice(tid);

    Copy_Atom<SM75_U32x4_LDSM_N, T> ldsm_n;
    Copy_Atom<SM75_U16x4_LDSM_T, T> ldsm_t;
    auto s2r_a = make_tiled_copy_A(ldsm_n, mma);   // row-major (M,K) tiles
    auto s2r_at = make_tiled_copy_A(ldsm_t, mma);  // strided (M,K) views
    auto s2r_b = make_tiled_copy_B(ldsm_n, mma);   // row-major (N,K) tiles
    auto s2r_bt = make_tiled_copy_B(ldsm_t, mma);  // strided (N,K) views
    auto thr_s2r_a = s2r_a.get_thread_slice(tid);
    auto thr_s2r_at = s2r_at.get_thread_slice(tid);
    auto thr_s2r_b = s2r_b.get_thread_slice(tid);
    auto thr_s2r_bt = s2r_bt.get_thread_slice(tid);

    Tensor sDo_rm = make_tensor(make_smem_ptr(sm.t.sDo), Layout<Shape<_64, _64>, Stride<Int<kCP>, _1>>{});
    Tensor sV_rm = make_tensor(make_smem_ptr(sm.t.sV), Layout<Shape<_64, _64>, Stride<Int<kCP>, _1>>{});
    Tensor sA_rm = make_tensor(make_smem_ptr(sm.t.sA), Layout<Shape<_64, _64>, Stride<Int<kCP>, _1>>{});
    // strided views over the same tiles
    Tensor sA_st = make_tensor(make_smem_ptr(sm.t.sA), Layout<Shape<_64, _64>, Stride<_1, Int<kCP>>>{});   // (M=a, K=r)
    Tensor sDo_st = make_tensor(make_smem_ptr(sm.t.sDo), Layout<Shape<_64, _64>, Stride<_1, Int<kCP>>>{});  // (N=v, K=r)

    Tensor cC = make_identity_tensor(Shape<_64, _64>{});
    Tensor tCcC = thr_mma.partition_C(cC);

    Tensor fdA = thr_mma.make_fragment_C(tCcC);
    clear(fdA);

    // acc += sA_view[64(M),64(K)] @ sB_view[64(N),64(K)]^T, K sliced 16-wide;
    // copies/mmas run k-block by k-block in ascending order (v1 numerics).
    auto gemm_rm_rm = [&](auto& acc, auto const& sA_t, auto const& sB_t) {
        Tensor tCrA = thr_mma.partition_fragment_A(sA_t);
        Tensor tCrB = thr_mma.partition_fragment_B(sB_t);
        Tensor tXsA = thr_s2r_a.partition_S(sA_t);
        Tensor tXsB = thr_s2r_b.partition_S(sB_t);
        Tensor tXrA = thr_s2r_a.retile_D(tCrA);
        Tensor tXrB = thr_s2r_b.retile_D(tCrB);
        constexpr int KB = decltype(size<2>(tXsA))::value;
        CUTE_UNROLL
        for (int kb = 0; kb < KB; ++kb) {
            copy(s2r_a, tXsA(_, _, kb), tXrA(_, _, kb));
            copy(s2r_b, tXsB(_, _, kb), tXrB(_, _, kb));
            gemm(mma, tCrA(_, _, kb), tCrB(_, _, kb), acc);
        }
    };
    auto gemm_st_st = [&](auto& acc, auto const& sA_t, auto const& sB_t) {
        Tensor tCrA = thr_mma.partition_fragment_A(sA_t);
        Tensor tCrB = thr_mma.partition_fragment_B(sB_t);
        Tensor tXsA = thr_s2r_at.partition_S(sA_t);
        Tensor tXsB = thr_s2r_bt.partition_S(sB_t);
        Tensor tXrA = thr_s2r_at.retile_D(tCrA);
        Tensor tXrB = thr_s2r_bt.retile_D(tCrB);
        constexpr int KB = decltype(size<2>(tXsA))::value;
        CUTE_UNROLL
        for (int kb = 0; kb < KB; ++kb) {
            copy(s2r_at, tXsA(_, _, kb), tXrA(_, _, kb));
            copy(s2r_bt, tXsB(_, _, kb), tXrB(_, _, kb));
            gemm(mma, tCrA(_, _, kb), tCrB(_, _, kb), acc);
        }
    };

    for (int v0 = 0; v0 < V; v0 += kBV) {
        int const cols_valid = min(kBV, V - v0);
        if (vec_ok) {
            stage_tile(sm.t.sDo, do_base + t0 * (int64_t)HV * V + v0, (int64_t)HV * V, rows_valid, cols_valid, tid);
            stage_tile(sm.t.sV, v_base + t0 * (int64_t)HV * V + v0, (int64_t)HV * V, rows_valid, cols_valid, tid);
            cp_async_commit();
            cp_async_wait<0>();
        } else {
            stage_tile_scalar(sm.t.sDo, do_base + t0 * (int64_t)HV * V + v0, (int64_t)HV * V, rows_valid, cols_valid, tid);
            stage_tile_scalar(sm.t.sV, v_base + t0 * (int64_t)HV * V + v0, (int64_t)HV * V, rows_valid, cols_valid, tid);
        }
        __syncthreads();

        // fdA[t, j] += sum_v do[t,v] * v_new[j,v]
        gemm_rm_rm(fdA, sDo_rm, sV_rm);

        // fdv[a, v] = sum_r tril(Aqk)^T[a,r] * do[r,v]
        Tensor fdv = thr_mma.make_fragment_C(tCcC);
        clear(fdv);
        gemm_st_st(fdv, sA_st, sDo_st);
        __syncthreads();  // sV is dead now; reuse it as the dv staging tile

        for (int i = 0; i < size(fdv); ++i) {
            int m = get<0>(tCcC(i)), n = get<1>(tCcC(i));
            sm.t.sV[m * kCP + n] = T(fdv(i));
        }
        __syncthreads();
        if (vec_ok) {
            constexpr int kCG = kBV / 8;
            CUTE_UNROLL
            for (int idx = tid; idx < kBT * kCG; idx += kThreads) {
                int r = idx / kCG, c = (idx % kCG) * 8;
                if (r < rows_valid && c < cols_valid) {
                    *reinterpret_cast<uint4*>(dv_base + (t0 + r) * (int64_t)HV * V + v0 + c) =
                        *reinterpret_cast<uint4 const*>(sm.t.sV + r * kCP + c);
                }
            }
        } else {
            for (int idx = tid; idx < kBT * kBV; idx += kThreads) {
                int r = idx / kBV, vv = idx % kBV;
                if (r < rows_valid && vv < cols_valid) {
                    dv_base[(t0 + r) * (int64_t)HV * V + v0 + vv] = sm.t.sV[r * kCP + vv];
                }
            }
        }
        __syncthreads();
    }

    // dA[t, j] = (t >= j) ? fdA * scale : 0, fp32, staged for 16B stores
    for (int i = 0; i < size(fdA); ++i) {
        int m = get<0>(tCcC(i)), n = get<1>(tCcC(i));
        sm.sDA[m * kBT + n] = (m >= n) ? fdA(i) * scale : 0.0f;
    }
    __syncthreads();
    {
        constexpr int kCG = kBT / 4;
        CUTE_UNROLL
        for (int idx = tid; idx < kBT * kCG; idx += kThreads) {
            int r = idx / kCG, c = (idx % kCG) * 4;
            if (r < rows_valid) {
                *reinterpret_cast<float4*>(dA_base + (t0 + r) * (int64_t)HV * kBT + c) =
                    *reinterpret_cast<float4 const*>(sm.sDA + r * kBT + c);
            }
        }
    }
}

template <typename T>
void launch_dav(
    T const* A, T const* v, T const* do_, T* dv, float* dA,
    float scale,
    int64_t const* cu_seqlens, int64_t const* chunk_indices,
    int64_t NT, int64_t B, int64_t T_len, int64_t HV, int64_t V,
    cudaStream_t stream
) {
    dim3 grid((unsigned)NT, (unsigned)(B * HV));
    dim3 block(kThreads);
    if (cu_seqlens) {
        chunk_kda_bwd_dav_kernel<T, true><<<grid, block, 0, stream>>>(
            A, v, do_, dv, dA, scale, cu_seqlens, chunk_indices, (int)T_len, (int)HV, (int)V);
    } else {
        chunk_kda_bwd_dav_kernel<T, false><<<grid, block, 0, stream>>>(
            A, v, do_, dv, dA, scale, nullptr, nullptr, (int)T_len, (int)HV, (int)V);
    }
}

}  // namespace kda_impl

using kda_impl::launch_dav;
using kda_impl::kBT;

std::tuple<torch::Tensor, torch::Tensor> chunk_kda_bwd_dAv(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor do_,
    torch::Tensor A,
    double scale,
    std::optional<torch::Tensor> cu_seqlens,
    std::optional<torch::Tensor> chunk_indices,
    int64_t chunk_size
) {
    TORCH_CHECK(chunk_size == kBT, "chunk_kda_bwd_dAv only supports chunk_size 64");
    TORCH_CHECK(do_.is_cuda() && do_.is_contiguous(), "do must be contiguous CUDA tensor");
    TORCH_CHECK(v.is_cuda() && v.is_contiguous() && v.sizes() == do_.sizes());
    TORCH_CHECK(A.is_cuda() && A.is_contiguous() && A.scalar_type() == do_.scalar_type());
    TORCH_CHECK(do_.dim() == 4, "do must be [B, T, HV, V]");

    int64_t B = do_.size(0), T = do_.size(1), HV = do_.size(2), V = do_.size(3);
    TORCH_CHECK(A.dim() == 4 && A.size(0) == B && A.size(1) == T && A.size(2) == HV && A.size(3) == kBT);

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

    auto dA = torch::empty({B, T, HV, kBT}, do_.options().dtype(torch::kFloat32));
    auto dv = torch::empty_like(do_);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    if (do_.scalar_type() == at::kBFloat16) {
        launch_dav<cutlass::bfloat16_t>(
            reinterpret_cast<cutlass::bfloat16_t const*>(A.data_ptr()),
            reinterpret_cast<cutlass::bfloat16_t const*>(v.data_ptr()),
            reinterpret_cast<cutlass::bfloat16_t const*>(do_.data_ptr()),
            reinterpret_cast<cutlass::bfloat16_t*>(dv.data_ptr()),
            dA.data_ptr<float>(), (float)scale, cu_ptr, ci_ptr, NT, B, T, HV, V, stream);
    } else if (do_.scalar_type() == at::kHalf) {
        launch_dav<cutlass::half_t>(
            reinterpret_cast<cutlass::half_t const*>(A.data_ptr()),
            reinterpret_cast<cutlass::half_t const*>(v.data_ptr()),
            reinterpret_cast<cutlass::half_t const*>(do_.data_ptr()),
            reinterpret_cast<cutlass::half_t*>(dv.data_ptr()),
            dA.data_ptr<float>(), (float)scale, cu_ptr, ci_ptr, NT, B, T, HV, V, stream);
    } else {
        TORCH_CHECK(false, "chunk_kda_bwd_dAv: unsupported dtype ", do_.scalar_type());
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {dA, dv};
}
