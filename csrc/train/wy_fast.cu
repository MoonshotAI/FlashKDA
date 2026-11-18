// Recompute w/u/qg/kg for the KDA forward path.
// Replicates fla/ops/kda/wy_fast.py::recompute_w_u_fwd_kda_kernel (BT=64).
//
// Per (chunk, hv-head) block:
//   u  = Akk @ (beta * v)            [V tiles of 64]
//   w  = Akk @ (beta * k * exp2(g))  [K tiles of 64]
//   qg = q * exp2(g)                 (optional)
//   kg = k * exp2(g_last - g)        g_last = g of the chunk's last valid token
// Akk is the bf16 lower-triangular inverse with unit diagonal. All decay math
// is fp32 exp2 on the log2-domain cumsum g; products are rounded to bf16 before
// the GEMMs, which run on tensor cores (SM80 16x8x16, fp32 accumulation),
// matching the Triton tl.dot semantics.
//
// Data movement: the Akk tile is staged once with 16B cp.async (masked rows
// zero-filled via src-size 0), gated tiles (beta*v, beta*k*exp2(g)) are filled
// with 16B vector loads/stores, smem rows are padded +8 halves against bank
// conflicts, MMA fragments load with ldmatrix (x4 LDSM_N for the row-major A,
// SM75_U16x4_LDSM_T for the strided B view), and u/w go through smem staging
// for 16B coalesced stores. Per-element math, rounding points, and the
// ascending-K MMA order are unchanged from v1.

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>

#include <cute/tensor.hpp>

#include "common.cuh"

// NOTE: named namespace, not anonymous — nvcc's launch stub generation
// mis-resolves anonymous-namespace kernels when cute headers are included.
namespace wy_fast_impl {

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

// s[r][c] = g[r*row_stride + c] via 16B cp.async; rows past rows_valid
// zero-filled. Requires row_stride to be a multiple of 8 halves (16B).
template <typename T>
__device__ __forceinline__ void stage_tile(T* s, T const* g, int64_t row_stride,
                                           int rows_valid, int tid) {
    constexpr int kCG = kBT / 8;
    CUTE_UNROLL
    for (int idx = tid; idx < kBT * kCG; idx += kThreads) {
        int r = idx / kCG, c = (idx % kCG) * 8;
        bool full = r < rows_valid;
        cp_async16(s + r * kCP + c, full ? g + (int64_t)r * row_stride + c : g, full);
    }
}

// Grid: (NT, B*HV). Block: 128 threads.
template <typename T, bool STORE_QG, bool IS_VARLEN>
__global__ void __launch_bounds__(kThreads) recompute_w_u_fwd_kernel(
    T const* __restrict__ q,
    T const* __restrict__ k,
    T* __restrict__ qg,
    T* __restrict__ kg,
    T const* __restrict__ v,
    float const* __restrict__ beta,
    T* __restrict__ w,
    T* __restrict__ u,
    T const* __restrict__ A,
    float const* __restrict__ gk,
    int64_t const* __restrict__ cu_seqlens,
    int64_t const* __restrict__ chunk_indices,
    int64_t T_len,
    int H,
    int HV,
    int K,
    int V
) {
    int64_t i_t = blockIdx.x;
    int64_t i_bh = blockIdx.y;
    int64_t i_b = i_bh / HV, i_hv = i_bh % HV;
    int i_h = int(i_hv / (HV / H));

    int64_t bos, t0, seq_len;
    if (IS_VARLEN) {
        int64_t i_n = chunk_indices[i_t * 2];
        int64_t i_tl = chunk_indices[i_t * 2 + 1];
        bos = cu_seqlens[i_n];
        seq_len = cu_seqlens[i_n + 1] - bos;
        t0 = i_tl * kBT;
    } else {
        bos = i_b * T_len;
        seq_len = T_len;
        t0 = i_t * kBT;
    }
    int64_t rem = seq_len - t0;
    int rows_valid = int(rem < (int64_t)kBT ? rem : (int64_t)kBT);
    if (rows_valid <= 0) return;

    const int64_t tok = bos + t0;
    T const* kp = k + (tok * H + i_h) * (int64_t)K;
    T const* qp = q + (tok * H + i_h) * (int64_t)K;
    T const* vp = v + (tok * HV + i_hv) * (int64_t)V;
    T* up = u + (tok * HV + i_hv) * (int64_t)V;
    T* wp = w + (tok * HV + i_hv) * (int64_t)K;
    T* qgp = qg + (tok * HV + i_hv) * (int64_t)K;
    T* kgp = kg + (tok * HV + i_hv) * (int64_t)K;
    float const* gkp = gk + (tok * HV + i_hv) * (int64_t)K;
    float const* betap = beta + tok * HV + i_hv;
    T const* Ap = A + (tok * HV + i_hv) * (int64_t)kBT;
    int64_t last_idx = (rem < (int64_t)kBT ? seq_len : t0 + kBT) - 1;
    float const* gnp = gk + ((bos + last_idx) * HV + i_hv) * (int64_t)K;

    __shared__ T sA[kBT * kCP];
    __shared__ T sB[kBT * kCP];
    __shared__ float sBeta[kBT];

    int const tid = threadIdx.x;
    bool const vec_ok = (K % 8) == 0 && (V % 8) == 0;

    if (tid < kBT) {
        sBeta[tid] = tid < rows_valid ? betap[tid * HV] : 0.0f;
    }
    if (vec_ok) {
        stage_tile(sA, Ap, (int64_t)HV * kBT, rows_valid, tid);
        cp_async_commit();
        cp_async_wait<0>();
    } else {
        for (int idx = tid; idx < kBT * kBT; idx += kThreads) {
            int r = idx >> 6;
            sA[r * kCP + (idx & 63)] = r < rows_valid ? Ap[r * (int64_t)HV * kBT + (idx & 63)] : T(0);
        }
    }
    __syncthreads();

    using Atom = typename MmaAtom<T>::type;
    auto mma = make_tiled_mma(Atom{}, Layout<Shape<_4, _1, _1>>{}, Tile<_64, _64, _16>{});
    auto thr_mma = mma.get_thread_slice(tid);

    Copy_Atom<SM75_U32x4_LDSM_N, T> ldsm_n;
    Copy_Atom<SM75_U16x4_LDSM_T, T> ldsm_t;
    auto s2r_a = make_tiled_copy_A(ldsm_n, mma);   // row-major (M,K) tile
    auto s2r_bt = make_tiled_copy_B(ldsm_t, mma);  // strided (N,K) view
    auto thr_s2r_a = s2r_a.get_thread_slice(tid);
    auto thr_s2r_bt = s2r_bt.get_thread_slice(tid);

    Tensor sA_rm = make_tensor(make_smem_ptr(sA), Layout<Shape<_64, _64>, Stride<Int<kCP>, _1>>{});
    Tensor sB_st = make_tensor(make_smem_ptr(sB), Layout<Shape<_64, _64>, Stride<_1, Int<kCP>>>{});

    Tensor cC = make_identity_tensor(Shape<_64, _64>{});
    Tensor tCcC = thr_mma.partition_C(cC);

    // rC[64,64] (fp32) = sA[64(M),64(K)] @ sB_st[64(N),64(K)]^T, K sliced
    // 16-wide ascending; then staged back through sB for 16B gmem stores.
    auto gemm_tile = [&](T* out, int64_t row_stride, int rows_v, int cols_v) {
        Tensor rC = thr_mma.make_fragment_C(tCcC);
        clear(rC);
        Tensor tCrA = thr_mma.partition_fragment_A(sA_rm);
        Tensor tCrB = thr_mma.partition_fragment_B(sB_st);
        Tensor tXsA = thr_s2r_a.partition_S(sA_rm);
        Tensor tXrA = thr_s2r_a.retile_D(tCrA);
        Tensor tXsB = thr_s2r_bt.partition_S(sB_st);
        Tensor tXrB = thr_s2r_bt.retile_D(tCrB);
        constexpr int KB = decltype(size<2>(tXsA))::value;
        CUTE_UNROLL
        for (int kb = 0; kb < KB; ++kb) {
            copy(s2r_a, tXsA(_, _, kb), tXrA(_, _, kb));
            copy(s2r_bt, tXsB(_, _, kb), tXrB(_, _, kb));
            gemm(mma, tCrA(_, _, kb), tCrB(_, _, kb), rC);
        }
        __syncthreads();  // sB reads are done; reuse it as the store staging
        CUTE_UNROLL
        for (int i = 0; i < size(rC); ++i) {
            sB[get<0>(tCcC(i)) * kCP + get<1>(tCcC(i))] = T(rC(i));
        }
        __syncthreads();
        if (vec_ok) {
            constexpr int kCG = kBT / 8;
            CUTE_UNROLL
            for (int idx = tid; idx < kBT * kCG; idx += kThreads) {
                int r = idx / kCG, c = (idx % kCG) * 8;
                if (r < rows_v && c < cols_v) {
                    *reinterpret_cast<uint4*>(out + r * row_stride + c) =
                        *reinterpret_cast<uint4 const*>(sB + r * kCP + c);
                }
            }
        } else {
            for (int idx = tid; idx < kBT * kBT; idx += kThreads) {
                int r = idx >> 6, c = idx & 63;
                if (r < rows_v && c < cols_v) {
                    out[r * row_stride + c] = sB[r * kCP + c];
                }
            }
        }
        __syncthreads();  // sB is reused by the next tile's fill
    };

    // u = Akk @ (beta * v), V tiles of 64
    // With STORE_QG the two output groups split across blockIdx.z (z=0: V
    // loop, z=1: K loop); without qg a single block does both loops (the split
    // only adds an extra Akk pass there).
    if (!STORE_QG || blockIdx.z == 0) {
    for (int v0 = 0; v0 < V; v0 += kBT) {
        int const cols_valid = V - v0 < kBT ? V - v0 : kBT;
        if (vec_ok) {
            constexpr int kCG = kBT / 8;
            CUTE_UNROLL
            for (int idx = tid; idx < kBT * kCG; idx += kThreads) {
                int r = idx / kCG, c = (idx % kCG) * 8;
                T vals[8] = {};
                if (r < rows_valid && c < cols_valid) {
                    uint4 raw = *reinterpret_cast<uint4 const*>(vp + r * (int64_t)HV * V + v0 + c);
                    T const* hv = reinterpret_cast<T const*>(&raw);
                    float const b = sBeta[r];
                    CUTE_UNROLL
                    for (int j = 0; j < 8; ++j) vals[j] = T(to_f32(hv[j]) * b);  // round before the dot
                }
                *reinterpret_cast<uint4*>(sB + r * kCP + c) = *reinterpret_cast<uint4 const*>(vals);
            }
        } else {
            for (int idx = tid; idx < kBT * kBT; idx += kThreads) {
                int r = idx >> 6, c = idx & 63;
                float val = 0.0f;
                if (r < rows_valid && c < cols_valid) {
                    val = to_f32(vp[r * (int64_t)HV * V + v0 + c]) * sBeta[r];
                }
                sB[r * kCP + c] = T(val);
            }
        }
        __syncthreads();
        gemm_tile(up + v0, (int64_t)HV * V, rows_valid, cols_valid);
    }
        if (STORE_QG) return;
    }

    // w = Akk @ (beta * k * exp2(g)), plus qg/kg elementwise, K tiles of 64
    for (int k0 = 0; k0 < K; k0 += kBT) {
        int const cols_valid = K - k0 < kBT ? K - k0 : kBT;
        if (vec_ok) {
            constexpr int kCG = kBT / 8;
            CUTE_UNROLL
            for (int idx = tid; idx < kBT * kCG; idx += kThreads) {
                int r = idx / kCG, c = (idx % kCG) * 8;
                T vals[8] = {};
                if (r < rows_valid && c < cols_valid) {
                    uint4 rk = *reinterpret_cast<uint4 const*>(kp + r * (int64_t)H * K + k0 + c);
                    T const* hk = reinterpret_cast<T const*>(&rk);
                    float4 const* pg = reinterpret_cast<float4 const*>(gkp + r * (int64_t)HV * K + k0 + c);
                    float4 g0 = pg[0], g1 = pg[1];
                    float4 const* pn = reinterpret_cast<float4 const*>(gnp + k0 + c);
                    float4 n0 = pn[0], n1 = pn[1];
                    float const b = sBeta[r];
                    float gv[8] = {g0.x, g0.y, g0.z, g0.w, g1.x, g1.y, g1.z, g1.w};
                    float nv[8] = {n0.x, n0.y, n0.z, n0.w, n1.x, n1.y, n1.z, n1.w};
                    T kgs[8];
                    CUTE_UNROLL
                    for (int j = 0; j < 8; ++j) {
                        float const b_k = to_f32(hk[j]);
                        float const eg = exp2f(gv[j]);
                        vals[j] = T(b_k * b * eg);
                        kgs[j] = T(b_k * exp2f(nv[j] - gv[j]));
                    }
                    *reinterpret_cast<uint4*>(kgp + r * (int64_t)HV * K + k0 + c) =
                        *reinterpret_cast<uint4 const*>(kgs);
                    if (STORE_QG) {
                        uint4 rq = *reinterpret_cast<uint4 const*>(qp + r * (int64_t)H * K + k0 + c);
                        T const* hq = reinterpret_cast<T const*>(&rq);
                        T qgs[8];
                        CUTE_UNROLL
                        for (int j = 0; j < 8; ++j) qgs[j] = T(to_f32(hq[j]) * exp2f(gv[j]));
                        *reinterpret_cast<uint4*>(qgp + r * (int64_t)HV * K + k0 + c) =
                            *reinterpret_cast<uint4 const*>(qgs);
                    }
                }
                *reinterpret_cast<uint4*>(sB + r * kCP + c) = *reinterpret_cast<uint4 const*>(vals);
            }
        } else {
            for (int idx = tid; idx < kBT * kBT; idx += kThreads) {
                int r = idx >> 6, c = idx & 63;
                bool valid = r < rows_valid && c < cols_valid;
                float b_k = 0.0f, b_g = 0.0f;
                if (valid) {
                    b_k = to_f32(kp[r * (int64_t)H * K + k0 + c]);
                    b_g = gkp[r * (int64_t)HV * K + k0 + c];
                }
                float eg = exp2f(b_g);
                sB[r * kCP + c] = T(b_k * sBeta[r] * eg);
                if (valid) {
                    if (STORE_QG) {
                        qgp[r * (int64_t)HV * K + k0 + c] = T(to_f32(qp[r * (int64_t)H * K + k0 + c]) * eg);
                    }
                    kgp[r * (int64_t)HV * K + k0 + c] = T(b_k * exp2f(gnp[k0 + c] - b_g));
                }
            }
        }
        __syncthreads();
        gemm_tile(wp + k0, (int64_t)HV * K, rows_valid, cols_valid);
    }
}

template <typename T>
void launch_recompute_w_u_fwd(
    T const* q,
    T const* k,
    T* qg,
    T* kg,
    T const* v,
    float const* beta,
    T* w,
    T* u,
    T const* A,
    float const* gk,
    int64_t const* cu_seqlens,
    int64_t const* chunk_indices,
    int64_t NT,
    int64_t B,
    int64_t T_len,
    int64_t H,
    int64_t HV,
    int64_t K,
    int64_t V,
    cudaStream_t stream
) {
    dim3 grid((unsigned)NT, (unsigned)(B * HV), qg ? 2u : 1u);
    dim3 block(kThreads);

    #define LAUNCH_RECOMPUTE(SQ, IS_VARLEN) \
        recompute_w_u_fwd_kernel<T, SQ, IS_VARLEN><<<grid, block, 0, stream>>>( \
            q, k, qg, kg, v, beta, w, u, A, gk, cu_seqlens, chunk_indices, \
            T_len, (int)H, (int)HV, (int)K, (int)V)

    #define DISPATCH_RECOMPUTE_VARLEN(SQ) \
        if (cu_seqlens) { LAUNCH_RECOMPUTE(SQ, true); } \
        else { LAUNCH_RECOMPUTE(SQ, false); }

    if (qg) { DISPATCH_RECOMPUTE_VARLEN(true); }
    else { DISPATCH_RECOMPUTE_VARLEN(false); }

    #undef DISPATCH_RECOMPUTE_VARLEN
    #undef LAUNCH_RECOMPUTE
}

}  // namespace wy_fast_impl

using wy_fast_impl::launch_recompute_w_u_fwd;

std::vector<torch::Tensor> recompute_w_u_fwd(
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor beta,
    torch::Tensor A,
    torch::Tensor gk,
    std::optional<torch::Tensor> q,
    std::optional<torch::Tensor> cu_seqlens,
    std::optional<torch::Tensor> chunk_indices
) {
    TORCH_CHECK(k.is_cuda() && k.is_contiguous(), "k must be contiguous CUDA tensor");
    TORCH_CHECK(v.is_cuda() && v.is_contiguous(), "v must be contiguous CUDA tensor");
    TORCH_CHECK(A.is_cuda() && A.is_contiguous(), "A must be contiguous CUDA tensor");
    TORCH_CHECK(gk.is_cuda() && gk.is_contiguous() && gk.scalar_type() == torch::kFloat32,
                "gk must be fp32 contiguous CUDA tensor");
    TORCH_CHECK(beta.is_cuda() && beta.is_contiguous(), "beta must be contiguous CUDA tensor");
    torch::Tensor beta_f = beta.scalar_type() == torch::kFloat32 ? beta : beta.to(torch::kFloat32);
    TORCH_CHECK(k.dim() == 4, "k must be [B, T, H, K]");
    TORCH_CHECK(v.dim() == 4, "v must be [B, T, HV, V]");

    int64_t B = k.size(0), T_len = k.size(1), H = k.size(2), K = k.size(3);
    int64_t HV = v.size(2), V = v.size(3);
    TORCH_CHECK(HV % H == 0, "HV must be a multiple of H");
    TORCH_CHECK(A.dim() == 4 && A.size(0) == B && A.size(1) == T_len && A.size(2) == HV &&
                A.size(3) == 64, "A must be [B, T, HV, 64]");

    int64_t const* cu_ptr = nullptr;
    int64_t const* ci_ptr = nullptr;
    int64_t NT = (T_len + 63) / 64;
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

    auto opts = v.options();
    auto w = torch::empty({B, T_len, HV, K}, opts);
    auto u = torch::empty({B, T_len, HV, V}, opts);
    auto kg = torch::empty({B, T_len, HV, K}, opts);
    torch::Tensor qg;
    void* qg_ptr = nullptr;
    void const* q_ptr = nullptr;
    if (q.has_value()) {
        TORCH_CHECK(q->is_cuda() && q->is_contiguous() && q->sizes() == k.sizes(),
                    "q must be contiguous [B, T, H, K]");
        qg = torch::empty({B, T_len, HV, K}, opts);
        qg_ptr = qg.data_ptr();
        q_ptr = q->data_ptr();
    }

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    if (k.scalar_type() == at::kBFloat16) {
        launch_recompute_w_u_fwd<cutlass::bfloat16_t>(
            reinterpret_cast<cutlass::bfloat16_t const*>(q_ptr),
            reinterpret_cast<cutlass::bfloat16_t const*>(k.data_ptr()),
            reinterpret_cast<cutlass::bfloat16_t*>(qg_ptr),
            reinterpret_cast<cutlass::bfloat16_t*>(kg.data_ptr()),
            reinterpret_cast<cutlass::bfloat16_t const*>(v.data_ptr()),
            beta_f.data_ptr<float>(),
            reinterpret_cast<cutlass::bfloat16_t*>(w.data_ptr()),
            reinterpret_cast<cutlass::bfloat16_t*>(u.data_ptr()),
            reinterpret_cast<cutlass::bfloat16_t const*>(A.data_ptr()),
            gk.data_ptr<float>(), cu_ptr, ci_ptr, NT, B, T_len, H, HV, K, V, stream);
    } else if (k.scalar_type() == at::kHalf) {
        launch_recompute_w_u_fwd<cutlass::half_t>(
            reinterpret_cast<cutlass::half_t const*>(q_ptr),
            reinterpret_cast<cutlass::half_t const*>(k.data_ptr()),
            reinterpret_cast<cutlass::half_t*>(qg_ptr),
            reinterpret_cast<cutlass::half_t*>(kg.data_ptr()),
            reinterpret_cast<cutlass::half_t const*>(v.data_ptr()),
            beta_f.data_ptr<float>(),
            reinterpret_cast<cutlass::half_t*>(w.data_ptr()),
            reinterpret_cast<cutlass::half_t*>(u.data_ptr()),
            reinterpret_cast<cutlass::half_t const*>(A.data_ptr()),
            gk.data_ptr<float>(), cu_ptr, ci_ptr, NT, B, T_len, H, HV, K, V, stream);
    } else {
        TORCH_CHECK(false, "recompute_w_u_fwd supports bf16/fp16 only");
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    // keep the 4-slot shape: qg is None when q is not given
    return {w, u, qg, kg};
}
