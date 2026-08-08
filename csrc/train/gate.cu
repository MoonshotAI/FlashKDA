// KDA gate activation + chunk-local cumsum, and its backward.
// Replicates fla/ops/kda/gate.py and fla/ops/utils/cumsum.py (vector kernels).

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>

#include "common.cuh"
#include "train_ops.h"

namespace {

constexpr int kBS = 64;  // threads per block, one channel column each

// Grid: (cdiv(S, BS), NT, B*H). Each thread owns one channel s of one
// (sequence, chunk, head) and walks the chunk's BT rows serially.
template <typename T, bool HAS_BIAS, bool USE_LOWER_BOUND, bool HAS_SCALE, bool IS_VARLEN>
__global__ void kda_gate_chunk_cumsum_kernel(
    T const* __restrict__ g,
    float const* __restrict__ A_log,
    float const* __restrict__ dt_bias,
    float* __restrict__ o,
    float scale,
    float lower_bound,
    int64_t const* __restrict__ cu_seqlens,
    int64_t const* __restrict__ chunk_indices,
    int T_len,
    int H,
    int S,
    int BT
) {
    int i_s = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t i_t = blockIdx.y;
    int64_t i_bh = blockIdx.z;
    int i_h = int(i_bh % H);
    if (i_s >= S) return;

    int64_t bos, i_t0;
    int64_t seq_len = T_len;
    if (IS_VARLEN) {
        int64_t i_n = chunk_indices[i_t * 2];
        int64_t i_tl = chunk_indices[i_t * 2 + 1];
        bos = cu_seqlens[i_n];
        seq_len = cu_seqlens[i_n + 1] - bos;
        i_t0 = i_tl * BT;
    } else {
        bos = (i_bh / H) * (int64_t)T_len;
        i_t0 = i_t * BT;
    }

    float b_A = expf(A_log[i_h]);
    float bias = 0.0f;
    if (HAS_BIAS) bias = dt_bias[i_h * S + i_s];

    int64_t base = (bos * H + i_h) * (int64_t)S + i_s;
    int64_t stride = (int64_t)H * S;

    float run = 0.0f;
    for (int t = 0; t < BT; ++t) {
        int64_t tt = i_t0 + t;
        if (tt >= seq_len) break;
        float x = to_f32(g[base + tt * stride]);
        if (HAS_BIAS) x += bias;
        float gate;
        if (USE_LOWER_BOUND) {
            gate = lower_bound * sigmoid_f32(b_A * x);
        } else {
            gate = -b_A * softplus_f32(x);
        }
        run += gate;
        o[base + tt * stride] = HAS_SCALE ? run * scale : run;
    }
}

// Grid: (cdiv(S, BS), NT, B*H). No gate activation; optional reverse (suffix) cumsum.
template <typename T, bool HAS_SCALE, bool REVERSE, bool IS_VARLEN>
__global__ void chunk_local_cumsum_kernel(
    T const* __restrict__ g,
    float* __restrict__ o,
    float scale,
    int64_t const* __restrict__ cu_seqlens,
    int64_t const* __restrict__ chunk_indices,
    int T_len,
    int H,
    int S,
    int BT
) {
    int i_s = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t i_t = blockIdx.y;
    int64_t i_bh = blockIdx.z;
    int i_h = int(i_bh % H);
    if (i_s >= S) return;

    int64_t bos, i_t0;
    int64_t seq_len = T_len;
    if (IS_VARLEN) {
        int64_t i_n = chunk_indices[i_t * 2];
        int64_t i_tl = chunk_indices[i_t * 2 + 1];
        bos = cu_seqlens[i_n];
        seq_len = cu_seqlens[i_n + 1] - bos;
        i_t0 = i_tl * BT;
    } else {
        bos = (i_bh / H) * (int64_t)T_len;
        i_t0 = i_t * BT;
    }

    int64_t base = (bos * H + i_h) * (int64_t)S + i_s;
    int64_t stride = (int64_t)H * S;

    int64_t rem = seq_len - i_t0;
    int t_valid = int(rem < (int64_t)BT ? rem : (int64_t)BT);
    float run = 0.0f;
    if (REVERSE) {
        for (int t = t_valid - 1; t >= 0; --t) {
            int64_t off = base + (i_t0 + t) * stride;
            run += to_f32(g[off]);
            o[off] = HAS_SCALE ? run * scale : run;
        }
    } else {
        for (int t = 0; t < t_valid; ++t) {
            int64_t off = base + (i_t0 + t) * stride;
            run += to_f32(g[off]);
            o[off] = HAS_SCALE ? run * scale : run;
        }
    }
}

constexpr int kGateBwdBT = 32;  // matches fla's fixed BT=32 in kda_gate_bwd

// Grid: (cdiv(B*T, 32), H). Block: 128 threads over the D dimension.
// Writes dg and the per-block partial dA; the caller sums dA_partial over dim 0.
template <typename T, bool HAS_BIAS, bool USE_LOWER_BOUND>
__global__ void kda_gate_bwd_kernel(
    T const* __restrict__ g,
    float const* __restrict__ A_log,
    float const* __restrict__ dt_bias,
    float const* __restrict__ dyg,
    float* __restrict__ dg,
    float* __restrict__ dA_partial,
    float lower_bound,
    int64_t T_total,
    int H,
    int D
) {
    int64_t i_t = blockIdx.x;
    int i_h = blockIdx.y;
    int64_t t0 = i_t * kGateBwdBT;

    float b_A = expf(A_log[i_h]);
    float partial = 0.0f;

    int64_t rem = T_total - t0;
    int t_valid = int(rem < (int64_t)kGateBwdBT ? rem : (int64_t)kGateBwdBT);
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        float bias = 0.0f;
        if (HAS_BIAS) bias = dt_bias[i_h * D + d];
        for (int t = 0; t < t_valid; ++t) {
            int64_t off = (t0 + t) * (int64_t)H * D + i_h * D + d;
            float x = to_f32(g[off]) + bias;
            float dy = dyg[off];
            float dgv;
            if (USE_LOWER_BOUND) {
                float sig = sigmoid_f32(b_A * x);
                dgv = dy * lower_bound * sig * (1.0f - sig) * b_A;
                partial += dgv * x;
            } else {
                float yg = -b_A * softplus_f32(x);
                dgv = -b_A * dy * sigmoid_f32(x);
                partial += dy * yg;
            }
            dg[off] = dgv;
        }
    }

    // block reduce partial
    __shared__ float smem[32];
    for (int offset = 16; offset > 0; offset >>= 1)
        partial += __shfl_down_sync(0xffffffff, partial, offset);
    if ((threadIdx.x & 31) == 0) smem[threadIdx.x >> 5] = partial;
    __syncthreads();
    if (threadIdx.x < 32) {
        int n_warps = (int(blockDim.x) + 31) / 32;
        float v = threadIdx.x < n_warps ? smem[threadIdx.x] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1)
            v += __shfl_down_sync(0xffffffff, v, offset);
        if (threadIdx.x == 0) dA_partial[i_t * H + i_h] = v;
    }
}

struct VarlenArgs {
    int64_t const* cu_seqlens = nullptr;
    int64_t const* chunk_indices = nullptr;
    int64_t NT = 0;
};

VarlenArgs resolve_varlen(
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

void check_gate_inputs(
    torch::Tensor const& g,
    torch::Tensor const& A_log,
    std::optional<torch::Tensor> const& dt_bias,
    torch::Tensor const& out
) {
    TORCH_CHECK(g.is_cuda() && g.is_contiguous(), "g must be contiguous CUDA tensor");
    TORCH_CHECK(g.dim() == 4, "g must be [B, T, H, S]");
    TORCH_CHECK(A_log.is_cuda() && A_log.is_contiguous() && A_log.dtype() == torch::kFloat32);
    TORCH_CHECK(A_log.dim() == 1 && A_log.size(0) == g.size(2), "A_log must be [H]");
    if (dt_bias.has_value()) {
        TORCH_CHECK(dt_bias->is_cuda() && dt_bias->is_contiguous() && dt_bias->dtype() == torch::kFloat32);
        TORCH_CHECK(dt_bias->numel() == g.size(2) * g.size(3), "dt_bias must have H*S elements");
    }
    TORCH_CHECK(out.is_cuda() && out.is_contiguous() && out.dtype() == torch::kFloat32);
    TORCH_CHECK(out.sizes() == g.sizes(), "out must match g shape");
}

template <typename T>
void launch_gate_cumsum(
    T const* g_ptr,
    float const* A_log_ptr,
    float const* bias_ptr,
    float* o_ptr,
    float scale,
    float lower_bound,
    bool has_bias,
    bool use_lower_bound,
    bool has_scale,
    VarlenArgs const& varlen,
    int T_len,
    int H,
    int S,
    int chunk_size,
    dim3 grid,
    cudaStream_t stream
) {
    dim3 block(kBS);

    #define LAUNCH_GATE(HAS_BIAS, USE_LB, HAS_SCALE, IS_VARLEN) \
        kda_gate_chunk_cumsum_kernel<T, HAS_BIAS, USE_LB, HAS_SCALE, IS_VARLEN><<<grid, block, 0, stream>>>( \
            g_ptr, A_log_ptr, bias_ptr, o_ptr, scale, lower_bound, \
            varlen.cu_seqlens, varlen.chunk_indices, T_len, H, S, chunk_size)

    #define DISPATCH_VARLEN(HAS_BIAS, USE_LB, HAS_SCALE) \
        if (varlen.cu_seqlens) { LAUNCH_GATE(HAS_BIAS, USE_LB, HAS_SCALE, true); } \
        else { LAUNCH_GATE(HAS_BIAS, USE_LB, HAS_SCALE, false); }
    #define DISPATCH_SCALE(HAS_BIAS, USE_LB) \
        if (has_scale) { DISPATCH_VARLEN(HAS_BIAS, USE_LB, true); } \
        else { DISPATCH_VARLEN(HAS_BIAS, USE_LB, false); }
    #define DISPATCH_LB(HAS_BIAS) \
        if (use_lower_bound) { DISPATCH_SCALE(HAS_BIAS, true); } \
        else { DISPATCH_SCALE(HAS_BIAS, false); }

    if (has_bias) { DISPATCH_LB(true); }
    else { DISPATCH_LB(false); }

    #undef DISPATCH_LB
    #undef DISPATCH_SCALE
    #undef DISPATCH_VARLEN
    #undef LAUNCH_GATE
}

template <typename T>
void launch_local_cumsum(
    T const* g_ptr,
    float* o_ptr,
    float scale,
    bool has_scale,
    bool reverse,
    VarlenArgs const& varlen,
    int T_len,
    int H,
    int S,
    int chunk_size,
    dim3 grid,
    cudaStream_t stream
) {
    dim3 block(kBS);

    #define LAUNCH_CUMSUM(HAS_SCALE, REVERSE, IS_VARLEN) \
        chunk_local_cumsum_kernel<T, HAS_SCALE, REVERSE, IS_VARLEN><<<grid, block, 0, stream>>>( \
            g_ptr, o_ptr, scale, varlen.cu_seqlens, varlen.chunk_indices, T_len, H, S, chunk_size)

    #define DISPATCH_VARLEN_C(HAS_SCALE, REVERSE) \
        if (varlen.cu_seqlens) { LAUNCH_CUMSUM(HAS_SCALE, REVERSE, true); } \
        else { LAUNCH_CUMSUM(HAS_SCALE, REVERSE, false); }
    #define DISPATCH_REVERSE(HAS_SCALE) \
        if (reverse) { DISPATCH_VARLEN_C(HAS_SCALE, true); } \
        else { DISPATCH_VARLEN_C(HAS_SCALE, false); }

    if (has_scale) { DISPATCH_REVERSE(true); }
    else { DISPATCH_REVERSE(false); }

    #undef DISPATCH_REVERSE
    #undef DISPATCH_VARLEN_C
    #undef LAUNCH_CUMSUM
}

template <typename T>
void launch_gate_bwd(
    T const* g_ptr,
    float const* A_log_ptr,
    float const* bias_ptr,
    float const* dyg_ptr,
    float* dg_ptr,
    float* dA_ptr,
    float lower_bound,
    bool use_lower_bound,
    int64_t T_total,
    int H,
    int D,
    int64_t NT,
    cudaStream_t stream
) {
    dim3 grid(NT, H);
    dim3 block(128);

    #define LAUNCH_GATE_BWD(HAS_BIAS, USE_LB) \
        kda_gate_bwd_kernel<T, HAS_BIAS, USE_LB><<<grid, block, 0, stream>>>( \
            g_ptr, A_log_ptr, bias_ptr, dyg_ptr, dg_ptr, dA_ptr, lower_bound, T_total, H, D)

    if (bias_ptr) {
        if (use_lower_bound) { LAUNCH_GATE_BWD(true, true); } else { LAUNCH_GATE_BWD(true, false); }
    } else {
        if (use_lower_bound) { LAUNCH_GATE_BWD(false, true); } else { LAUNCH_GATE_BWD(false, false); }
    }
    #undef LAUNCH_GATE_BWD
}

// Map torch scalar type to the CUDA value type used by the kernels.
template <typename scalar_t>
struct KernelType {
    using type = std::conditional_t<std::is_same_v<scalar_t, at::BFloat16>, cutlass::bfloat16_t,
                std::conditional_t<std::is_same_v<scalar_t, at::Half>, cutlass::half_t, float>>;
};

}  // namespace

void kda_gate_chunk_cumsum(
    torch::Tensor g,
    torch::Tensor A_log,
    std::optional<torch::Tensor> dt_bias,
    torch::Tensor out,
    double scale,
    bool has_scale,
    double lower_bound,
    bool use_lower_bound,
    int64_t chunk_size,
    std::optional<torch::Tensor> cu_seqlens,
    std::optional<torch::Tensor> chunk_indices
) {
    check_gate_inputs(g, A_log, dt_bias, out);
    int64_t B = g.size(0), T_len = g.size(1), H = g.size(2), S = g.size(3);
    auto varlen = resolve_varlen(cu_seqlens, chunk_indices, B, T_len, chunk_size);

    dim3 grid((S + kBS - 1) / kBS, varlen.NT, B * H);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kBFloat16, at::kHalf, g.scalar_type(), "kda_gate_chunk_cumsum", [&] {
        using T = typename KernelType<scalar_t>::type;
        launch_gate_cumsum<T>(
            reinterpret_cast<T const*>(g.data_ptr()), A_log.data_ptr<float>(),
            dt_bias.has_value() ? dt_bias->data_ptr<float>() : nullptr,
            out.data_ptr<float>(),
            float(scale), float(lower_bound),
            dt_bias.has_value(), use_lower_bound, has_scale,
            varlen, int(T_len), int(H), int(S), int(chunk_size), grid, stream
        );
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void chunk_local_cumsum(
    torch::Tensor g,
    torch::Tensor out,
    double scale,
    bool has_scale,
    bool reverse,
    int64_t chunk_size,
    std::optional<torch::Tensor> cu_seqlens,
    std::optional<torch::Tensor> chunk_indices
) {
    TORCH_CHECK(g.is_cuda() && g.is_contiguous() && g.dim() == 4, "g must be 4D contiguous CUDA tensor");
    TORCH_CHECK(out.is_cuda() && out.is_contiguous() && out.dtype() == torch::kFloat32);
    TORCH_CHECK(out.sizes() == g.sizes(), "out must match g shape");
    int64_t B = g.size(0), T_len = g.size(1), H = g.size(2), S = g.size(3);
    auto varlen = resolve_varlen(cu_seqlens, chunk_indices, B, T_len, chunk_size);

    dim3 grid((S + kBS - 1) / kBS, varlen.NT, B * H);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kBFloat16, at::kHalf, g.scalar_type(), "chunk_local_cumsum", [&] {
        using T = typename KernelType<scalar_t>::type;
        launch_local_cumsum<T>(
            reinterpret_cast<T const*>(g.data_ptr()), out.data_ptr<float>(),
            float(scale), has_scale, reverse,
            varlen, int(T_len), int(H), int(S), int(chunk_size), grid, stream
        );
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void kda_gate_bwd(
    torch::Tensor g,
    torch::Tensor A_log,
    std::optional<torch::Tensor> dt_bias,
    torch::Tensor dyg,
    torch::Tensor dg,
    torch::Tensor dA_partial,
    double lower_bound,
    bool use_lower_bound
) {
    TORCH_CHECK(g.is_cuda() && g.is_contiguous() && g.dim() == 4, "g must be 4D contiguous CUDA tensor");
    TORCH_CHECK(A_log.is_cuda() && A_log.is_contiguous() && A_log.dtype() == torch::kFloat32);
    TORCH_CHECK(A_log.dim() == 1 && A_log.size(0) == g.size(2), "A_log must be [H]");
    if (dt_bias.has_value()) {
        TORCH_CHECK(dt_bias->is_cuda() && dt_bias->is_contiguous() && dt_bias->dtype() == torch::kFloat32);
        TORCH_CHECK(dt_bias->numel() == g.size(2) * g.size(3), "dt_bias must have H*D elements");
    }
    TORCH_CHECK(dyg.is_cuda() && dyg.is_contiguous() && dyg.dtype() == torch::kFloat32);
    TORCH_CHECK(dg.is_cuda() && dg.is_contiguous() && dg.dtype() == torch::kFloat32);
    TORCH_CHECK(dyg.sizes() == g.sizes() && dg.sizes() == g.sizes());

    int64_t B = g.size(0), T_len = g.size(1), H = g.size(2), D = g.size(3);
    int64_t T_total = B * T_len;
    int64_t NT = (T_total + kGateBwdBT - 1) / kGateBwdBT;
    TORCH_CHECK(dA_partial.dim() == 2 && dA_partial.size(0) == NT && dA_partial.size(1) == H,
                "dA_partial must be [cdiv(B*T, 32), H]");

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kBFloat16, at::kHalf, g.scalar_type(), "kda_gate_bwd", [&] {
        using T = typename KernelType<scalar_t>::type;
        launch_gate_bwd<T>(
            reinterpret_cast<T const*>(g.data_ptr()), A_log.data_ptr<float>(),
            dt_bias.has_value() ? dt_bias->data_ptr<float>() : nullptr,
            dyg.data_ptr<float>(), dg.data_ptr<float>(), dA_partial.data_ptr<float>(),
            float(lower_bound), use_lower_bound, T_total, int(H), int(D), NT, stream
        );
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
