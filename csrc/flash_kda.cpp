#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include "fwd.h"

// Declared in warmup_chunks.cu
void get_warmup_chunks_cuda(
    torch::Tensor g, torch::Tensor A_log, torch::Tensor dt_bias,
    double lower_bound, torch::Tensor cu_seqlens,
    int chunk_size, double threshold,
    torch::Tensor num_warmup, torch::Tensor fallback);

int64_t get_workspace_size(
    int64_t T_total,
    int64_t H,
    int64_t N = 1
) {
    constexpr int CHUNK = 16;
    constexpr int D = 128;

    // Upper bound: each of N sequences adds at most 1 extra tile vs floor division
    int64_t total_tiles = (T_total + CHUNK - 1) / CHUNK + N;

    static_assert(CHUNK * D * 2 % 128 == 0, "k_decayed/q_decayed/k_restored size must be 128-byte aligned");
    static_assert(D * 4 % 128 == 0, "g_total size must be 128-byte aligned");
    static_assert(CHUNK * CHUNK * 2 % 128 == 0, "INV/Mqk size must be 128-byte aligned");

    int64_t per_tile_bytes = 3 * CHUNK * D * 2 + D * 4 + 2 * CHUNK * CHUNK * 2;

    return H * total_tiles * per_tile_bytes;
}

void fwd(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor g,
    torch::Tensor beta,
    float scale,
    torch::Tensor out,
    torch::Tensor workspace,
    torch::Tensor A_log,
    torch::Tensor dt_bias,
    double lower_bound,
    std::optional<torch::Tensor> initial_state = std::nullopt,
    std::optional<torch::Tensor> final_state = std::nullopt,
    std::optional<torch::Tensor> cu_seqlens = std::nullopt
) {
    TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda() && g.is_cuda() && beta.is_cuda() && out.is_cuda() && workspace.is_cuda(),
                "all tensors must be on CUDA");
    TORCH_CHECK(q.is_contiguous() && k.is_contiguous() && v.is_contiguous() && g.is_contiguous() && beta.is_contiguous() && out.is_contiguous() && workspace.is_contiguous(),
                "all tensors must be contiguous");

    TORCH_CHECK(q.dtype() == torch::kBFloat16, "q must be bfloat16");
    TORCH_CHECK(k.dtype() == torch::kBFloat16, "k must be bfloat16");
    TORCH_CHECK(v.dtype() == torch::kBFloat16, "v must be bfloat16");
    TORCH_CHECK(g.dtype() == torch::kBFloat16, "g must be bfloat16");
    TORCH_CHECK(beta.dtype() == torch::kBFloat16, "beta must be bfloat16");
    TORCH_CHECK(out.dtype() == torch::kBFloat16, "out must be bfloat16");

    // Validate state tensors if present
    bool has_state_in = initial_state.has_value();
    bool has_state_out = final_state.has_value();
    bool state_fp32 = false;

    if (has_state_in) {
        auto& is = initial_state.value();
        TORCH_CHECK(is.is_cuda() && is.is_contiguous(), "initial_state must be contiguous CUDA tensor");
        TORCH_CHECK(is.dtype() == torch::kBFloat16 || is.dtype() == torch::kFloat32,
                     "initial_state must be bfloat16 or float32");
        if (is.dtype() == torch::kFloat32) state_fp32 = true;
    }
    if (has_state_out) {
        auto& fs = final_state.value();
        TORCH_CHECK(fs.is_cuda() && fs.is_contiguous(), "final_state must be contiguous CUDA tensor");
        TORCH_CHECK(fs.dtype() == torch::kBFloat16 || fs.dtype() == torch::kFloat32,
                     "final_state must be bfloat16 or float32");
        if (fs.dtype() == torch::kFloat32) state_fp32 = true;
    }
    // If both present, dtypes must match
    if (has_state_in && has_state_out) {
        TORCH_CHECK(initial_state->dtype() == final_state->dtype(),
                     "initial_state and final_state must have the same dtype");
    }

    TORCH_CHECK(A_log.is_cuda() && A_log.is_contiguous(), "A_log must be contiguous CUDA tensor");
    TORCH_CHECK(A_log.dtype() == torch::kFloat32, "A_log must be float32");
    TORCH_CHECK(dt_bias.is_cuda() && dt_bias.is_contiguous(), "dt_bias must be contiguous CUDA tensor");
    TORCH_CHECK(dt_bias.dtype() == torch::kFloat32, "dt_bias must be float32");

    // Accept 4D input [B, T, H, D]
    TORCH_CHECK(q.dim() == 4, "q must be [B, T, H, D]");
    TORCH_CHECK(k.dim() == 4, "k must be [B, T, H, D]");
    TORCH_CHECK(v.dim() == 4, "v must be [B, T, H, D]");
    TORCH_CHECK(g.dim() == 4, "g must be [B, T, H, D]");
    TORCH_CHECK(beta.dim() == 3, "beta must be [B, T, H]");
    TORCH_CHECK(out.dim() == 4, "out must be [B, T, H, D]");

    int64_t B = q.size(0);
    int64_t T_seq = q.size(1);
    int64_t H = q.size(2);
    int64_t D = q.size(3);
    int64_t T_total = B * T_seq;

    TORCH_CHECK(k.sizes() == q.sizes(), "k must match q shape");
    TORCH_CHECK(v.sizes() == q.sizes(), "v must match q shape");
    TORCH_CHECK(g.sizes() == q.sizes(), "g must match q shape");
    TORCH_CHECK(out.sizes() == q.sizes(), "out must match q shape");
    TORCH_CHECK(beta.size(0) == B && beta.size(1) == T_seq && beta.size(2) == H,
                "beta must be [B, T, H] matching q");

    TORCH_CHECK(A_log.dim() == 1 && A_log.size(0) == H, "A_log must be [H]");
    TORCH_CHECK(dt_bias.dim() == 2 && dt_bias.size(0) == H && dt_bias.size(1) == D, "dt_bias must be [H, D]");

    TORCH_CHECK(D == 128, "currently only supports D == 128");

    // Flatten [B, T, H, D] -> [B*T, H, D] (contiguous, same data pointer)
    auto q_3d = q.reshape({T_total, H, D});
    auto k_3d = k.reshape({T_total, H, D});
    auto v_3d = v.reshape({T_total, H, D});
    auto g_3d = g.reshape({T_total, H, D});
    auto beta_2d = beta.reshape({T_total, H});
    auto out_3d = out.reshape({T_total, H, D});

    auto q_ptr = reinterpret_cast<cutlass::bfloat16_t const*>(q_3d.data_ptr<at::BFloat16>());
    auto k_ptr = reinterpret_cast<cutlass::bfloat16_t const*>(k_3d.data_ptr<at::BFloat16>());
    auto v_ptr = reinterpret_cast<cutlass::bfloat16_t const*>(v_3d.data_ptr<at::BFloat16>());
    auto g_ptr = reinterpret_cast<cutlass::bfloat16_t const*>(g_3d.data_ptr<at::BFloat16>());
    float scale_f = scale;
    auto out_ptr = reinterpret_cast<cutlass::bfloat16_t*>(out_3d.data_ptr<at::BFloat16>());
    auto A_log_ptr = A_log.data_ptr<float>();
    auto dt_bias_ptr = dt_bias.data_ptr<float>();
    float gate_scale = float(lower_bound * 1.4426950408889634);

    // Transpose beta: [T_total, H] -> [H, T_total] (1D TMA, no T alignment constraint)
    auto beta_t = beta_2d.t().contiguous();
    auto beta_t_ptr = reinterpret_cast<cutlass::bfloat16_t const*>(beta_t.data_ptr<at::BFloat16>());

    auto workspace_ptr = workspace.data_ptr();

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    constexpr int CHUNK = 16;

    // Get state pointers (nullptr if not present)
    void const* initial_state_raw = has_state_in ? initial_state->data_ptr() : nullptr;
    void* final_state_raw = has_state_out ? final_state->data_ptr() : nullptr;

    // Determine cu_seqlens and N
    bool is_varlen = cu_seqlens.has_value();
    int64_t N_val;
    int64_t const* cu_seqlens_dev = nullptr;

    if (is_varlen) {
        TORCH_CHECK(B == 1, "B must be 1 when cu_seqlens is provided");
        auto& cu_seqlens_t = cu_seqlens.value();
        TORCH_CHECK(cu_seqlens_t.is_cuda(), "cu_seqlens must be on CUDA");
        TORCH_CHECK(cu_seqlens_t.dtype() == torch::kLong, "cu_seqlens must be int64");
        TORCH_CHECK(cu_seqlens_t.dim() == 1, "cu_seqlens must be 1D");
        N_val = cu_seqlens_t.numel() - 1;
        TORCH_CHECK(N_val > 0, "cu_seqlens must have at least 2 elements");
        cu_seqlens_dev = cu_seqlens_t.data_ptr<int64_t>();
    } else {
        N_val = B;
    }

    // Validate state shapes: always [N, H, D, D]
    if (has_state_in) {
        auto& is = initial_state.value();
        TORCH_CHECK(is.dim() == 4, "initial_state must be [N, H, D, D]");
        TORCH_CHECK(is.size(0) == N_val && is.size(1) == H && is.size(2) == D && is.size(3) == D,
                     "initial_state must be [N, H, D, D]");
    }
    if (has_state_out) {
        auto& fs = final_state.value();
        TORCH_CHECK(fs.dim() == 4, "final_state must be [N, H, D, D]");
        TORCH_CHECK(fs.size(0) == N_val && fs.size(1) == H && fs.size(2) == D && fs.size(3) == D,
                     "final_state must be [N, H, D, D]");
    }

    int total_tiles;
    if (is_varlen) {
        total_tiles = int((T_total + CHUNK - 1) / CHUNK + N_val);  // upper bound for varlen
    } else {
        total_tiles = int(N_val * ((T_seq + CHUNK - 1) / CHUNK));   // exact for batched
    }

    // Dispatch based on state configuration and varlen
    #define LAUNCH(HI, HO, FP32, VL) \
        launch_fwd<128, HI, HO, FP32, VL>( \
            q_ptr, k_ptr, v_ptr, g_ptr, beta_t_ptr, \
            initial_state_raw, scale_f, final_state_raw, out_ptr, \
            workspace_ptr, total_tiles, \
            int(T_total), int(H), int(N_val), cu_seqlens_dev, \
            A_log_ptr, dt_bias_ptr, gate_scale, stream)

    #define DISPATCH_STATE(VL) \
        if (!has_state_in && !has_state_out) { \
            LAUNCH(false, false, false, VL); \
        } else if (has_state_in && has_state_out && state_fp32) { \
            LAUNCH(true, true, true, VL); \
        } else if (has_state_in && has_state_out && !state_fp32) { \
            LAUNCH(true, true, false, VL); \
        } else if (!has_state_in && has_state_out && state_fp32) { \
            LAUNCH(false, true, true, VL); \
        } else if (!has_state_in && has_state_out && !state_fp32) { \
            LAUNCH(false, true, false, VL); \
        } else if (has_state_in && !has_state_out && state_fp32) { \
            LAUNCH(true, false, true, VL); \
        } else { \
            LAUNCH(true, false, false, VL); \
        }

    if (is_varlen) {
        DISPATCH_STATE(true);
    } else {
        DISPATCH_STATE(false);
    }

    #undef DISPATCH_STATE
    #undef LAUNCH
}

void state_only(
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor g,
    torch::Tensor beta,
    torch::Tensor A_log,
    torch::Tensor dt_bias,
    double lower_bound,
    torch::Tensor final_state,
    std::optional<torch::Tensor> mt_out,
    torch::Tensor cu_seqlens,
    torch::Tensor num_warmup_chunks
) {
    TORCH_CHECK(k.is_cuda() && v.is_cuda() && g.is_cuda() && beta.is_cuda(),
                "all tensors must be on CUDA");
    TORCH_CHECK(k.is_contiguous() && v.is_contiguous() && g.is_contiguous() && beta.is_contiguous(),
                "all tensors must be contiguous");
    TORCH_CHECK(k.dtype() == torch::kBFloat16, "k must be bfloat16");
    TORCH_CHECK(v.dtype() == torch::kBFloat16, "v must be bfloat16");
    TORCH_CHECK(g.dtype() == torch::kBFloat16, "g must be bfloat16");
    TORCH_CHECK(beta.dtype() == torch::kBFloat16, "beta must be bfloat16");

    TORCH_CHECK(A_log.is_cuda() && A_log.is_contiguous(), "A_log must be contiguous CUDA");
    TORCH_CHECK(A_log.dtype() == torch::kFloat32, "A_log must be float32");
    TORCH_CHECK(dt_bias.is_cuda() && dt_bias.is_contiguous(), "dt_bias must be contiguous CUDA");
    TORCH_CHECK(dt_bias.dtype() == torch::kFloat32, "dt_bias must be float32");

    TORCH_CHECK(final_state.is_cuda() && final_state.is_contiguous(), "final_state must be contiguous CUDA");
    TORCH_CHECK(final_state.dtype() == torch::kFloat32, "final_state must be float32");

    bool has_mt = mt_out.has_value();
    if (has_mt) {
        auto& mt = mt_out.value();
        TORCH_CHECK(mt.is_cuda() && mt.is_contiguous(), "mt must be contiguous CUDA");
        TORCH_CHECK(mt.dtype() == torch::kFloat32, "mt must be float32");
    }

    TORCH_CHECK(cu_seqlens.is_cuda() && cu_seqlens.is_contiguous(), "cu_seqlens must be contiguous CUDA");
    TORCH_CHECK(cu_seqlens.dtype() == torch::kLong, "cu_seqlens must be int64");

    TORCH_CHECK(num_warmup_chunks.is_cuda() && num_warmup_chunks.is_contiguous(),
                "num_warmup_chunks must be contiguous CUDA");
    TORCH_CHECK(num_warmup_chunks.dtype() == torch::kInt32, "num_warmup_chunks must be int32");

    TORCH_CHECK(k.dim() == 4, "k must be [B, T, H, D]");
    int64_t B = k.size(0);
    TORCH_CHECK(B == 1, "state_only requires B=1 (varlen mode)");
    int64_t T_seq = k.size(1);
    int64_t H = k.size(2);
    int64_t D = k.size(3);
    TORCH_CHECK(D == 128, "currently only supports D == 128");
    int64_t T_total = B * T_seq;

    int64_t N = cu_seqlens.numel() - 1;
    TORCH_CHECK(N > 0, "cu_seqlens must have at least 2 elements");
    TORCH_CHECK(num_warmup_chunks.numel() == N, "num_warmup_chunks must have N elements");

    TORCH_CHECK(final_state.dim() == 4, "final_state must be [N, H, D, D]");
    TORCH_CHECK(final_state.size(0) == N && final_state.size(1) == H &&
                final_state.size(2) == D && final_state.size(3) == D,
                "final_state must be [N, H, D, D]");
    if (has_mt) {
        auto& mt = mt_out.value();
        TORCH_CHECK(mt.dim() == 4, "mt must be [N, H, D, D]");
        TORCH_CHECK(mt.size(0) == N && mt.size(1) == H &&
                    mt.size(2) == D && mt.size(3) == D,
                    "mt must be [N, H, D, D]");
    }

    constexpr int CHUNK = 16;
    int64_t total_tiles = (T_total + CHUNK - 1) / CHUNK + N;

    // Allocate workspace for kernel1
    int64_t ws_bytes = get_workspace_size(T_total, H, N);
    auto workspace = torch::empty({ws_bytes}, torch::TensorOptions().dtype(torch::kUInt8).device(k.device()));

    // Transpose beta: [T_total, H] -> [H, T_total]
    auto beta_2d = beta.reshape({T_total, H});
    auto beta_t = beta_2d.t().contiguous();

    auto k_ptr = reinterpret_cast<cutlass::bfloat16_t const*>(k.reshape({T_total, H, D}).data_ptr<at::BFloat16>());
    auto v_ptr = reinterpret_cast<cutlass::bfloat16_t const*>(v.reshape({T_total, H, D}).data_ptr<at::BFloat16>());
    auto g_ptr = reinterpret_cast<cutlass::bfloat16_t const*>(g.reshape({T_total, H, D}).data_ptr<at::BFloat16>());
    auto beta_t_ptr = reinterpret_cast<cutlass::bfloat16_t const*>(beta_t.data_ptr<at::BFloat16>());
    auto A_log_ptr = A_log.data_ptr<float>();
    auto dt_bias_ptr = dt_bias.data_ptr<float>();
    float gate_scale = float(lower_bound * 1.4426950408889634);

    auto workspace_ptr = workspace.data_ptr();
    auto final_state_ptr = final_state.data_ptr<float>();
    auto cu_seqlens_ptr = cu_seqlens.data_ptr<int64_t>();
    auto warmup_ptr = num_warmup_chunks.data_ptr<int32_t>();

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    // kernel1 needs q to compute q_decayed, but state_only doesn't use it.
    // Pass k as dummy q — q_decayed/Mqk workspace entries are unused.
    auto q_ptr = k_ptr;
    float scale = 1.0f;  // dummy, not used by state_only kernel

    // Run kernel1 with warmup-only optimization (non-warmup tiles early-exit)
    launch_kernel1_warmup_only<128>(
        q_ptr, k_ptr, g_ptr, beta_t_ptr,
        scale, workspace_ptr, int(total_tiles),
        int(T_total), int(H), int(N), cu_seqlens_ptr,
        A_log_ptr, dt_bias_ptr, gate_scale, warmup_ptr, stream
    );

    // Run state_only kernel (computes ht from h0=0)
    launch_state_only<128>(
        v_ptr, beta_t_ptr,
        workspace_ptr, final_state_ptr,
        int(total_tiles), int(T_total), int(H), int(N),
        cu_seqlens_ptr, warmup_ptr, stream
    );

    // Run mt kernel if requested (computes transition matrix)
    if (has_mt) {
        auto mt_ptr = mt_out.value().data_ptr<float>();
        launch_mt_only<128>(
            v_ptr, beta_t_ptr,
            workspace_ptr, mt_ptr,
            int(total_tiles), int(T_total), int(H), int(N),
            cu_seqlens_ptr, warmup_ptr, stream
        );
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fwd", &fwd, "FlashKDA Forward (CUDA)",
        py::arg("q"), py::arg("k"), py::arg("v"), py::arg("g"), py::arg("beta"),
        py::arg("scale"), py::arg("out"),
        py::arg("workspace"),
        py::arg("A_log"), py::arg("dt_bias"), py::arg("lower_bound"),
        py::arg("initial_state") = py::none(), py::arg("final_state") = py::none(),
        py::arg("cu_seqlens") = py::none());
    m.def("get_workspace_size",
        static_cast<int64_t(*)(int64_t, int64_t, int64_t)>(&get_workspace_size),
        "Get workspace size in bytes",
        py::arg("T_total"), py::arg("H"),
        py::arg("N") = 1);
    m.def("state_only", &state_only, "FlashKDA State-Only (CUDA)",
        py::arg("k"), py::arg("v"), py::arg("g"), py::arg("beta"),
        py::arg("A_log"), py::arg("dt_bias"), py::arg("lower_bound"),
        py::arg("final_state"), py::arg("mt") = py::none(),
        py::arg("cu_seqlens"), py::arg("num_warmup_chunks"));
    m.def("get_warmup_chunks", &get_warmup_chunks_cuda, "Get warmup chunks (CUDA)",
        py::arg("g"), py::arg("A_log"), py::arg("dt_bias"),
        py::arg("lower_bound"), py::arg("cu_seqlens"),
        py::arg("chunk_size"), py::arg("threshold"),
        py::arg("num_warmup"), py::arg("fallback"));
}
