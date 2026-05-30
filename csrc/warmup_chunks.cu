#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cmath>

// get_warmup_chunks CUDA kernel
// Grid: (N,)  Block: (H,) where H <= 256
// Each block processes one segment, each thread handles one head.
// Scans backwards from the end accumulating gate decay until all heads converge.
__global__ void get_warmup_chunks_kernel(
    const __nv_bfloat16* __restrict__ g_ptr,  // [T, H, D]
    const float* __restrict__ A_log,          // [H]
    const float* __restrict__ dt_bias,        // [H, D]
    float gate_scale,                         // lower_bound * log2(e)
    int chunk_size,
    float threshold,
    const int64_t* __restrict__ cu_seqlens,   // [N+1]
    int H, int D,
    int32_t* __restrict__ num_warmup,         // output [N]
    bool* __restrict__ fallback               // output [N]
) {
    int seg_idx = blockIdx.x;
    int h = threadIdx.x;  // head index

    if (h >= H) return;

    int64_t bos = cu_seqlens[seg_idx];
    int64_t eos = cu_seqlens[seg_idx + 1];
    int seg_len = (int)(eos - bos);
    int nc = (seg_len + chunk_size - 1) / chunk_size;

    // Precompute exp(A_log[h]) and mean(dt_bias[h, :])
    float a_exp = expf(A_log[h]);
    float dt_mean = 0.0f;
    for (int d = 0; d < D; d++) {
        dt_mean += dt_bias[h * D + d];
    }
    dt_mean /= (float)D;

    // Per-head cumulative decay (negative, grows more negative)
    float g_cumsum = 0.0f;

    // Shared memory for cross-head reduction
    extern __shared__ float smem[];  // [H]

    int result_warmup = nc;
    bool found = false;

    for (int c = 0; c < nc && !found; c++) {
        int chunk_end = (int)(eos - c * chunk_size - 1);
        if (chunk_end < (int)bos) chunk_end = (int)bos;

        // Compute mean(g[chunk_end, h, :]) over D dimension
        float g_mean = 0.0f;
        const __nv_bfloat16* g_row = g_ptr + (int64_t)chunk_end * H * D + (int64_t)h * D;
        for (int d = 0; d < D; d++) {
            g_mean += __bfloat162float(g_row[d]);
        }
        g_mean /= (float)D;

        // gate_activation = gate_scale * sigmoid(exp(A_log[h]) * (g_mean + dt_mean))
        float x = a_exp * (g_mean + dt_mean);
        float sig = 1.0f / (1.0f + expf(-x));
        float decay = gate_scale * sig * (float)chunk_size;

        g_cumsum += decay;

        // Reduction: find max g_cumsum across heads (max = least negative = slowest head)
        smem[h] = g_cumsum;
        __syncthreads();

        // Simple tree reduction for max
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (h < stride && h + stride < H) {
                smem[h] = fmaxf(smem[h], smem[h + stride]);
            }
            __syncthreads();
        }

        // Check convergence: all heads' cumulative decay exceeded threshold
        float max_val = smem[0];
        if (max_val < threshold) {
            result_warmup = c + 1;
            found = true;
        }
        __syncthreads();
    }

    // Write output (only head 0)
    if (h == 0) {
        num_warmup[seg_idx] = result_warmup;
        fallback[seg_idx] = !found;
    }
}

// Host wrapper
void get_warmup_chunks_cuda(
    torch::Tensor g,            // [1, T, H, D] bf16
    torch::Tensor A_log,        // [H] fp32
    torch::Tensor dt_bias,      // [H, D] fp32
    double lower_bound,
    torch::Tensor cu_seqlens,   // [N+1] int64
    int chunk_size,
    double threshold,
    torch::Tensor num_warmup,   // output [N] int32
    torch::Tensor fallback      // output [N] bool
) {
    int N = cu_seqlens.numel() - 1;
    int H = A_log.numel();
    int D = dt_bias.size(1);

    float gate_scale = (float)(lower_bound * 1.4426950408889634);

    auto g_flat = g.reshape({-1, H, D});  // [T, H, D]

    dim3 grid(N);
    dim3 block(H);
    int smem_size = H * sizeof(float);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    get_warmup_chunks_kernel<<<grid, block, smem_size, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(g_flat.data_ptr<at::BFloat16>()),
        A_log.data_ptr<float>(),
        dt_bias.data_ptr<float>(),
        gate_scale,
        chunk_size,
        (float)threshold,
        cu_seqlens.data_ptr<int64_t>(),
        H, D,
        num_warmup.data_ptr<int32_t>(),
        reinterpret_cast<bool*>(fallback.data_ptr<bool>())
    );
}
