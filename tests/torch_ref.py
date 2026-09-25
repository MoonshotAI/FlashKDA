import torch
from torch.utils.cpp_extension import load_inline

# ============================================================
# sigmoid via tanh.approx.f32: tanh(x*0.5)*0.5+0.5
# ============================================================
_sigmoid_cuda_src = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void sigmoid_tanh_fp32_kernel(const float* __restrict__ input,
                                         float* __restrict__ output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float xh = input[idx] * 0.5f;
        float th;
        asm("tanh.approx.f32 %0, %1;" : "=f"(th) : "f"(xh));
        output[idx] = th * 0.5f + 0.5f;
    }
}

torch::Tensor sigmoid_tanh_fp32(torch::Tensor input) {
    auto output = torch::empty_like(input);
    int n = input.numel();
    sigmoid_tanh_fp32_kernel<<<(n + 255) / 256, 256>>>(
        input.data_ptr<float>(), output.data_ptr<float>(), n);
    return output;
}
"""

sigmoid_ext = load_inline(
    name='sigmoid_ext',
    cpp_sources='torch::Tensor sigmoid_tanh_fp32(torch::Tensor input);',
    cuda_sources=_sigmoid_cuda_src,
    functions=['sigmoid_tanh_fp32'],
    extra_cuda_cflags=['-O2'],
    verbose=False,
)

# ============================================================
# Numeric helpers
# ============================================================

LOG2E = 1.4426950408889634


def fp32_ex2_ftz(x):
    if x.dtype == torch.float16:
        x = x.to(torch.float32)
    ret = torch.special.exp2(x)
    ret = torch.where(ret.abs() < torch.finfo(torch.float32).tiny, torch.zeros_like(ret), ret)
    return ret


def fp32_fma(c, a, b):
    assert c.dtype == torch.float32
    assert a.dtype == torch.float32
    assert b.dtype == torch.float32
    return (c.to(torch.float64) + a.to(torch.float64) * b.to(torch.float64)).to(torch.float32)


def l2_normalize_kernel_match(x):
    """L2 normalize matching kernel's warp-shuffle tree reduction with FMA.
    x: [..., D] bf16, D must be 128.
    """
    x_f32 = x.float()
    groups = x_f32.reshape(*x_f32.shape[:-1], 16, 8)

    partials = torch.zeros(*x_f32.shape[:-1], 16, dtype=torch.float32, device=x.device)
    for i in range(8):
        partials = fp32_fma(partials, groups[..., i], groups[..., i])

    for offset in [8, 4, 2, 1]:
        indices = torch.arange(16, device=x.device) ^ offset
        partials = partials + partials[..., indices]

    inv_norm = torch.rsqrt(partials[..., 0:1] + 1e-6)
    return (x_f32 * inv_norm).to(x.dtype)


# ============================================================
# (I + L)^-1: 8x8 fp32 forward substitution + 16x16 bf16 block merge.
# Mirrors the kernel's inv_fwd_subst_fused_1warp bit-for-bit (replaces the fp16
# Neumann series, which loses accuracy when |L| -> 1 near-collinear keys).
# ============================================================

def inv_fwd_subst_16(L):
    """(I + L)^-1 for strictly-lower fp32 L [16, 16] or [N, 16, 16], bf16 out."""
    squeeze = L.dim() == 2
    if squeeze:
        L = L.unsqueeze(0)
    n = L.shape[0]
    device = L.device
    seed = L  # strictly-lower fp32; X = I + L = I + seed

    # Diagonal 8x8 blocks inverted by fp32 forward substitution
    # (kernel-exact FMA order: rank-1 updates below the pivot row,
    # pivot row broadcast from its finalized values at step s).
    inv8 = torch.cat([seed[..., :8, :8], seed[..., 8:, 8:]], dim=0)
    idx8 = torch.arange(8, device=device)
    inv8[:, idx8, idx8] = 1.0
    for s in range(7):
        row_scale = -inv8[:, :, s]  # [2N, 8]
        for p in range(s):
            inv8[:, s + 1:, p] = fp32_fma(
                inv8[:, s + 1:, p], row_scale[:, s + 1:], inv8[:, s, p:p + 1])
        inv8[:, s + 1:, s] = row_scale[:, s + 1:]

    # Merge: P = diag(A^-1, B^-1) bf16, M = [0 0; C 0] bf16 (the kernel
    # quantizes fp32 -> bf16 only at the HMMA inputs), dc = P @ M (fp32 acc),
    # o = bf16(-dc) @ P (fp32 acc), INV = P + bf16(o) (disjoint blocks).
    P = torch.zeros(n, 16, 16, dtype=torch.bfloat16, device=device)
    P[:, :8, :8] = inv8[:n].to(torch.bfloat16)
    P[:, 8:, 8:] = inv8[n:].to(torch.bfloat16)
    M = torch.zeros_like(P)
    M[:, 8:, :8] = seed[..., 8:, :8].to(torch.bfloat16)
    INV = torch.empty_like(P)
    for b in range(n):
        dc = torch.mm(P[b], M[b], out_dtype=torch.float32)
        o = torch.mm((-dc).to(torch.bfloat16), P[b], out_dtype=torch.float32)
        INV[b] = P[b] + o.to(torch.bfloat16)
    return INV.squeeze(0) if squeeze else INV


# ============================================================
# Torch reference implementation
# ============================================================

def torch_ref(q, k, v, g, beta, scale, out, A_log, dt_bias, lower_bound, initial_state=None, final_state=None, cu_seqlens=None, intermediate_state=None):
    """Torch reference, supports both fixed-length and variable-length sequences.

    Input: [B, T, H, D] (4D). B must be 1 when cu_seqlens is provided.

    initial_state/final_state can be:
      - None: no state (zero-init / skip store)
      - bf16 tensor: [N, H, D, D]
      - fp32 tensor: [N, H, D, D] (converted to bf16 for compute, back to fp32 for output)
    """
    assert q.dim() == 4, f"Expected 4D input [B, T, H, D], got {q.dim()}D"
    B = q.shape[0]
    if cu_seqlens is not None:
        assert B == 1, f"B must be 1 when cu_seqlens is provided, got B={B}"
    # Reshape to [B*T, H, D] for internal processing
    q = q.reshape(-1, *q.shape[2:])
    k = k.reshape(-1, *k.shape[2:])
    v = v.reshape(-1, *v.shape[2:])
    g = g.reshape(-1, *g.shape[2:])
    beta = beta.reshape(-1, *beta.shape[2:])
    out = out.reshape(-1, *out.shape[2:])
    if B > 1:
        T_seq = q.shape[0] // B
        cu_seqlens = torch.arange(0, B * T_seq + 1, T_seq, dtype=torch.long, device=q.device)
    _, H, D = q.shape
    CHUNK = 16
    device = q.device
    scale_bf16 = torch.tensor(scale, dtype=torch.bfloat16, device=device)

    q = l2_normalize_kernel_match(q)
    k = l2_normalize_kernel_match(k)

    if A_log is not None:
        assert dt_bias is not None
        assert A_log.dtype == torch.float32
        assert g.dtype == torch.bfloat16
        assert dt_bias.dtype == torch.float32
        g = g.to(torch.float32) + dt_bias.unsqueeze(0)
        a_log_exp = fp32_ex2_ftz(A_log * LOG2E).unsqueeze(0).unsqueeze(-1)
        scale = lower_bound * LOG2E
        g = scale * sigmoid_ext.sigmoid_tanh_fp32(a_log_exp * g)

    state_fp32 = (initial_state is not None and initial_state.dtype == torch.float32) or \
                 (final_state is not None and final_state.dtype == torch.float32)

    if cu_seqlens is None:
        T = q.shape[0]
        cu_seqlens = torch.tensor([0, T], dtype=torch.long, device=device)

    N = len(cu_seqlens) - 1

    if initial_state is not None:
        work_state = initial_state.to(torch.bfloat16).clone()
    else:
        work_state = torch.zeros(N, H, D, D, dtype=torch.bfloat16, device=device)

    tile_base = 0
    for seq_idx in range(N):
        bos = cu_seqlens[seq_idx].item()
        eos = cu_seqlens[seq_idx + 1].item()
        seq_len = eos - bos
        n_chunks = (seq_len + CHUNK - 1) // CHUNK

        for chunk_idx in range(n_chunks):
            t0 = bos + chunk_idx * CHUNK
            actual_len = min(CHUNK, eos - t0)

            for h in range(H):
                g_chunk = torch.zeros(CHUNK, D, dtype=g.dtype, device=device)
                q_chunk = torch.zeros(CHUNK, D, dtype=q.dtype, device=device)
                k_chunk = torch.zeros(CHUNK, D, dtype=k.dtype, device=device)
                v_chunk = torch.zeros(CHUNK, D, dtype=v.dtype, device=device)
                beta_chunk = torch.zeros(CHUNK, dtype=beta.dtype, device=device)

                g_chunk[:actual_len] = g[t0:t0 + actual_len, h, :]
                q_chunk[:actual_len] = q[t0:t0 + actual_len, h, :]
                k_chunk[:actual_len] = k[t0:t0 + actual_len, h, :]
                v_chunk[:actual_len] = v[t0:t0 + actual_len, h, :]
                beta_chunk[:actual_len] = beta[t0:t0 + actual_len, h]

                g_cumsum = g_chunk.cumsum(dim=0)
                g_total = g_cumsum[-1:]
                k_decayed = k_chunk * fp32_ex2_ftz(g_cumsum).to(torch.bfloat16)
                q_decayed = q_chunk * fp32_ex2_ftz(g_cumsum).to(torch.bfloat16) * scale_bf16
                neg_g_cumsum_bf16 = fp32_ex2_ftz(-g_cumsum).to(torch.bfloat16)
                k_inv = k_chunk * neg_g_cumsum_bf16
                g_total_exp_bf16 = fp32_ex2_ftz(g_total).to(torch.bfloat16)
                k_restored = k_inv * g_total_exp_bf16
                L = torch.mm(k_decayed, k_inv.t(), out_dtype=torch.float32)
                Mqk = torch.matmul(q_decayed, k_inv.t())

                # Fuse sigmoid via tanh.approx: beta is bf16 logits
                beta_activated = sigmoid_ext.sigmoid_tanh_fp32(beta_chunk.to(torch.float32))
                beta_val_bf16 = beta_activated.to(torch.bfloat16).unsqueeze(-1)
                L = torch.tril(L, diagonal=-1) * beta_activated.unsqueeze(-1)
                Mqk = torch.tril(Mqk)

                INV = inv_fwd_subst_16(L)

                state_slice = work_state[seq_idx, h]
                v_chunk = v_chunk - torch.matmul(k_decayed, state_slice.t())
                v_chunk = v_chunk * beta_val_bf16

                U = torch.matmul(INV, v_chunk)
                _out = torch.matmul(q_decayed, state_slice.t())
                _out = _out + torch.matmul(Mqk, U)

                delta_s = torch.mm(k_restored.t(), U, out_dtype=torch.float32)

                g_total_exp = fp32_ex2_ftz(g_total)
                g_total_exp = g_total_exp.squeeze(0).unsqueeze(-1)
                work_state[seq_idx, h] = fp32_fma(delta_s, state_slice.to(torch.float32).t(), g_total_exp).to(torch.bfloat16).t()

                out[t0:t0 + actual_len, h] = _out[:actual_len]
                if intermediate_state is not None:
                    intermediate_state[h, tile_base + chunk_idx].copy_(work_state[seq_idx, h])

        tile_base += n_chunks

    if final_state is not None:
        if state_fp32:
            final_state.copy_(work_state.to(torch.float32))
        else:
            final_state.copy_(work_state)
