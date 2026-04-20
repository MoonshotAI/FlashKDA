import torch
from flash_kda_C import fwd as _fwd_raw, get_workspace_size


def fwd(q, k, v, g, beta, scale, out, A_log, dt_bias, lower_bound, initial_state=None, final_state=None, cu_seqlens=None):
    """FlashKDA forward (Flash Kimi Delta Attention).

    Args:
        q (torch.Tensor): Query, bf16, shape ``[B, T, H, K]``.
        k (torch.Tensor): Key, bf16, shape ``[B, T, H, K]``.
        v (torch.Tensor): Value, bf16, shape ``[B, T, H, V]``.
        g (torch.Tensor): Gate before activation, bf16, shape ``[B, T, H, K]``.
        beta (torch.Tensor): Beta logits (pre-activation; sigmoid is applied
            internally), bf16, shape ``[B, T, H]``.
        scale (float): Scaling factor.
        out (torch.Tensor): Output buffer, bf16, shape ``[B, T, H, V]``. Written
            in place.
        A_log (torch.Tensor): Log-gate parameter, fp32, shape ``[H]``.
        dt_bias (torch.Tensor): Gate bias, fp32, shape ``[H, K]``.
        lower_bound (float): Gate lower bound, expected in ``[-5.0, 0]``.
        initial_state (torch.Tensor, optional): Initial recurrent state, bf16
            or fp32. Shape ``[B, H, V, K]`` for batched mode, or ``[N, H, V, K]``
            for varlen mode. ``None`` means start from zero.
        final_state (torch.Tensor, optional): Output buffer for the final
            recurrent state. Same dtype/shape rules as ``initial_state``.
        cu_seqlens (torch.Tensor, optional): Cumulative sequence lengths, int64,
            shape ``[N+1]``. When provided, ``B`` must be 1.

    Notes:
        * Currently requires ``K = V = 128``.
        * All input tensors must be CUDA, contiguous, and have the dtypes
          listed above.
    """
    B, T_seq, H = q.shape[0], q.shape[1], q.shape[2]
    T_total = B * T_seq
    N = cu_seqlens.numel() - 1 if cu_seqlens is not None else B

    workspace = torch.empty(get_workspace_size(T_total, H, N), dtype=torch.uint8, device=q.device)

    _fwd_raw(q, k, v, g, beta, float(scale), out, workspace, A_log, dt_bias, lower_bound,
             initial_state=initial_state, final_state=final_state, cu_seqlens=cu_seqlens)
