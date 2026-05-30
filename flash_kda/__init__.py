import torch
from flash_kda_C import fwd as _fwd_raw, get_workspace_size, state_only as _state_only_raw


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


def state_only(k, v, g, beta, A_log, dt_bias, lower_bound, cu_seqlens, num_warmup_chunks, calc_mt=False):
    """FlashKDA state-only kernel: computes final recurrent state without output.

    Runs only the state recurrence on the last `num_warmup_chunks` chunks of
    each segment. Used by CP to efficiently obtain ht without a full forward pass.

    Args:
        k, v, g, beta: Same as fwd().
        A_log, dt_bias, lower_bound: Same as fwd().
        cu_seqlens (torch.Tensor): Cumulative sequence lengths, int64, [N+1].
        num_warmup_chunks (torch.Tensor): Number of trailing chunks to process
            per segment, int32, shape [N].
        calc_mt (bool): If True, also compute the transition matrix mt.

    Returns:
        ht (torch.Tensor): Final state per segment, fp32, shape [N, H, D, D].
        mt (torch.Tensor or None): Transition matrix per segment, fp32,
            shape [N, H, D, D]. Only returned when calc_mt=True.
    """
    H = k.shape[2]
    D = k.shape[3]
    N = cu_seqlens.numel() - 1
    ht = torch.empty(N, H, D, D, dtype=torch.float32, device=k.device)
    mt = torch.empty(N, H, D, D, dtype=torch.float32, device=k.device) if calc_mt else None
    _state_only_raw(k, v, g, beta, A_log, dt_bias, lower_bound, ht, mt, cu_seqlens, num_warmup_chunks)
    if calc_mt:
        return ht, mt
    return ht


def fwd_cp(q, k, v, g, beta, scale, out, A_log, dt_bias, lower_bound,
           initial_state=None, final_state=None, cu_seqlens=None, auto_cp=True):
    """FlashKDA forward with intra-card context parallelism. See flash_kda.cp for details."""
    from flash_kda.cp import fwd_cp as _fwd_cp
    return _fwd_cp(q, k, v, g, beta, scale, out, A_log, dt_bias, lower_bound,
                   initial_state, final_state, cu_seqlens, auto_cp)
