import torch

from .torch_backend import has_cpu_extension, torch_kda

try:
    from flash_kda_C import fwd as _fwd_raw, get_workspace_size
except ImportError:
    _fwd_raw = None
    get_workspace_size = None


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
        * The native extension requires contiguous CUDA tensors with the
          dtypes listed above.
        * CPU tensors use the differentiable PyTorch backend. Prefer
          :func:`torch_kda` for new training code because it returns tensors
          instead of writing into caller-provided buffers.
    """
    if q.is_cuda and _fwd_raw is not None:
        B, T_seq, H = q.shape[0], q.shape[1], q.shape[2]
        T_total = B * T_seq
        N = cu_seqlens.numel() - 1 if cu_seqlens is not None else B

        workspace = torch.empty(get_workspace_size(T_total, H, N), dtype=torch.uint8, device=q.device)

        _fwd_raw(q, k, v, g, beta, float(scale), out, workspace, A_log, dt_bias, lower_bound,
                 initial_state=initial_state, final_state=final_state, cu_seqlens=cu_seqlens)
        return

    output, state = torch_kda(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=scale,
        initial_state=initial_state,
        output_final_state=final_state is not None,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        use_beta_sigmoid_in_kernel=True,
        state_v_first=True,
        cu_seqlens=cu_seqlens,
        safe_gate=True,
        lower_bound=lower_bound,
        A_log=A_log,
        dt_bias=dt_bias,
    )
    out.copy_(output)
    if final_state is not None:
        final_state.copy_(state)


def has_cuda_extension():
    """Return whether the compiled FlashKDA CUDA extension is available."""
    return _fwd_raw is not None


__all__ = ["fwd", "has_cpu_extension", "has_cuda_extension", "torch_kda"]
