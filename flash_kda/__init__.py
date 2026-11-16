import torch
from flash_kda_C import fwd as _fwd_raw, get_workspace_size

__all__ = ["allocate_workspace", "fwd", "get_workspace_size"]


def _workspace_size_from_inputs(q, cu_seqlens=None):
    """Return the workspace size required by a particular forward call."""
    B, T_seq, H = q.shape[:3]
    N = cu_seqlens.numel() - 1 if cu_seqlens is not None else B
    return get_workspace_size(B * T_seq, H, N)


def allocate_workspace(q, cu_seqlens=None):
    """Allocate a reusable workspace for :func:`fwd`.

    The returned byte tensor is large enough for ``q`` and ``cu_seqlens``. It
    can be passed to multiple sequential calls, including calls with smaller
    shapes. A workspace must not be used by overlapping calls on different
    CUDA streams because the kernels write to it.

    Args:
        q (torch.Tensor): Query tensor whose shape and device determine the
            workspace capacity and placement.
        cu_seqlens (torch.Tensor, optional): Cumulative sequence lengths for a
            variable-length call.
    """
    if q.device.type != "cuda":
        raise ValueError("q must be on CUDA when allocating a workspace")
    return torch.empty(
        _workspace_size_from_inputs(q, cu_seqlens),
        dtype=torch.uint8,
        device=q.device,
    )


def fwd(q, k, v, g, beta, scale, out=None, A_log=None, dt_bias=None,
        lower_bound=None, initial_state=None, final_state=None,
        cu_seqlens=None, workspace=None):
    """FlashKDA forward (Flash Kimi Delta Attention).

    Args:
        q (torch.Tensor): Query, bf16, shape ``[B, T, H, K]``.
        k (torch.Tensor): Key, bf16, shape ``[B, T, H, K]``.
        v (torch.Tensor): Value, bf16, shape ``[B, T, H, V]``.
        g (torch.Tensor): Gate before activation, bf16, shape ``[B, T, H, K]``.
        beta (torch.Tensor): Beta logits (pre-activation; sigmoid is applied
            internally), bf16, shape ``[B, T, H]``.
        scale (float): Scaling factor.
        out (torch.Tensor, optional): Output buffer, bf16, shape
            ``[B, T, H, V]``. When omitted, a tensor is allocated with the same
            shape, dtype, and device as ``q``.
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
        workspace (torch.Tensor, optional): Contiguous CUDA byte tensor used as
            temporary storage. When omitted, an exactly-sized tensor is
            allocated for this call. Use :func:`allocate_workspace` and pass
            the same tensor to sequential calls to avoid repeated allocation.

    Returns:
        torch.Tensor: ``out`` after it has been written by the kernel.

    Notes:
        * Currently requires ``K = V = 128``.
        * All input tensors must be CUDA, contiguous, and have the dtypes
          listed above.
    """
    missing = [
        name for name, value in (
            ("A_log", A_log),
            ("dt_bias", dt_bias),
            ("lower_bound", lower_bound),
        )
        if value is None
    ]
    if missing:
        names = ", ".join(repr(name) for name in missing)
        raise TypeError(f"fwd() missing required argument(s): {names}")

    if out is None:
        out = torch.empty_like(q)
    if workspace is None:
        workspace = allocate_workspace(q, cu_seqlens)

    _fwd_raw(q, k, v, g, beta, float(scale), out, workspace, A_log, dt_bias, lower_bound,
             initial_state=initial_state, final_state=final_state, cu_seqlens=cu_seqlens)
    return out
