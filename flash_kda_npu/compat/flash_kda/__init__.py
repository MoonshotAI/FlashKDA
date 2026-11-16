"""FlashKDA entry point used by FLA's existing inference backend on NPUs."""

import torch

from flash_kda_npu import fwd as _npu_fwd

__all__ = ["fwd"]


def fwd(q, k, v, g, beta, scale, out, A_log, dt_bias=None, lower_bound=-5.0,
        initial_state=None, final_state=None, cu_seqlens=None):
    """Adapt FLA's FlashKDA call to the native NPU forward interface."""
    if lower_bound is None:
        raise ValueError("lower_bound must be specified for the safe KDA gate")

    if dt_bias is None:
        dt_bias = torch.zeros((v.shape[2], q.shape[-1]), dtype=torch.float32,
                              device=q.device)
    elif dt_bias.ndim == 1:
        dt_bias = dt_bias.view(v.shape[2], q.shape[-1])

    # FLA allocates an FP32 final state even when the initial state is BF16.
    if (initial_state is not None and final_state is not None
            and initial_state.dtype == torch.bfloat16
            and final_state.dtype == torch.float32):
        initial_state = initial_state.float()

    return _npu_fwd(
        q, k, v, g, beta, scale, out, A_log, dt_bias, lower_bound,
        initial_state=initial_state, final_state=final_state,
        cu_seqlens=cu_seqlens,
    )
