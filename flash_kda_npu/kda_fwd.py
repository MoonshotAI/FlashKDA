"""Public NPU validation and fixed/packed dispatch.

Keep validation ahead of both graph replay and eager execution. The private
execution layer is also used by explicit graph capture, without cache lookup.
"""
from __future__ import annotations

import os
from typing import Optional

import torch

LOWER_BOUND_DEFAULT = -5.0


def is_available() -> bool:
    try:
        import torch_npu  # noqa: F401

        return bool(torch.npu.is_available())
    except (ImportError, AttributeError):
        return False


def _check_public_inputs(q, k, v, g, beta, out, A_log, dt_bias,
                         initial_state, final_state, cu_seqlens):
    """Match the CUDA public tensor contract before any conversion or launch."""
    tensors = dict(q=q, k=k, v=v, g=g, beta=beta, out=out,
                   A_log=A_log, dt_bias=dt_bias)
    if initial_state is not None:
        tensors["initial_state"] = initial_state
    if final_state is not None:
        tensors["final_state"] = final_state
    if cu_seqlens is not None:
        tensors["cu_seqlens"] = cu_seqlens
    for name, tensor in tensors.items():
        if tensor.device.type != "npu":
            raise RuntimeError(f"{name} must be an NPU tensor")
        if tensor.device != q.device:
            raise RuntimeError("all inputs must be on q's NPU device")
        if not tensor.is_contiguous():
            raise RuntimeError(f"{name} must be contiguous")

    for name in ("q", "k", "v", "g", "beta", "out"):
        if tensors[name].dtype != torch.bfloat16:
            raise RuntimeError(f"{name} must be bfloat16")
    for name in ("A_log", "dt_bias"):
        if tensors[name].dtype != torch.float32:
            raise RuntimeError(f"{name} must be float32")
    for name in ("initial_state", "final_state"):
        if name in tensors and tensors[name].dtype not in (torch.bfloat16, torch.float32):
            raise RuntimeError(f"{name} must be bfloat16 or float32")
    if (initial_state is not None and final_state is not None
            and initial_state.dtype != final_state.dtype):
        raise RuntimeError("initial_state and final_state must have the same dtype")

    if q.ndim != 4:
        raise RuntimeError("q must be [B, T, H, D]")
    batch, steps, heads, dim = q.shape
    for name in ("k", "v", "g", "out"):
        if tensors[name].shape != q.shape:
            raise RuntimeError(f"{name} must match q shape")
    if beta.shape != (batch, steps, heads):
        raise RuntimeError("beta must be [B, T, H]")
    if A_log.shape != (heads,):
        raise RuntimeError("A_log must be [H]")
    if dt_bias.shape != (heads, dim):
        raise RuntimeError("dt_bias must be [H, D]")
    if dim != 128:
        raise RuntimeError("currently only supports D == 128")
    sequences = batch
    if cu_seqlens is not None:
        if batch != 1:
            raise RuntimeError("B must be 1 when cu_seqlens is provided")
        if cu_seqlens.dtype != torch.int64:
            raise RuntimeError("cu_seqlens must be int64")
        if cu_seqlens.ndim != 1 or cu_seqlens.numel() < 2:
            raise RuntimeError("cu_seqlens must be 1D with at least 2 elements")
        sequences = cu_seqlens.numel() - 1
    if batch < 1 or heads < 1 or (steps < 1 and cu_seqlens is None):
        raise RuntimeError("NPU AscendC requires B,H > 0 and T > 0 for fixed-length inputs")
    for name in ("initial_state", "final_state"):
        if name in tensors and tensors[name].shape != (sequences, heads, dim, dim):
            raise RuntimeError(f"{name} must be [N, H, D, D]")


def _sequence_offsets(cu_seqlens, steps):
    # Validate metadata before any kernel can partially modify caller buffers.
    offsets = cu_seqlens.detach().cpu().tolist()
    if offsets[0] != 0 or offsets[-1] != steps:
        raise RuntimeError("cu_seqlens must start at 0 and end at T")
    if any(a > b for a, b in zip(offsets, offsets[1:])):
        raise RuntimeError("cu_seqlens must be nondecreasing")
    return offsets


def _run_ascendc(q, k, v, g, beta, scale, out, A_log, dt_bias, lower_bound,
                 initial_state, final_state, cu_seqlens):
    """Execute validated inputs via fixed, aligned-packed, or general-packed paths."""
    from . import ascendc

    # Convert gates in UB to avoid a full-size FP32 copy; beta and state
    # must match the kernel's FP32 ABI.
    if cu_seqlens is None:
        ascendc.kda_fwd_ascendc(
            q, k, v, g, beta.float(), scale=float(scale),
            initial_state=None if initial_state is None else initial_state.float(),
            output_final_state=final_state is not None,
            A_log=A_log, bias=dt_bias, lower_bound=float(lower_bound),
            out=out, final_state=final_state,
        )
        return

    offsets = tuple(_sequence_offsets(cu_seqlens, q.shape[1]))
    if q.shape[1] == 0:
        if final_state is not None:
            if initial_state is None:
                final_state.zero_()
            else:
                final_state.copy_(initial_state)
        ascendc._merge_profiles([], 0, 0)
        return
    uniform = offsets[1] - offsets[0]
    if (uniform > 0 and uniform % ascendc.CHUNK == 0
            and all(b - a == uniform for a, b in zip(offsets, offsets[1:]))):
        # Treat equal aligned sequences as a batch to retain metadata-free
        # indexing and the head-major beta fast path.
        shape = (len(offsets) - 1, uniform, *q.shape[2:])
        ascendc.kda_fwd_ascendc(
            q.view(shape), k.view(shape), v.view(shape), g.view(shape),
            beta.view(shape[:3]).float(), scale=float(scale), out=out.view(shape),
            initial_state=None if initial_state is None else initial_state.float(),
            output_final_state=final_state is not None, final_state=final_state,
            A_log=A_log, bias=dt_bias, lower_bound=float(lower_bound))
        return
    ascendc.kda_fwd_ascendc(
        q, k, v, g, beta.float(), scale=float(scale),
        initial_state=None if initial_state is None else initial_state.float(),
        output_final_state=final_state is not None,
        A_log=A_log, bias=dt_bias, lower_bound=float(lower_bound),
        out=out, final_state=final_state, sequence_offsets=offsets)


def fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    out: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    lower_bound: float = LOWER_BOUND_DEFAULT,
    initial_state: Optional[torch.Tensor] = None,
    final_state: Optional[torch.Tensor] = None,
    cu_seqlens: Optional[torch.Tensor] = None,
) -> None:
    """Run native AscendC with CUDA's public dtypes and no automatic fallback.

    Writes ``out`` and optional ``final_state`` in place and returns ``None``.
    Supports fixed and packed variable-length NPU inputs, including partial chunks.
    Input, import, compilation and launch failures propagate to the caller.
    """
    _check_public_inputs(q, k, v, g, beta, out, A_log, dt_bias,
                         initial_state, final_state, cu_seqlens)
    if os.environ.get("KDA_AUTO_GRAPH", "0") == "1":
        from .auto_graph import try_auto_graph
        if try_auto_graph((q, k, v, g, beta, float(scale), out, A_log, dt_bias,
                           float(lower_bound), initial_state, final_state, cu_seqlens)):
            return
    _run_ascendc(q, k, v, g, beta, float(scale), out, A_log, dt_bias,
                 float(lower_bound), initial_state, final_state, cu_seqlens)
