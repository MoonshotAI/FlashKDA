from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.autograd.function import once_differentiable

try:
    from flash_kda_cpu_C import recurrent_backward as _cpu_recurrent_backward
    from flash_kda_cpu_C import recurrent_forward as _cpu_recurrent_forward
except ImportError:
    _cpu_recurrent_backward = None
    _cpu_recurrent_forward = None


class _CppRecurrentKDA(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, g, beta, initial_state):
        output, final_state = _cpu_recurrent_forward(
            q.contiguous(),
            k.contiguous(),
            v.contiguous(),
            g.contiguous(),
            beta.contiguous(),
            initial_state.contiguous(),
        )
        ctx.save_for_backward(q, k, v, g, beta, initial_state)
        return output, final_state

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output, grad_final_state):
        q, k, v, g, beta, initial_state = ctx.saved_tensors
        if grad_output is None:
            grad_output = torch.zeros_like(v)
        if grad_final_state is None:
            grad_final_state = torch.zeros_like(initial_state)
        return tuple(
            _cpu_recurrent_backward(
                q.contiguous(),
                k.contiguous(),
                v.contiguous(),
                g.contiguous(),
                beta.contiguous(),
                initial_state.contiguous(),
                grad_output.contiguous(),
                grad_final_state.contiguous(),
            )
        )


def has_cpu_extension() -> bool:
    """Return whether the compiled CPU recurrent extension is available."""
    return _cpu_recurrent_forward is not None


def _compute_dtype(dtype: torch.dtype) -> torch.dtype:
    if dtype in (torch.float16, torch.bfloat16):
        return torch.float32
    if dtype in (torch.float32, torch.float64):
        return dtype
    raise TypeError(f"KDA inputs must use a floating-point dtype, got {dtype}")


def _l2_normalize(value: torch.Tensor, eps: float) -> torch.Tensor:
    return value * torch.rsqrt(value.square().sum(dim=-1, keepdim=True) + eps)


def _validate_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
) -> tuple[int, int, int, int, int, int]:
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4 or g.ndim != 4 or beta.ndim != 3:
        raise ValueError("expected q, k, v, g to be 4D and beta to be 3D")
    if q.shape != k.shape:
        raise ValueError(f"q and k must have the same shape, got {q.shape} and {k.shape}")

    batch, sequence_length, query_heads, key_dim = q.shape
    value_heads, value_dim = v.shape[2], v.shape[3]
    if v.shape[:2] != (batch, sequence_length):
        raise ValueError("v must match q's batch and sequence dimensions")
    if value_heads % query_heads != 0:
        raise ValueError(f"value heads ({value_heads}) must be divisible by query heads ({query_heads})")
    if g.shape != (batch, sequence_length, value_heads, key_dim):
        raise ValueError(
            f"g must have shape {(batch, sequence_length, value_heads, key_dim)}, got {tuple(g.shape)}"
        )
    if beta.shape != (batch, sequence_length, value_heads):
        raise ValueError(
            f"beta must have shape {(batch, sequence_length, value_heads)}, got {tuple(beta.shape)}"
        )

    devices = {tensor.device for tensor in (q, k, v, g, beta)}
    if len(devices) != 1:
        raise ValueError("q, k, v, g, and beta must be on the same device")
    if not all(tensor.is_floating_point() for tensor in (q, k, v, g, beta)):
        raise TypeError("q, k, v, g, and beta must use floating-point dtypes")

    return batch, sequence_length, query_heads, key_dim, value_heads, value_dim


def _activate_gate(
    g: torch.Tensor,
    A_log: torch.Tensor | None,
    dt_bias: torch.Tensor | None,
    lower_bound: float | None,
    value_heads: int,
    key_dim: int,
) -> torch.Tensor:
    if A_log is None:
        raise ValueError("A_log is required when use_gate_in_kernel=True")
    if A_log.shape != (value_heads,):
        raise ValueError(f"A_log must have shape {(value_heads,)}, got {tuple(A_log.shape)}")

    if dt_bias is None:
        biased_gate = g
    else:
        if dt_bias.numel() != value_heads * key_dim:
            raise ValueError(f"dt_bias must contain {value_heads * key_dim} values, got {dt_bias.numel()}")
        bias = dt_bias.reshape(1, 1, value_heads, key_dim)
        biased_gate = g + bias
    gate_scale = A_log.exp().reshape(1, 1, value_heads, 1)
    if lower_bound is None:
        return -gate_scale * F.softplus(biased_gate)
    return float(lower_bound) * torch.sigmoid(gate_scale * biased_gate)


def _run_sequence(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    outputs = []
    for token_index in range(q.shape[1]):
        q_token = q[:, token_index]
        k_token = k[:, token_index]
        v_token = v[:, token_index]
        gate_token = g[:, token_index]
        beta_token = beta[:, token_index]

        state = state * gate_token.exp().unsqueeze(-1)
        predicted_value = torch.einsum("bhk,bhkv->bhv", k_token, state)
        value_delta = beta_token.unsqueeze(-1) * (v_token - predicted_value)
        state = state + torch.einsum("bhk,bhv->bhkv", k_token, value_delta)
        outputs.append(torch.einsum("bhk,bhkv->bhv", q_token, state))

    if outputs:
        output = torch.stack(outputs, dim=1)
    else:
        output = v.new_empty(v.shape)
    return output, state


def _run_sequence_dispatch(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    state: torch.Tensor,
    use_cpp_backend: bool | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    can_use_cpp = has_cpu_extension() and q.device.type == "cpu" and q.shape[1] > 0
    if use_cpp_backend is True and not can_use_cpp:
        raise RuntimeError("the compiled FlashKDA CPU extension is not available for these inputs")
    if use_cpp_backend is not False and can_use_cpp:
        return _CppRecurrentKDA.apply(q, k, v, g, beta, state)
    return _run_sequence(q, k, v, g, beta, state)


def torch_kda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    use_gate_in_kernel: bool = False,
    use_beta_sigmoid_in_kernel: bool = False,
    allow_neg_eigval: bool = False,
    state_v_first: bool = False,
    cu_seqlens: torch.Tensor | None = None,
    cu_seqlens_cpu: torch.Tensor | None = None,
    safe_gate: bool = False,
    lower_bound: float | None = None,
    A_log: torch.Tensor | None = None,
    dt_bias: torch.Tensor | None = None,
    l2norm_eps: float = 1e-6,
    disable_recompute: bool = False,
    return_intermediate_states: bool = False,
    cp_context=None,
    chunk_size: int = 64,
    use_cpp_backend: bool | None = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Run differentiable Kimi Delta Attention with native PyTorch operations.

    This backend is intended for CPU research, model prototyping, correctness
    checks, and small workloads. It follows FLA's ``chunk_kda`` argument
    conventions while using the recurrent KDA definition, so PyTorch autograd
    supplies the backward pass automatically.
    """
    if kwargs:
        unknown = ", ".join(sorted(kwargs))
        raise TypeError(f"unexpected keyword arguments: {unknown}")
    del safe_gate, disable_recompute, chunk_size
    if return_intermediate_states:
        raise ValueError("return_intermediate_states is not supported by the PyTorch backend")
    if cp_context is not None:
        raise ValueError("context parallelism is not supported by the PyTorch backend")

    batch, sequence_length, query_heads, key_dim, value_heads, value_dim = _validate_inputs(q, k, v, g, beta)
    compute_dtype = _compute_dtype(v.dtype)
    tensors = (q, k, v, g, beta)
    q_compute, k_compute, v_compute, g_compute, beta_compute = (
        tensor.to(compute_dtype) for tensor in tensors
    )

    if use_qk_l2norm_in_kernel:
        q_compute = _l2_normalize(q_compute, l2norm_eps)
        k_compute = _l2_normalize(k_compute, l2norm_eps)

    group_size = value_heads // query_heads
    q_compute = q_compute.repeat_interleave(group_size, dim=2)
    k_compute = k_compute.repeat_interleave(group_size, dim=2)
    q_compute = q_compute * (key_dim ** -0.5 if scale is None else float(scale))

    if use_gate_in_kernel:
        if A_log is not None and A_log.device != q.device:
            raise ValueError("A_log must be on the same device as q")
        if dt_bias is not None and dt_bias.device != q.device:
            raise ValueError("dt_bias must be on the same device as q")
        g_compute = _activate_gate(
            g_compute,
            None if A_log is None else A_log.to(compute_dtype),
            None if dt_bias is None else dt_bias.to(compute_dtype),
            lower_bound,
            value_heads,
            key_dim,
        )
    if use_beta_sigmoid_in_kernel:
        beta_compute = torch.sigmoid(beta_compute)
        if allow_neg_eigval:
            beta_compute = beta_compute * 2.0

    sequence_lengths = cu_seqlens_cpu if cu_seqlens_cpu is not None else cu_seqlens
    if sequence_lengths is not None:
        if batch != 1:
            raise ValueError("batch size must be 1 when cu_seqlens is provided")
        if sequence_lengths.ndim != 1 or sequence_lengths.numel() < 2:
            raise ValueError("cu_seqlens must be a 1D tensor with at least two elements")
        sequence_offsets = sequence_lengths.detach().to(device="cpu", dtype=torch.long).tolist()
        if sequence_offsets[0] != 0 or sequence_offsets[-1] != sequence_length:
            raise ValueError("cu_seqlens must start at 0 and end at the total sequence length")
        if any(end < start for start, end in zip(sequence_offsets, sequence_offsets[1:])):
            raise ValueError("cu_seqlens must be nondecreasing")
        state_count = len(sequence_offsets) - 1
    else:
        sequence_offsets = None
        state_count = batch

    expected_state_shape = (
        (state_count, value_heads, value_dim, key_dim)
        if state_v_first
        else (state_count, value_heads, key_dim, value_dim)
    )
    if initial_state is None:
        state = v_compute.new_zeros(state_count, value_heads, key_dim, value_dim)
    else:
        if tuple(initial_state.shape) != expected_state_shape:
            raise ValueError(f"initial_state must have shape {expected_state_shape}, got {tuple(initial_state.shape)}")
        if initial_state.device != q.device:
            raise ValueError("initial_state must be on the same device as q")
        state = initial_state.to(compute_dtype)
        if state_v_first:
            state = state.transpose(-1, -2)

    if sequence_offsets is None:
        output, final_state = _run_sequence_dispatch(
            q_compute,
            k_compute,
            v_compute,
            g_compute,
            beta_compute,
            state,
            use_cpp_backend,
        )
    else:
        output_parts = []
        final_states = []
        for sequence_index, (start, end) in enumerate(zip(sequence_offsets, sequence_offsets[1:])):
            sequence_output, sequence_state = _run_sequence_dispatch(
                q_compute[:, start:end],
                k_compute[:, start:end],
                v_compute[:, start:end],
                g_compute[:, start:end],
                beta_compute[:, start:end],
                state[sequence_index:sequence_index + 1],
                use_cpp_backend,
            )
            output_parts.append(sequence_output)
            final_states.append(sequence_state)
        output = torch.cat(output_parts, dim=1)
        final_state = torch.cat(final_states, dim=0)

    if state_v_first:
        final_state = final_state.transpose(-1, -2)
    return output.to(v.dtype), final_state if output_final_state else None
