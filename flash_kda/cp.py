"""Exact PyTorch transition algebra for chunked KDA context parallelism.

This correctness prototype uses the kernel's post-preprocessing space: key is
already L2-normalized, log_decay is the A_log/dt_bias gate in log2 units, and
beta is already sigmoid-activated. A chunk maps S [H, V, K] as S @ A + B.
The affine map, unlike a boundary-state snapshot, supports an exact CP scan.

"Exact" here means the high-precision affine recurrence. Materializing the
state as BF16 after every chunk introduces a nonlinear rounding step, so the
legacy bitwise path requires a serial fallback rather than a fixed (A, B) map.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - exercised only by CPU-only PyTorch builds.
    triton = None
    tl = None


KDA_CHUNK_SIZE = 16


if triton is not None:

    @triton.jit
    def _kda_segment_accumulate_kernel(
        key_decayed_ptr,
        right_factor_ptr,
        total_decay_ptr,
        value_ptr,
        matrix_ptr,
        bias_ptr,
        num_chunks,
        HEADS: tl.constexpr,
        KEY_DIM: tl.constexpr,
        VALUE_DIM: tl.constexpr,
        CHUNK: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        """Keep one A/B row resident while applying every low-rank chunk."""
        program = tl.program_id(0)
        rows_per_head = KEY_DIM + VALUE_DIM
        head = program // rows_per_head
        row = program - head * rows_per_head
        offsets = tl.arange(0, BLOCK_K)
        is_matrix = row < KEY_DIM
        state = tl.where(is_matrix & (offsets == row), 1.0, 0.0).to(tl.float32)
        value_row = row - KEY_DIM

        for chunk in range(0, num_chunks):
            decay_offsets = (chunk * HEADS + head) * KEY_DIM + offsets
            decay = tl.load(total_decay_ptr + decay_offsets)
            update = tl.zeros((BLOCK_K,), dtype=tl.float32)
            for token in tl.static_range(0, CHUNK):
                factor_offsets = (
                    ((chunk * HEADS + head) * CHUNK + token) * KEY_DIM + offsets
                )
                key_decayed = tl.load(key_decayed_ptr + factor_offsets)
                coefficient = tl.sum(state * key_decayed, axis=0)
                value_offset = (
                    ((chunk * HEADS + head) * CHUNK + token) * VALUE_DIM
                    + value_row
                )
                value = tl.load(
                    value_ptr + value_offset, mask=~is_matrix, other=0.0
                )
                right_factor = tl.load(right_factor_ptr + factor_offsets)
                update += (value - coefficient) * right_factor
            state = state * decay + update

        matrix_offsets = head * KEY_DIM * KEY_DIM + row * KEY_DIM + offsets
        bias_offsets = head * VALUE_DIM * KEY_DIM + value_row * KEY_DIM + offsets
        tl.store(matrix_ptr + matrix_offsets, state, mask=is_matrix)
        tl.store(bias_ptr + bias_offsets, state, mask=~is_matrix)


@dataclass(frozen=True)
class KDATransition:
    """Affine state map with shapes [..., H, K, K] and [..., H, V, K]."""

    matrix: torch.Tensor
    bias: torch.Tensor


def _check_transition(transition: KDATransition) -> None:
    if transition.matrix.ndim < 3 or transition.bias.ndim < 3:
        raise ValueError("transition tensors must have at least 3 dimensions")
    if transition.matrix.shape[:-3] != transition.bias.shape[:-3]:
        raise ValueError("transition matrix and bias must have identical batch shapes")
    heads, key_dim, key_dim_2 = transition.matrix.shape[-3:]
    if key_dim != key_dim_2:
        raise ValueError("transition matrix must have shape [..., H, K, K]")
    if transition.bias.shape[-3] != heads or transition.bias.shape[-1] != key_dim:
        raise ValueError("transition bias must have shape [..., H, V, K]")
    if transition.matrix.device != transition.bias.device:
        raise ValueError("transition matrix and bias must share a device")
    if transition.matrix.dtype != transition.bias.dtype:
        raise ValueError("transition matrix and bias must share a dtype")


def identity_transition(
    heads: int,
    value_dim: int,
    key_dim: int,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> KDATransition:
    """Return the identity map for a [H, V, K] recurrent state."""
    if heads <= 0 or value_dim <= 0 or key_dim <= 0:
        raise ValueError("heads, value_dim, and key_dim must be positive")
    matrix = torch.eye(key_dim, device=device, dtype=dtype).expand(heads, -1, -1).clone()
    bias = torch.zeros((heads, value_dim, key_dim), device=device, dtype=dtype)
    return KDATransition(matrix=matrix, bias=bias)


def apply_transition(state: torch.Tensor, transition: KDATransition) -> torch.Tensor:
    """Apply an affine transition, broadcasting leading segment dimensions."""
    _check_transition(transition)
    if state.ndim < 3:
        raise ValueError("state must have shape [..., H, V, K]")
    if state.shape[-3:] != transition.bias.shape[-3:]:
        raise ValueError("state shape is incompatible with transition")
    if state.device != transition.matrix.device:
        raise ValueError("state and transition must share a device")
    try:
        torch.broadcast_shapes(state.shape[:-3], transition.matrix.shape[:-3])
    except RuntimeError as error:
        raise ValueError("state and transition batch shapes are not broadcastable") from error
    state = state.to(transition.matrix.dtype)
    return state @ transition.matrix + transition.bias


def compose_transitions(first: KDATransition, second: KDATransition) -> KDATransition:
    """Return the transition for applying first and then second.

    For first(S) = S @ A0 + B0 and second(S) = S @ A1 + B1, this returns
    S @ (A0 @ A1) + (B0 @ A1 + B1).
    """
    _check_transition(first)
    _check_transition(second)
    if first.matrix.shape != second.matrix.shape or first.bias.shape != second.bias.shape:
        raise ValueError("transitions must have identical shapes")
    if first.matrix.device != second.matrix.device or first.matrix.dtype != second.matrix.dtype:
        raise ValueError("transitions must share device and dtype")
    return KDATransition(
        matrix=first.matrix @ second.matrix,
        bias=first.bias @ second.matrix + second.bias,
    )


def compose_transition_sequence(transitions: Sequence[KDATransition]) -> KDATransition:
    """Compose a non-empty ordered sequence of transitions."""
    if not transitions:
        raise ValueError("cannot compose an empty transition sequence")
    result = transitions[0]
    for transition in transitions[1:]:
        result = compose_transitions(result, transition)
    return result


def stack_transitions(transitions: Sequence[KDATransition]) -> KDATransition:
    """Stack equal-shaped per-segment transitions along a new leading axis."""
    if not transitions:
        raise ValueError("cannot stack an empty transition sequence")
    first = transitions[0]
    _check_transition(first)
    for transition in transitions[1:]:
        _check_transition(transition)
        if transition.matrix.shape != first.matrix.shape or transition.bias.shape != first.bias.shape:
            raise ValueError("transitions must have identical shapes")
        if transition.matrix.device != first.matrix.device or transition.matrix.dtype != first.matrix.dtype:
            raise ValueError("transitions must share device and dtype")
    return KDATransition(
        matrix=torch.stack([transition.matrix for transition in transitions]),
        bias=torch.stack([transition.bias for transition in transitions]),
    )


def exclusive_prefix_transitions(
    transitions: Sequence[KDATransition],
) -> tuple[KDATransition, ...]:
    """Serial reference for the start transition of every CP segment."""
    if not transitions:
        return ()
    first = transitions[0]
    _check_transition(first)
    prefix = identity_transition(
        first.matrix.shape[-3],
        first.bias.shape[-2],
        first.matrix.shape[-1],
        device=first.matrix.device,
        dtype=first.matrix.dtype,
    )
    result = []
    for transition in transitions:
        _check_transition(transition)
        result.append(prefix)
        prefix = compose_transitions(prefix, transition)
    return tuple(result)


def exclusive_prefix_scan(transitions: Sequence[KDATransition]) -> KDATransition:
    """GPU-resident O(log P) exclusive scan over P segment transitions.

    This Hillis-Steele reference launches batched matrix multiplications at each
    tree level and supports any positive segment count, not only powers of two.
    A fused implementation can retain the same composition law.
    """
    stacked = stack_transitions(transitions)
    matrix, bias = stacked.matrix, stacked.bias
    segment_count = matrix.shape[0]
    stride = 1
    while stride < segment_count:
        right_matrix = matrix[stride:]
        matrix = torch.cat(
            (matrix[:stride], matrix[:-stride] @ right_matrix), dim=0
        )
        bias = torch.cat(
            (bias[:stride], bias[:-stride] @ right_matrix + bias[stride:]), dim=0
        )
        stride *= 2

    identity = identity_transition(
        matrix.shape[-3],
        bias.shape[-2],
        matrix.shape[-1],
        device=matrix.device,
        dtype=matrix.dtype,
    )
    return KDATransition(
        matrix=torch.cat((identity.matrix.unsqueeze(0), matrix[:-1]), dim=0),
        bias=torch.cat((identity.bias.unsqueeze(0), bias[:-1]), dim=0),
    )


def segment_start_states(
    initial_state: torch.Tensor,
    transitions: Sequence[KDATransition],
) -> torch.Tensor:
    """Return [P, H, V, K] initial states for all P CP segments."""
    return apply_transition(initial_state, exclusive_prefix_scan(transitions))


def distributed_exclusive_prefix_transition(
    local_transition: KDATransition,
    group=None,
) -> KDATransition:
    """Compute this rank's exclusive CP prefix with a NCCL all-gather.

    This correctness baseline communicates one dense segment summary per rank,
    then runs the GPU-resident tree scan locally. It has O(P) communication and
    memory per rank; a production implementation should replace the all-gather
    with a fused distributed tree while retaining the same composition law.
    """
    import torch.distributed as dist

    _check_transition(local_transition)
    if local_transition.matrix.ndim != 3 or local_transition.bias.ndim != 3:
        raise ValueError("each rank must contribute one unbatched transition")
    if not local_transition.matrix.is_cuda:
        raise ValueError("distributed CP transitions must reside on CUDA")
    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError("torch.distributed must be initialized")

    rank = dist.get_rank(group)
    world_size = dist.get_world_size(group)
    matrix = local_transition.matrix.contiguous()
    bias = local_transition.bias.contiguous()
    matrix_elements = matrix.numel()
    payload = torch.cat((matrix.reshape(-1), bias.reshape(-1)))
    gathered = torch.empty(
        world_size * payload.numel(), device=payload.device, dtype=payload.dtype
    )
    all_gather = getattr(dist, "all_gather_single", dist.all_gather_into_tensor)
    all_gather(gathered, payload, group=group)
    gathered = gathered.view(world_size, -1)
    transitions = [
        KDATransition(
            matrix=gathered[index, :matrix_elements].view_as(matrix),
            bias=gathered[index, matrix_elements:].view_as(bias),
        )
        for index in range(world_size)
    ]
    prefixes = exclusive_prefix_scan(transitions)
    return KDATransition(matrix=prefixes.matrix[rank], bias=prefixes.bias[rank])


def _chunk_factors(
    key: torch.Tensor,
    log_decay: torch.Tensor,
    beta: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build the chunk-local KDA factors in a high-precision reference space."""
    if key.ndim != 3:
        raise ValueError("key and log_decay must have shape [T, H, K]")
    if log_decay.shape != key.shape:
        raise ValueError("log_decay must have the same shape as key")
    if beta.shape != key.shape[:2]:
        raise ValueError("beta must have shape [T, H]")
    if not key.is_floating_point() or not log_decay.is_floating_point() or not beta.is_floating_point():
        raise ValueError("key, log_decay, and beta must be floating point")
    if key.device != log_decay.device or key.device != beta.device:
        raise ValueError("key, log_decay, and beta must share a device")
    if key.shape[0] == 0 or key.shape[0] > KDA_CHUNK_SIZE:
        raise ValueError(f"chunk length must be in [1, {KDA_CHUNK_SIZE}]")

    dtype = torch.promote_types(torch.promote_types(key.dtype, log_decay.dtype), beta.dtype)
    if dtype in (torch.float16, torch.bfloat16):
        dtype = torch.float32
    key_h = key.transpose(0, 1).to(dtype)
    decay_h = log_decay.transpose(0, 1).to(dtype)
    beta_h = beta.transpose(0, 1).to(dtype)
    cumulative_decay = torch.cumsum(decay_h, dim=1)
    decay_prefix = torch.exp2(cumulative_decay)
    key_decayed = key_h * decay_prefix
    key_inverse = key_h * torch.exp2(-cumulative_decay)
    total_decay = decay_prefix[:, -1, :]
    key_restored = key_inverse * total_decay.unsqueeze(1)

    gram = key_decayed @ key_inverse.transpose(-1, -2)
    lower = torch.tril(gram, diagonal=-1) * beta_h.unsqueeze(-1)
    chunk_len = key.shape[0]
    eye = torch.eye(chunk_len, dtype=dtype, device=key.device).expand(key.shape[1], -1, -1)
    inverse = eye.clone()
    power = eye
    for _ in range(1, chunk_len):
        power = power @ lower
        inverse = inverse + power
    return key_decayed, key_restored, total_decay, inverse, beta_h


def kda_chunk_transition(
    key: torch.Tensor,
    value: torch.Tensor,
    log_decay: torch.Tensor,
    beta: torch.Tensor,
) -> KDATransition:
    """Return the exact affine state map for one KDA chunk (T <= 16)."""
    if value.ndim != 3 or value.shape[:2] != key.shape[:2]:
        raise ValueError("value must have shape [T, H, V] matching key")
    if value.device != key.device or not value.is_floating_point():
        raise ValueError("value must be floating point and share key's device")

    key_decayed, key_restored, total_decay, inverse, beta_h = _chunk_factors(
        key, log_decay, beta
    )
    value_h = value.transpose(0, 1).to(inverse.dtype)
    # P = (I - L)^-1 diag(beta), which multiplies the chunk residual.
    residual_operator = inverse * beta_h.unsqueeze(-2)
    right_factor = residual_operator.transpose(-1, -2) @ key_restored
    return KDATransition(
        matrix=torch.diag_embed(total_decay) - key_decayed.transpose(-1, -2) @ right_factor,
        bias=value_h.transpose(-1, -2) @ right_factor,
    )


def _segment_factors_batched(
    key: torch.Tensor,
    value: torch.Tensor,
    log_decay: torch.Tensor,
    beta: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build all rank-16 chunk factors with a fixed number of GPU launches."""
    tokens, heads, key_dim = key.shape
    value_dim = value.shape[-1]
    padding = (-tokens) % KDA_CHUNK_SIZE
    if padding:
        key = F.pad(key, (0, 0, 0, 0, 0, padding))
        value = F.pad(value, (0, 0, 0, 0, 0, padding))
        log_decay = F.pad(log_decay, (0, 0, 0, 0, 0, padding))
        beta = F.pad(beta, (0, 0, 0, padding))

    chunks = key.shape[0] // KDA_CHUNK_SIZE
    dtype = torch.promote_types(
        torch.promote_types(key.dtype, log_decay.dtype), beta.dtype
    )
    if dtype in (torch.float16, torch.bfloat16):
        dtype = torch.float32
    key_h = (
        key.reshape(chunks, KDA_CHUNK_SIZE, heads, key_dim)
        .permute(0, 2, 1, 3)
        .to(dtype)
    )
    value_h = (
        value.reshape(chunks, KDA_CHUNK_SIZE, heads, value_dim)
        .permute(0, 2, 1, 3)
        .to(dtype)
    )
    decay_h = (
        log_decay.reshape(chunks, KDA_CHUNK_SIZE, heads, key_dim)
        .permute(0, 2, 1, 3)
        .to(dtype)
    )
    beta_h = (
        beta.reshape(chunks, KDA_CHUNK_SIZE, heads)
        .permute(0, 2, 1)
        .to(dtype)
    )
    cumulative_decay = torch.cumsum(decay_h, dim=2)
    decay_prefix = torch.exp2(cumulative_decay)
    key_decayed = key_h * decay_prefix
    key_inverse = key_h * torch.exp2(-cumulative_decay)
    total_decay = decay_prefix[:, :, -1, :]
    key_restored = key_inverse * total_decay.unsqueeze(2)

    gram = key_decayed @ key_inverse.transpose(-1, -2)
    lower = torch.tril(gram, diagonal=-1) * beta_h.unsqueeze(-1)
    eye = torch.eye(
        KDA_CHUNK_SIZE, dtype=dtype, device=key.device
    ).view(1, 1, KDA_CHUNK_SIZE, KDA_CHUNK_SIZE)
    inverse = eye.expand(chunks, heads, -1, -1).clone()
    power = eye.expand(chunks, heads, -1, -1)
    for _ in range(1, KDA_CHUNK_SIZE):
        power = power @ lower
        inverse = inverse + power
    residual_operator = inverse * beta_h.unsqueeze(-2)
    right_factor = residual_operator.transpose(-1, -2) @ key_restored
    return (
        key_decayed.contiguous(),
        right_factor.contiguous(),
        total_decay.contiguous(),
        value_h.contiguous(),
    )


def _validate_segment_inputs(
    key: torch.Tensor,
    value: torch.Tensor,
    log_decay: torch.Tensor,
    beta: torch.Tensor,
) -> None:
    if key.ndim != 3 or key.shape[0] == 0:
        raise ValueError("key must have non-empty shape [T, H, K]")
    if value.ndim != 3 or value.shape[:2] != key.shape[:2]:
        raise ValueError("value must have shape [T, H, V] matching key")
    if log_decay.shape != key.shape or beta.shape != key.shape[:2]:
        raise ValueError("log_decay and beta shapes must match key")
    if value.device != key.device or log_decay.device != key.device or beta.device != key.device:
        raise ValueError("all KDA inputs must share a device")
    if not all(
        tensor.is_floating_point() for tensor in (key, value, log_decay, beta)
    ):
        raise ValueError("all KDA inputs must be floating point")


def kda_segment_transition_reference(
    key: torch.Tensor,
    value: torch.Tensor,
    log_decay: torch.Tensor,
    beta: torch.Tensor,
) -> KDATransition:
    """PyTorch reference using each chunk's rank-at-most-16 update.

    Unlike composing a dense transition after every chunk, this costs
    O((K + V) * K * C) per chunk for C <= 16 instead of O(K^3 + V*K^2).
    The final dense (A, B) pair is the segment summary communicated by CP.
    """
    _validate_segment_inputs(key, value, log_decay, beta)

    first_end = min(KDA_CHUNK_SIZE, key.shape[0])
    first_factors = _chunk_factors(
        key[:first_end], log_decay[:first_end], beta[:first_end]
    )
    dtype = first_factors[3].dtype
    transition = identity_transition(
        key.shape[1], value.shape[2], key.shape[2], device=key.device, dtype=dtype
    )
    matrix, bias = transition.matrix, transition.bias

    for start in range(0, key.shape[0], KDA_CHUNK_SIZE):
        end = min(start + KDA_CHUNK_SIZE, key.shape[0])
        key_decayed, key_restored, total_decay, inverse, beta_h = _chunk_factors(
            key[start:end], log_decay[start:end], beta[start:end]
        )
        value_h = value[start:end].transpose(0, 1).to(dtype)
        residual_operator = inverse * beta_h.unsqueeze(-2)
        right_factor = residual_operator.transpose(-1, -2) @ key_restored
        key_decayed_t = key_decayed.transpose(-1, -2)

        # Post-multiply the accumulated dense transition by
        # diag(total_decay) - key_decayed.T @ right_factor without forming the
        # chunk's dense KxK matrix first.
        matrix = (
            matrix * total_decay.unsqueeze(-2)
            - (matrix @ key_decayed_t) @ right_factor
        )
        bias = (
            bias * total_decay.unsqueeze(-2)
            - (bias @ key_decayed_t) @ right_factor
            + value_h.transpose(-1, -2) @ right_factor
        )
    return KDATransition(matrix=matrix, bias=bias)


def kda_segment_transition_triton(
    key: torch.Tensor,
    value: torch.Tensor,
    log_decay: torch.Tensor,
    beta: torch.Tensor,
) -> KDATransition:
    """Fused CUDA D=V=128 segment transition with a register-resident row."""
    _validate_segment_inputs(key, value, log_decay, beta)
    if triton is None:
        raise RuntimeError("Triton is required for the fused segment transition")
    if not key.is_cuda:
        raise ValueError("the fused segment transition requires CUDA tensors")
    if key.shape[-1] != 128 or value.shape[-1] != 128:
        raise ValueError("the fused segment transition currently requires K = V = 128")

    key_decayed, right_factor, total_decay, value_h = _segment_factors_batched(
        key, value, log_decay, beta
    )
    chunks, heads, _, key_dim = key_decayed.shape
    value_dim = value_h.shape[-1]
    matrix = torch.empty(
        (heads, key_dim, key_dim), device=key.device, dtype=torch.float32
    )
    bias = torch.empty(
        (heads, value_dim, key_dim), device=key.device, dtype=torch.float32
    )
    _kda_segment_accumulate_kernel[(heads * (key_dim + value_dim),)](
        key_decayed,
        right_factor,
        total_decay,
        value_h,
        matrix,
        bias,
        chunks,
        HEADS=heads,
        KEY_DIM=key_dim,
        VALUE_DIM=value_dim,
        CHUNK=KDA_CHUNK_SIZE,
        BLOCK_K=triton.next_power_of_2(key_dim),
        num_warps=4,
    )
    return KDATransition(matrix=matrix, bias=bias)


def kda_segment_transition(
    key: torch.Tensor,
    value: torch.Tensor,
    log_decay: torch.Tensor,
    beta: torch.Tensor,
) -> KDATransition:
    """Return a segment summary, dispatching the fused CUDA D=128 path."""
    if (
        triton is not None
        and key.is_cuda
        and key.ndim == 3
        and value.ndim == 3
        and key.shape[-1] == 128
        and value.shape[-1] == 128
    ):
        return kda_segment_transition_triton(key, value, log_decay, beta)
    return kda_segment_transition_reference(key, value, log_decay, beta)


def kda_chunk_update(
    state: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    log_decay: torch.Tensor,
    beta: torch.Tensor,
) -> torch.Tensor:
    """Direct recurrence for checking kda_chunk_transition independently."""
    key_decayed, key_restored, total_decay, inverse, beta_h = _chunk_factors(
        key, log_decay, beta
    )
    if value.ndim != 3 or value.shape[:2] != key.shape[:2]:
        raise ValueError("value must have shape [T, H, V] matching key")
    if state.shape != (key.shape[1], value.shape[2], key.shape[2]):
        raise ValueError("state must have shape [H, V, K] compatible with inputs")

    state_h = state.to(inverse.dtype)
    value_h = value.transpose(0, 1).to(inverse.dtype)
    residual = value_h - key_decayed @ state_h.transpose(-1, -2)
    update = key_restored.transpose(-1, -2) @ (
        inverse @ (residual * beta_h.unsqueeze(-1))
    )
    return (update + total_decay.unsqueeze(-1) * state_h.transpose(-1, -2)).transpose(-1, -2)
