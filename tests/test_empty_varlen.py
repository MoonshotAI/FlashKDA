"""Regression tests for zero-length sequences in variable-length batches."""

import math

import pytest
import torch
import torch.nn.functional as F

import flash_kda
from torch_ref import torch_ref


D = 128
LOWER_BOUND = -5.0


def _make_inputs(total_tokens, heads):
    torch.manual_seed(42)
    q = F.normalize(
        torch.randn((1, total_tokens, heads, D), dtype=torch.float32, device="cuda"),
        p=2,
        dim=-1,
    ).to(torch.bfloat16)
    k = F.normalize(
        torch.randn((1, total_tokens, heads, D), dtype=torch.float32, device="cuda"),
        p=2,
        dim=-1,
    ).to(torch.bfloat16)
    v = torch.randn((1, total_tokens, heads, D), dtype=torch.bfloat16, device="cuda")
    g = torch.randn((1, total_tokens, heads, D), dtype=torch.bfloat16, device="cuda")
    beta = torch.randn((1, total_tokens, heads), dtype=torch.bfloat16, device="cuda")
    a_log = torch.rand(heads, dtype=torch.float32, device="cuda")
    dt_bias = torch.rand(heads, D, dtype=torch.float32, device="cuda")
    return q, k, v, g, beta, a_log, dt_bias, 1.0 / math.sqrt(D)


def _make_state(shape, dtype):
    return torch.arange(
        math.prod(shape), dtype=torch.float32, device="cuda"
    ).reshape(shape).to(torch.bfloat16).to(dtype)


@pytest.mark.parametrize(
    "seq_lens",
    [
        [0, 17],
        [17, 0, 33],
        [0, 0, 17, 0],
        [0, 0],
    ],
    ids=["leading_empty", "middle_empty", "multiple_empty", "all_empty"],
)
@pytest.mark.parametrize("heads", [1, 4])
@pytest.mark.parametrize("state_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("has_initial_state", [False, True])
def test_fwd_varlen_with_empty_sequences(
    seq_lens, heads, state_dtype, has_initial_state
):
    """Empty sequences have no output tokens and preserve their input state."""
    total_tokens = sum(seq_lens)
    sequence_count = len(seq_lens)
    cu_seqlens = torch.tensor(
        [0, *torch.tensor(seq_lens).cumsum(0).tolist()],
        dtype=torch.long,
        device="cuda",
    )
    q, k, v, g, beta, a_log, dt_bias, scale = _make_inputs(total_tokens, heads)

    initial_kernel = (
        _make_state((sequence_count, heads, D, D), state_dtype) if has_initial_state else None
    )
    initial_reference = initial_kernel.clone() if initial_kernel is not None else None
    final_kernel = torch.zeros(
        sequence_count, heads, D, D, dtype=state_dtype, device="cuda"
    )
    final_reference = torch.zeros_like(final_kernel)
    output_kernel = torch.empty_like(q)
    output_reference = torch.empty_like(q)

    flash_kda.fwd(
        q,
        k,
        v,
        g,
        beta,
        scale,
        output_kernel,
        A_log=a_log,
        dt_bias=dt_bias,
        lower_bound=LOWER_BOUND,
        initial_state=initial_kernel,
        final_state=final_kernel,
        cu_seqlens=cu_seqlens,
    )
    torch.cuda.synchronize()
    torch_ref(
        q,
        k,
        v,
        g,
        beta,
        scale,
        output_reference,
        A_log=a_log,
        dt_bias=dt_bias,
        lower_bound=LOWER_BOUND,
        initial_state=initial_reference,
        final_state=final_reference,
        cu_seqlens=cu_seqlens,
    )

    assert torch.equal(output_kernel, output_reference)
    assert torch.equal(final_kernel, final_reference)

