"""Pure-PyTorch reference for the FlashKDA forward formulation."""

from __future__ import annotations

import math

import torch
from torch.nn import functional as F


LOG2E = 1.4426950408889634


def _forward_substitute(lower: torch.Tensor) -> torch.Tensor:
    """Return ``(I + lower)^-1`` for a strictly lower-triangular matrix."""
    size = lower.shape[-1]
    inverse = torch.eye(size, dtype=lower.dtype, device=lower.device)
    for row in range(1, size):
        for col in range(row):
            inverse[..., row, :] -= lower[..., row, col] * inverse[..., col, :]
    return inverse


def _chunk_cumsum(gate: torch.Tensor, chunk_size: int) -> torch.Tensor:
    batch, steps, heads, dim = gate.shape
    chunks = steps // chunk_size
    gated = gate.view(batch, chunks, chunk_size, heads, dim)
    return gated.cumsum(dim=2).view(batch, steps, heads, dim)


def flashkda_reference_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    lower_bound: float,
    chunk_size: int = 16,
    initial_state: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the chunked KDA forward in FlashKDA's state-first layout.

    ``q/k/v`` and ``g`` are ``[B, T, H, D]``, while ``beta`` is pre-activation
    ``[B, T, H]``. The returned state is fp32 ``[B, H, V, K]``.
    """
    if q.ndim != 4 or q.shape != k.shape or q.shape != v.shape:
        raise ValueError("q/k/v must have identical [B, T, H, D] shapes")
    if g.shape != q.shape or g.dtype != torch.float32:
        raise ValueError("g must be FP32 [B, T, H, D]")
    batch, steps, heads, dim = q.shape
    if steps % chunk_size:
        raise ValueError(f"T={steps} must be divisible by chunk_size={chunk_size}")
    if beta.shape != (batch, steps, heads) or beta.dtype != torch.float32:
        raise ValueError("beta must be FP32 [B, T, H]")
    if A_log.shape != (heads,) or A_log.dtype != torch.float32:
        raise ValueError("A_log must be FP32 [H]")
    if dt_bias.shape != (heads, dim) or dt_bias.dtype != torch.float32:
        raise ValueError("dt_bias must be FP32 [H, D]")
    if initial_state is not None and initial_state.shape != (batch, heads, dim, dim):
        raise ValueError("initial_state must be [B, H, D, D]")

    solve_dtype = torch.float32 if q.device.type == "npu" else torch.float64
    q_normalized = (q.float() / torch.sqrt(q.float().square().sum(-1, keepdim=True) + 1e-6)).to(q.dtype)
    k_normalized = (k.float() / torch.sqrt(k.float().square().sum(-1, keepdim=True) + 1e-6)).to(k.dtype)
    gate_input = g + dt_bias.view(1, 1, heads, dim)
    gate = lower_bound * torch.sigmoid(torch.exp(A_log).view(1, 1, heads, 1) * gate_input) * LOG2E
    gate_cumsum = _chunk_cumsum(gate, chunk_size)
    beta_sigmoid = torch.sigmoid(beta)

    state = (torch.zeros(batch, heads, dim, dim, dtype=torch.float32, device=q.device)
             if initial_state is None else initial_state.float().clone())
    output = torch.empty_like(q, dtype=torch.float32)

    for batch_index in range(batch):
        for head_index in range(heads):
            state_bh = state[batch_index, head_index]
            for chunk_index in range(steps // chunk_size):
                start = chunk_index * chunk_size
                stop = start + chunk_size
                q_chunk = q_normalized[batch_index, start:stop, head_index].float()
                k_chunk = k_normalized[batch_index, start:stop, head_index].float()
                v_chunk = v[batch_index, start:stop, head_index].float()
                g_chunk = gate_cumsum[batch_index, start:stop, head_index]
                beta_chunk = beta_sigmoid[batch_index, start:stop, head_index].float()
                g_middle = g_chunk[chunk_size // 2]
                g_recentred = g_chunk - g_middle

                q_decayed = q_chunk * torch.exp2(g_recentred)
                k_recentred = k_chunk * torch.exp2(g_recentred)
                k_inverse = k_chunk * torch.exp2(-g_recentred)
                aqk = torch.tril(q_decayed @ k_inverse.T * scale)
                akk = torch.tril(k_recentred @ k_inverse.T * beta_chunk[:, None], diagonal=-1)
                inverse = _forward_substitute(akk.to(solve_dtype)).float()

                k_weighted = k_chunk * beta_chunk[:, None] * torch.exp2(g_chunk)
                v_weighted = v_chunk * beta_chunk[:, None]
                w = inverse @ k_weighted
                u = inverse @ v_weighted

                g_last = g_chunk[-1]
                v_new = u - w @ state_bh.T
                output[batch_index, start:stop, head_index] = (
                    (q_chunk * torch.exp2(g_chunk)) @ state_bh.T * scale + aqk @ v_new
                )
                k_update = k_chunk * torch.exp2(g_last[None, :] - g_chunk)
                state_bh = state_bh * torch.exp2(g_last)[None, :] + v_new.T @ k_update
            state[batch_index, head_index] = state_bh

    return output, state

STATE_IO = [(True, True), (True, False), (False, True), (False, False)]
STATE_IDS = ["in+out", "in_only", "out_only", "no_state"]


def make_inputs(batch, steps, heads, device="cpu", dim=128, gate_dtype=torch.bfloat16):
    generator = torch.Generator().manual_seed(42)
    shape = (batch, steps, heads, dim)
    q, k = [F.normalize(torch.randn(shape, generator=generator), dim=-1).bfloat16()
            for _ in range(2)]
    v = torch.randn(shape, generator=generator).bfloat16()
    # Mild decay supplies stable ordinary cases alongside the extreme-gate tests.
    g = (torch.randn(shape, generator=generator) * 0.5 - 2).to(gate_dtype)
    beta = torch.randn(shape[:-1], generator=generator).bfloat16()
    a_log = torch.rand(heads, generator=generator)
    bias = torch.rand(heads, dim, generator=generator) * 0.25
    return tuple(x.to(device) for x in (q, k, v, g, beta, a_log, bias))


def make_state(sequences, heads, dim, dtype, device):
    generator = torch.Generator().manual_seed(123)
    return (torch.randn(sequences, heads, dim, dim, generator=generator) * 0.1).to(
        device=device, dtype=dtype)


def recurrent_reference(inputs, initial_state=None, seq_lens=None, lower_bound=-5.0,
                        scale=None):
    """Token recurrence in public [sequence, head, value, key] state layout."""
    q, k, v, g, beta, a_log, bias = [x.detach().cpu().double() for x in inputs]
    batch, steps, heads, dim = q.shape
    q, k = [x / (x.square().sum(-1, keepdim=True) + 1e-6).sqrt() for x in (q, k)]
    scale = dim ** -0.5 if scale is None else scale
    decay = torch.exp(lower_bound * torch.sigmoid(
        a_log.exp()[None, None, :, None] * (g + bias)))
    beta = beta.sigmoid()
    lengths = [steps] * batch if seq_lens is None else seq_lens
    state = (torch.zeros(len(lengths), heads, dim, dim, dtype=torch.float64)
             if initial_state is None else initial_state.detach().cpu().double().clone())
    output = torch.empty_like(v).reshape(-1, heads, dim)
    q, k, v, decay = [x.reshape(-1, heads, dim) for x in (q, k, v, decay)]
    beta = beta.reshape(-1, heads)
    start = 0
    for sequence, length in enumerate(lengths):
        s = state[sequence]
        for token in range(start, start + length):
            s = s * decay[token].unsqueeze(-2)
            residual = (v[token] - (s * k[token].unsqueeze(-2)).sum(-1)) * beta[token, :, None]
            s = s + residual.unsqueeze(-1) * k[token].unsqueeze(-2)
            output[token] = (s * q[token].unsqueeze(-2)).sum(-1) * scale
        state[sequence] = s
        start += length
    return output.reshape(batch, steps, heads, dim).float(), state.float()


def make_cuda_inputs(batch, steps, heads, device="cpu", seed=42):
    """CUDA input distributions/order, generated on CPU for reproducible fixtures.

    CPU and CUDA RNG implementations differ: this matches distributions and
    generation order, not the bytes generated by CUDA's random-number engine.
    """
    generator = torch.Generator().manual_seed(seed)
    shape = (batch, steps, heads, 128)
    q, k = [F.normalize(torch.randn(shape, generator=generator), dim=-1).bfloat16()
            for _ in range(2)]
    v, g = [torch.randn(shape, dtype=torch.bfloat16, generator=generator) for _ in range(2)]
    beta = torch.randn(shape[:-1], dtype=torch.bfloat16, generator=generator)
    a_log = torch.rand(heads, generator=generator)
    bias = torch.rand(heads, 128, generator=generator)
    return tuple(x.to(device) for x in (q, k, v, g, beta, a_log, bias))


def make_cuda_state(sequences, heads, dtype, device="cpu"):
    """Match CUDA's arange -> BF16 -> requested state dtype, including FP32."""
    return torch.arange(sequences * heads * 128 * 128, dtype=torch.float32).reshape(
        sequences, heads, 128, 128).bfloat16().to(device=device, dtype=dtype)


CUDA_GATE_CASE_IDS = [
    "g=-8_bias=0", "g=-4_bias=0", "g=0_bias=0", "g=8_bias=0",
    "g=U(-8,8)_bias=0", "g=N(0,8)_bias=0", "g=0_bias={-4,4}",
    "g=0_bias={-8,8}", "g=0_bias=U(-8,8)", "g=0_bias=N(0,8)",
]


def make_cuda_gate_case(case, seq_lens=None):
    """CUDA's 8192-token gate/bias distributions, using CPU seed 42.

    Reference and kernel receive the same BF16-rounded initial state. The CUDA
    FLA comparison instead gives FLA the original FP32 state; avoid conflating
    input quantization with backend error in this independent oracle check.
    """
    if case not in range(len(CUDA_GATE_CASE_IDS)):
        raise ValueError("unknown CUDA gate case")
    steps = 8192 if seq_lens is None else sum(seq_lens)
    sequences = 1 if seq_lens is None else len(seq_lens)
    generator = torch.Generator().manual_seed(42)
    shape = (1, steps, 1, 128)
    q, k, v = [torch.rand(shape, dtype=torch.bfloat16, generator=generator)
               for _ in range(3)]
    beta = torch.randn(shape[:-1], dtype=torch.bfloat16, generator=generator)
    initial = torch.randn(sequences, 1, 128, 128, generator=generator).bfloat16()
    bias_cases = [
        torch.where(torch.rand(1, 128, generator=generator) < 0.5, -4.0, 4.0),
        torch.where(torch.rand(1, 128, generator=generator) < 0.5, -8.0, 8.0),
        torch.empty(1, 128).uniform_(-8, 8, generator=generator),
        torch.randn(1, 128, generator=generator) * 8,
    ]
    if case < 4:
        g = torch.full(shape, [-8.0, -4.0, 0.0, 8.0][case], dtype=torch.bfloat16)
    elif case == 4:
        g = torch.empty(shape, dtype=torch.bfloat16).uniform_(-8, 8, generator=generator)
    elif case == 5:
        g = torch.randn(shape, dtype=torch.bfloat16, generator=generator) * 8
    else:
        g = torch.zeros(shape, dtype=torch.bfloat16)
    bias = torch.zeros(1, 128) if case < 6 else bias_cases[case - 6]
    return (q, k, v, g, beta, torch.zeros(1), bias), initial


def chunked_reference(inputs, initial_state=None, seq_lens=None, lower_bound=-5.0,
                      scale=None, chunk_size=16):
    """CPU FP64 closed-form recurrence, with no backend-specific rounding.

    Use short independent reference chunks to keep the full CUDA-size matrix
    practical. Unit tests check this algebra against the token recurrence,
    including different chunk sizes, tiny norms, empty sequences and scale.
    """
    q, k, v, g, beta, a_log, bias = [x.detach().cpu() for x in inputs]
    batch, steps, heads, dim = q.shape
    scale = dim ** -0.5 if scale is None else scale
    lengths = [steps] * batch if seq_lens is None else seq_lens
    if chunk_size < 1 or sum(lengths) != batch * steps or any(n < 0 for n in lengths):
        raise ValueError("invalid reference chunk size or sequence lengths")
    state = (torch.zeros(len(lengths), heads, dim, dim, dtype=torch.float64)
             if initial_state is None else initial_state.detach().cpu().double().clone())
    output = torch.empty((batch * steps, heads, dim), dtype=torch.float32)
    q, k, v, g = [x.reshape(-1, heads, dim) for x in (q, k, v, g)]
    beta = beta.reshape(-1, heads)
    a_exp, bias = a_log.double().exp()[:, None, None], bias.double()[:, None, :]
    start = 0
    for sequence, length in enumerate(lengths):
        s = state[sequence]
        for lo in range(start, start + length, chunk_size):
            hi = min(start + length, lo + chunk_size)
            qc, kc, vc, gc = [x[lo:hi].double().transpose(0, 1) for x in (q, k, v, g)]
            qc, kc = [x / (x.square().sum(-1, keepdim=True) + 1e-6).sqrt() for x in (qc, kc)]
            bc = beta[lo:hi].double().transpose(0, 1).sigmoid().unsqueeze(-1)
            cumulative = (lower_bound * torch.sigmoid(a_exp * (gc + bias))).cumsum(1)
            reference = cumulative[:, (hi - lo) // 2:(hi - lo) // 2 + 1]
            recentred = (cumulative - reference).exp()
            inverse_decay_k = kc / recentred
            lower = torch.tril((kc * recentred) @ inverse_decay_k.transpose(-1, -2) * bc,
                               diagonal=-1)
            identity = torch.eye(hi - lo, dtype=torch.float64)
            rhs = torch.cat((kc * cumulative.exp() * bc, vc * bc), dim=-1)
            solved = torch.linalg.solve_triangular(lower + identity, rhs,
                                                   upper=False, unitriangular=True)
            w, u = solved.split(dim, dim=-1)
            residual = u - w @ s.transpose(-1, -2)
            aqk = torch.tril((qc * recentred) @ inverse_decay_k.transpose(-1, -2))
            out = ((qc * cumulative.exp()) @ s.transpose(-1, -2) + aqk @ residual) * scale
            output[lo:hi] = out.transpose(0, 1).float()
            s = (s * cumulative[:, -1:, :].exp()
                 + residual.transpose(-1, -2) @ (kc * (cumulative[:, -1:, :] - cumulative).exp()))
        state[sequence] = s
        start += length
    return output.reshape(batch, steps, heads, dim), state.float()


def relative_rms(actual, expected):
    actual, expected = actual.detach().cpu().double(), expected.detach().cpu().double()
    assert torch.isfinite(actual).all(), "nonfinite actual output/state"
    assert torch.isfinite(expected).all(), "nonfinite reference output/state"
    return ((actual - expected).square().mean().sqrt()
            / expected.square().mean().sqrt().clamp_min(1e-8)).item()


def assert_cuda_accuracy(actual, expected, label, record_property, threshold=0.005,
                         check_windows=True):
    """Public FP64 accuracy criterion; retain metrics for the validation report.

    Check output windows separately so a large arange initial state cannot
    dominate the global error and hide later failures. This does not replace
    the existing pointwise/native-equivalence assertions in test_fwd.py.
    """
    actual = actual.detach().cpu().float()
    ratio = relative_rms(actual, expected)
    record_property(label + "_relative_rms", ratio)
    assert ratio < threshold, f"{label} relative RMS {ratio:.6g} >= {threshold}"
    if label == "output":
        maximum = max(relative_rms(actual[:, lo:lo + 1024], expected[:, lo:lo + 1024])
                      for lo in range(0, actual.shape[1], 1024))
        record_property("output_max_window_relative_rms", maximum)
        if check_windows:
            assert maximum < threshold, f"output window relative RMS {maximum:.6g} >= {threshold}"


def assert_post_transient_accuracy(actual, expected, lengths, record_property):
    """Check the steady region separately from large initial-state transients.

    BF16 Cube operands and result handoffs can amplify cancellation in the
    arange-state transient. Allow 1% there, but keep 0.5% after the first 64
    tokens of each sequence, irrespective of the configured native chunk size.
    """
    actual = actual.detach().cpu().float().reshape(-1, *actual.shape[2:])
    expected = expected.reshape_as(actual)
    ratios = []
    start = 0
    for length in lengths:
        for lo in range(start + 64, start + length, 1024):
            hi = min(lo + 1024, start + length)
            ratios.append(relative_rms(actual[lo:hi], expected[lo:hi]))
        start += length
    if ratios:
        maximum = max(ratios)
        record_property("output_post_transient_max_relative_rms", maximum)
        assert maximum < 0.005, f"post-transient relative RMS {maximum:.6g} >= 0.005"


def assert_close(actual, expected):
    torch.testing.assert_close(actual.detach().cpu().float(), expected.float(),
                               rtol=5e-3, atol=5e-3)


def assert_relative_rms(actual, expected):
    """CUDA gate-sweep criterion: relative RMS error below 0.5%."""
    actual, expected = actual.detach().cpu().float(), expected.float()
    assert torch.isfinite(actual).all(), "nonfinite output/state"
    ratio = (actual - expected).square().mean().sqrt() / expected.square().mean().sqrt().clamp_min(1e-8)
    assert ratio.item() < 5e-3, f"relative RMS error {ratio.item():.6g} >= 0.005"
