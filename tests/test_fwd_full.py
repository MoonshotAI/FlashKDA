"""Full correctness tests for FlashKDA forward pass.

Compares cutlass kernel vs torch_ref for exact match across:
  - state_in / state_out: 4 combinations of None/present
  - state dtype: bf16, fp32
  - various H values (up to 256)
  - various sequence lengths (up to 1M, fixed-length and varlen)
  - D=128 (fixed)

Run with: pytest tests/test_fwd_full.py -x -v
Parallel:  pytest tests/test_fwd_full.py -x -v -n auto  (requires pytest-xdist)
"""

import torch
import torch.nn.functional as F
import math
import pytest

import flash_kda
from torch_ref import torch_ref

D = 128
LOWER_BOUND = -5.0


def _make_inputs(T, H, device="cuda"):
    torch.manual_seed(42)
    q = F.normalize(torch.randn((1, T, H, D), dtype=torch.float32, device=device), p=2, dim=-1).to(torch.bfloat16)
    k = F.normalize(torch.randn((1, T, H, D), dtype=torch.float32, device=device), p=2, dim=-1).to(torch.bfloat16)
    v = torch.randn((1, T, H, D), dtype=torch.bfloat16, device=device)
    g = torch.randn((1, T, H, D), dtype=torch.bfloat16, device=device)
    beta = torch.randn((1, T, H), dtype=torch.bfloat16, device=device)
    A_log = torch.rand(H, dtype=torch.float32, device=device)
    dt_bias = torch.rand(H, D, dtype=torch.float32, device=device)
    scale = 1.0 / math.sqrt(D)
    return q, k, v, g, beta, A_log, dt_bias, scale


def _make_state(shape, dtype):
    """Create a state tensor with the given shape and dtype."""
    n_elems = 1
    for s in shape:
        n_elems *= s
    return torch.arange(n_elems, dtype=torch.float32, device="cuda").reshape(shape).to(torch.bfloat16).to(dtype)


# state_in x state_out combinations: (has_in, has_out)
STATE_IO = [
    (True, True),
    (True, False),
    (False, True),
    (False, False),
]
STATE_IO_IDS = ["in+out", "in_only", "out_only", "no_state"]

STATE_DTYPES = ["bf16", "fp32"]

H_VALUES = [1, 4, 32, 96]
T_VALUES = [16, 64, 256, 1024, 4096, 8192, 17, 37, 97]


# ---------------------------------------------------------------------------
# Fixed-length tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("H", H_VALUES, ids=[f"H{h}" for h in H_VALUES])
@pytest.mark.parametrize("T", T_VALUES, ids=[f"T{t}" for t in T_VALUES])
@pytest.mark.parametrize("state_dtype", STATE_DTYPES)
@pytest.mark.parametrize("has_in,has_out", STATE_IO, ids=STATE_IO_IDS)
def test_fwd_fixed(T, H, state_dtype, has_in, has_out):
    q, k, v, g, beta, A_log, dt_bias, scale = _make_inputs(T, H)
    dtype = torch.bfloat16 if state_dtype == "bf16" else torch.float32

    init_k = _make_state((1, H, D, D), dtype).clone() if has_in else None
    init_r = _make_state((1, H, D, D), dtype).clone() if has_in else None
    final_k = torch.zeros(1, H, D, D, dtype=dtype, device="cuda") if has_out else None
    final_r = torch.zeros(1, H, D, D, dtype=dtype, device="cuda") if has_out else None

    out_kernel = torch.zeros_like(q)
    flash_kda.fwd(q, k, v, g, beta, scale, out_kernel,
                  A_log=A_log, dt_bias=dt_bias, lower_bound=LOWER_BOUND,
                  initial_state=init_k, final_state=final_k)
    torch.cuda.synchronize()

    out_ref = torch.zeros_like(q)
    torch_ref(q, k, v, g, beta, scale, out_ref,
              A_log=A_log, dt_bias=dt_bias, lower_bound=LOWER_BOUND,
              initial_state=init_r, final_state=final_r)

    assert torch.equal(out_kernel, out_ref), \
        f"output mismatch: T={T} H={H} dtype={state_dtype} in={has_in} out={has_out}"
    if final_k is not None:
        assert torch.equal(final_k, final_r), \
            f"final_state mismatch: T={T} H={H} dtype={state_dtype} in={has_in} out={has_out}"


# ---------------------------------------------------------------------------
# Variable-length tests
# ---------------------------------------------------------------------------

VARLEN_CASES = [
    [16],
    [16, 16],
    [4, 8, 12],
    [64, 128, 256],
    [1300, 547, 2048, 963, 271, 3063],
    [1024] * 8,
    [20, 50, 100],
    [17, 33, 65],
]
VARLEN_H = [1, 4, 96]


@pytest.mark.parametrize("H", VARLEN_H, ids=[f"H{h}" for h in VARLEN_H])
@pytest.mark.parametrize("seq_lens", VARLEN_CASES,
                         ids=[f"seqs{'_'.join(str(s) for s in sl)}" for sl in VARLEN_CASES])
@pytest.mark.parametrize("state_dtype", STATE_DTYPES)
@pytest.mark.parametrize("has_in,has_out", STATE_IO, ids=STATE_IO_IDS)
def test_fwd_varlen(seq_lens, H, state_dtype, has_in, has_out):
    T_total = sum(seq_lens)
    N = len(seq_lens)
    cu_seqlens = torch.tensor(
        [0] + list(torch.cumsum(torch.tensor(seq_lens), dim=0).tolist()),
        dtype=torch.long, device="cuda",
    )

    q, k, v, g, beta, A_log, dt_bias, scale = _make_inputs(T_total, H)
    dtype = torch.bfloat16 if state_dtype == "bf16" else torch.float32

    init_k = _make_state((N, H, D, D), dtype).clone() if has_in else None
    init_r = _make_state((N, H, D, D), dtype).clone() if has_in else None
    final_k = torch.zeros(N, H, D, D, dtype=dtype, device="cuda") if has_out else None
    final_r = torch.zeros(N, H, D, D, dtype=dtype, device="cuda") if has_out else None

    out_kernel = torch.zeros_like(q)
    flash_kda.fwd(q, k, v, g, beta, scale, out_kernel,
                  A_log=A_log, dt_bias=dt_bias, lower_bound=LOWER_BOUND,
                  initial_state=init_k, final_state=final_k, cu_seqlens=cu_seqlens)
    torch.cuda.synchronize()

    out_ref = torch.zeros_like(q)
    torch_ref(q, k, v, g, beta, scale, out_ref,
              A_log=A_log, dt_bias=dt_bias, lower_bound=LOWER_BOUND,
              initial_state=init_r, final_state=final_r, cu_seqlens=cu_seqlens)

    assert torch.equal(out_kernel, out_ref), \
        f"output mismatch: seqs={seq_lens} H={H} dtype={state_dtype} in={has_in} out={has_out}"
    if final_k is not None:
        assert torch.equal(final_k, final_r), \
            f"final_state mismatch: seqs={seq_lens} H={H} dtype={state_dtype} in={has_in} out={has_out}"


# ---------------------------------------------------------------------------
# Batched tests (B > 1, equal-length sequences)
# ---------------------------------------------------------------------------

BATCH_CASES = [
    (2, 64),
    (2, 1024),
    (4, 256),
    (8, 128),
]
BATCH_H = [1, 4, 96]


@pytest.mark.parametrize("H", BATCH_H, ids=[f"H{h}" for h in BATCH_H])
@pytest.mark.parametrize("B,T", BATCH_CASES, ids=[f"B{b}_T{t}" for b, t in BATCH_CASES])
@pytest.mark.parametrize("state_dtype", STATE_DTYPES)
@pytest.mark.parametrize("has_in,has_out", STATE_IO, ids=STATE_IO_IDS)
def test_fwd_batched(B, T, H, state_dtype, has_in, has_out):
    torch.manual_seed(42)
    dtype = torch.bfloat16 if state_dtype == "bf16" else torch.float32

    q = F.normalize(torch.randn((B, T, H, D), dtype=torch.float32, device="cuda"), p=2, dim=-1).to(torch.bfloat16)
    k = F.normalize(torch.randn((B, T, H, D), dtype=torch.float32, device="cuda"), p=2, dim=-1).to(torch.bfloat16)
    v = torch.randn((B, T, H, D), dtype=torch.bfloat16, device="cuda")
    g = torch.randn((B, T, H, D), dtype=torch.bfloat16, device="cuda")
    beta = torch.randn((B, T, H), dtype=torch.bfloat16, device="cuda")
    A_log = torch.rand(H, dtype=torch.float32, device="cuda")
    dt_bias = torch.rand(H, D, dtype=torch.float32, device="cuda")
    scale = 1.0 / math.sqrt(D)

    init_k = _make_state((B, H, D, D), dtype).clone() if has_in else None
    init_r = _make_state((B, H, D, D), dtype).clone() if has_in else None
    final_k = torch.zeros(B, H, D, D, dtype=dtype, device="cuda") if has_out else None
    final_r = torch.zeros(B, H, D, D, dtype=dtype, device="cuda") if has_out else None

    # batched flash_kda (B > 1, auto cu_seqlens)
    out_kernel = torch.zeros_like(q)
    flash_kda.fwd(q, k, v, g, beta, scale, out_kernel,
                  A_log=A_log, dt_bias=dt_bias, lower_bound=LOWER_BOUND,
                  initial_state=init_k, final_state=final_k)
    torch.cuda.synchronize()

    # torch ref
    out_ref = torch.zeros_like(q)
    torch_ref(q, k, v, g, beta, scale, out_ref,
              A_log=A_log, dt_bias=dt_bias, lower_bound=LOWER_BOUND,
              initial_state=init_r, final_state=final_r)

    assert torch.equal(out_kernel, out_ref), \
        f"output mismatch: B={B} T={T} H={H} dtype={state_dtype} in={has_in} out={has_out}"
    if final_k is not None:
        assert torch.equal(final_k, final_r), \
            f"final_state mismatch: B={B} T={T} H={H} dtype={state_dtype} in={has_in} out={has_out}"


# ---------------------------------------------------------------------------
# Long-sequence tests (H=1, in+out bf16 only)
# ---------------------------------------------------------------------------

LONG_T_VALUES = [131072, 1048576]
LONG_VARLEN_CASES = [
    [131072],
    [524288, 524288],
]


@pytest.mark.parametrize("T", LONG_T_VALUES, ids=[f"T{t}" for t in LONG_T_VALUES])
def test_fwd_long(T):
    H = 1
    q, k, v, g, beta, A_log, dt_bias, scale = _make_inputs(T, H)

    init_k = _make_state((1, H, D, D), torch.bfloat16).clone()
    init_r = _make_state((1, H, D, D), torch.bfloat16).clone()
    final_k = torch.zeros(1, H, D, D, dtype=torch.bfloat16, device="cuda")
    final_r = torch.zeros(1, H, D, D, dtype=torch.bfloat16, device="cuda")

    out_kernel = torch.zeros_like(q)
    flash_kda.fwd(q, k, v, g, beta, scale, out_kernel,
                  A_log=A_log, dt_bias=dt_bias, lower_bound=LOWER_BOUND,
                  initial_state=init_k, final_state=final_k)
    torch.cuda.synchronize()

    out_ref = torch.zeros_like(q)
    torch_ref(q, k, v, g, beta, scale, out_ref,
              A_log=A_log, dt_bias=dt_bias, lower_bound=LOWER_BOUND,
              initial_state=init_r, final_state=final_r)

    assert torch.equal(out_kernel, out_ref), f"output mismatch: T={T}"
    assert torch.equal(final_k, final_r), f"final_state mismatch: T={T}"


@pytest.mark.parametrize("seq_lens", LONG_VARLEN_CASES,
                         ids=[f"seqs{'_'.join(str(s) for s in sl)}" for sl in LONG_VARLEN_CASES])
def test_fwd_long_varlen(seq_lens):
    H = 1
    T_total = sum(seq_lens)
    N = len(seq_lens)
    cu_seqlens = torch.tensor(
        [0] + list(torch.cumsum(torch.tensor(seq_lens), dim=0).tolist()),
        dtype=torch.long, device="cuda",
    )

    q, k, v, g, beta, A_log, dt_bias, scale = _make_inputs(T_total, H)

    init_k = _make_state((N, H, D, D), torch.bfloat16).clone()
    init_r = _make_state((N, H, D, D), torch.bfloat16).clone()
    final_k = torch.zeros(N, H, D, D, dtype=torch.bfloat16, device="cuda")
    final_r = torch.zeros(N, H, D, D, dtype=torch.bfloat16, device="cuda")

    out_kernel = torch.zeros_like(q)
    flash_kda.fwd(q, k, v, g, beta, scale, out_kernel,
                  A_log=A_log, dt_bias=dt_bias, lower_bound=LOWER_BOUND,
                  initial_state=init_k, final_state=final_k, cu_seqlens=cu_seqlens)
    torch.cuda.synchronize()

    out_ref = torch.zeros_like(q)
    torch_ref(q, k, v, g, beta, scale, out_ref,
              A_log=A_log, dt_bias=dt_bias, lower_bound=LOWER_BOUND,
              initial_state=init_r, final_state=final_r, cu_seqlens=cu_seqlens)

    assert torch.equal(out_kernel, out_ref), f"output mismatch: seqs={seq_lens}"
    assert torch.equal(final_k, final_r), f"final_state mismatch: seqs={seq_lens}"
