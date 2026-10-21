"""CUDA's full forward test matrix, executed through the public NPU API.

Case names, parameters and state construction mirror tests/test_fwd_full.py.
CUDA's instruction-matching torch.equal oracle is replaced by independent CPU
FP64 arithmetic with explicit BF16 error budgets (see README). Native-specific
pointwise and bitwise checks remain in test_fwd.py.
"""
import pytest
import torch

import flash_kda
from flash_kda_npu import kda_fwd
from flash_kda_npu.tests.cuda_cases import (
    BATCH_CASES, BATCH_H, H_VALUES, LONG_T_VALUES, LONG_VARLEN_CASES,
    STATE_DTYPES, STATE_IO, STATE_IO_IDS, T_VALUES, VARLEN_CASES, VARLEN_H,
)
from flash_kda_npu.tests.torch_ref import (
    assert_cuda_accuracy, assert_post_transient_accuracy, chunked_reference,
    make_cuda_inputs, make_cuda_state,
)

pytestmark = pytest.mark.skipif(not kda_fwd.is_available(), reason="requires an Ascend NPU")


def _check_case(batch, steps, heads, state_dtype, has_in, has_out, record_property,
                seq_lens=None, seed=42):
    dtype = torch.bfloat16 if state_dtype == "bf16" else torch.float32
    inputs = make_cuda_inputs(batch, steps, heads, seed=seed)
    sequences = batch if seq_lens is None else len(seq_lens)
    initial = make_cuda_state(sequences, heads, dtype) if has_in else None
    q, k, v, g, beta, a_log, bias = [x.to("npu") for x in inputs]
    init = None if initial is None else initial.to("npu")
    saved = None if init is None else init.clone()
    final = (torch.full((sequences, heads, 128, 128), float("nan"),
                        device="npu", dtype=dtype) if has_out else None)
    out = torch.full_like(q, float("nan"))
    cu = (None if seq_lens is None else torch.tensor(
        [0, *torch.tensor(seq_lens).cumsum(0).tolist()], dtype=torch.int64, device="npu"))
    assert flash_kda.fwd(q, k, v, g, beta, 128 ** -0.5, out, a_log, bias, -5.0,
                         initial_state=init, final_state=final, cu_seqlens=cu) is None
    torch.npu.synchronize()
    from flash_kda_npu.ascendc import get_last_profile
    assert get_last_profile()["launch_counts"].get("kda_k2_persistent_loop", 0) >= 1
    if init is not None:
        torch.testing.assert_close(init, saved, rtol=0, atol=0)
    expected, expected_state = chunked_reference(inputs, initial, seq_lens)
    # Large arange states amplify BF16 dot-product cancellation in the first
    # few tokens. Retain the tighter criterion after that initial transient.
    assert_cuda_accuracy(out, expected, "output", record_property,
                         threshold=0.01 if has_in else (0.006 if seq_lens else 0.005))
    if has_in:
        lengths = [steps] * batch if seq_lens is None else seq_lens
        assert_post_transient_accuracy(out, expected, lengths, record_property)
    if final is not None:
        assert_cuda_accuracy(final, expected_state, "state", record_property)


@pytest.mark.parametrize("seq_lens", [None, [1300, 547, 2048, 963, 271, 3063]],
                         ids=["fixed", "varlen"])
def test_cuda_smoke(seq_lens, record_property):
    """The two H=96, seed=0 smoke scenarios from CUDA's test_fwd.py."""
    _check_case(1, 8192, 96, "bf16", True, True, record_property, seq_lens, seed=0)


@pytest.mark.parametrize("H", H_VALUES, ids=[f"H{h}" for h in H_VALUES])
@pytest.mark.parametrize("T", T_VALUES, ids=[f"T{t}" for t in T_VALUES])
@pytest.mark.parametrize("state_dtype", STATE_DTYPES)
@pytest.mark.parametrize("has_in,has_out", STATE_IO, ids=STATE_IO_IDS)
def test_fwd_fixed(T, H, state_dtype, has_in, has_out, record_property):
    _check_case(1, T, H, state_dtype, has_in, has_out, record_property)


@pytest.mark.parametrize("H", VARLEN_H, ids=[f"H{h}" for h in VARLEN_H])
@pytest.mark.parametrize("seq_lens", VARLEN_CASES,
                         ids=[f"seqs{'_'.join(str(s) for s in sl)}" for sl in VARLEN_CASES])
@pytest.mark.parametrize("state_dtype", STATE_DTYPES)
@pytest.mark.parametrize("has_in,has_out", STATE_IO, ids=STATE_IO_IDS)
def test_fwd_varlen(seq_lens, H, state_dtype, has_in, has_out, record_property):
    _check_case(1, sum(seq_lens), H, state_dtype, has_in, has_out, record_property, seq_lens)


@pytest.mark.parametrize("H", BATCH_H, ids=[f"H{h}" for h in BATCH_H])
@pytest.mark.parametrize("B,T", BATCH_CASES, ids=[f"B{b}_T{t}" for b, t in BATCH_CASES])
@pytest.mark.parametrize("state_dtype", STATE_DTYPES)
@pytest.mark.parametrize("has_in,has_out", STATE_IO, ids=STATE_IO_IDS)
def test_fwd_batched(B, T, H, state_dtype, has_in, has_out, record_property):
    _check_case(B, T, H, state_dtype, has_in, has_out, record_property)


@pytest.mark.parametrize("T", LONG_T_VALUES, ids=[f"T{t}" for t in LONG_T_VALUES])
def test_fwd_long(request, T, record_property):
    if not request.config.getoption("--run-long"):
        pytest.skip("use --run-long for CUDA's full long-sequence matrix")
    _check_case(1, T, 1, "bf16", True, True, record_property)


@pytest.mark.parametrize("seq_lens", LONG_VARLEN_CASES,
                         ids=[f"seqs{'_'.join(str(s) for s in sl)}" for sl in LONG_VARLEN_CASES])
def test_fwd_long_varlen(request, seq_lens, record_property):
    if not request.config.getoption("--run-long"):
        pytest.skip("use --run-long for CUDA's full long-sequence matrix")
    _check_case(1, sum(seq_lens), 1, "bf16", True, True, record_property, seq_lens)
