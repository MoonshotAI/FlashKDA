"""Native AscendC accuracy and launcher coverage."""
import warnings

import pytest
import torch

import flash_kda
from flash_kda_npu import kda_fwd
from flash_kda_npu.tests.torch_ref import (
    CUDA_GATE_CASE_IDS, assert_cuda_accuracy, chunked_reference, make_cuda_gate_case,
    relative_rms,
    STATE_IDS, STATE_IO, assert_close, assert_relative_rms, make_inputs, make_state, recurrent_reference,
)

pytestmark = pytest.mark.skipif(not kda_fwd.is_available(), reason="requires an Ascend NPU")


@pytest.mark.parametrize("case", range(len(CUDA_GATE_CASE_IDS)), ids=CUDA_GATE_CASE_IDS)
@pytest.mark.parametrize("seq_lens", [None, [1300, 547, 2048, 963, 271, 3063]],
                         ids=["fixed", "varlen"])
def test_cuda_gate_bias_accuracy(case, seq_lens, record_property):
    """CUDA's FLA sweep scenarios checked against the independent FP64 oracle."""
    inputs, initial = make_cuda_gate_case(case, seq_lens)
    q, k, v, g, beta, a_log, bias = [x.to("npu") for x in inputs]
    init = initial.to("npu")
    saved = init.clone()
    final = torch.full_like(init, float("nan"))
    out = torch.full_like(q, float("nan"))
    cu = (None if seq_lens is None else torch.tensor(
        [0, *torch.tensor(seq_lens).cumsum(0).tolist()], dtype=torch.int64, device="npu"))
    assert flash_kda.fwd(q, k, v, g, beta, 128 ** -0.5, out, a_log, bias, -5.0,
                         initial_state=init, final_state=final, cu_seqlens=cu) is None
    torch.npu.synchronize()
    from flash_kda_npu.ascendc import get_last_profile
    assert get_last_profile()["launch_counts"].get("kda_k2_persistent_loop", 0) >= 1
    torch.testing.assert_close(init, saved, rtol=0, atol=0)
    expected, expected_state = chunked_reference(inputs, initial, seq_lens)
    threshold = 0.005 if seq_lens is None else 0.006
    # CUDA's FLA comparison asserts global RMS and plots window errors.
    assert_cuda_accuracy(out, expected, "output", record_property, threshold,
                         check_windows=False)
    # CUDA's FLA sweep reports final-state threshold violations as warnings.
    state_ratio = relative_rms(final, expected_state)  # Nonfinite values still fail.
    record_property("state_relative_rms", state_ratio)
    if state_ratio >= threshold:
        warnings.warn(f"gate sweep state relative RMS {state_ratio:.6g} >= {threshold}",
                      RuntimeWarning, stacklevel=2)


def check_native(batch, steps, heads, state_dtype=torch.float32,
                 has_in=True, has_out=True, gate_dtype=torch.bfloat16,
                 gate_value=None, lower_bound=-5.0):
    inputs = make_inputs(batch, steps, heads, "npu", gate_dtype=gate_dtype)
    if gate_value is not None:
        inputs[3].fill_(gate_value)
    initial = make_state(batch, heads, 128, state_dtype, "npu") if has_in else None
    saved_initial = initial.clone() if initial is not None else None
    final = torch.full((batch, heads, 128, 128), float("nan"), dtype=state_dtype,
                       device="npu") if has_out else None
    out = torch.full_like(inputs[2], float("nan"))
    q, k, v, g, beta, a_log, bias = inputs
    if gate_dtype == torch.bfloat16:
        result = flash_kda.fwd(q, k, v, g, beta, 128 ** -0.5, out,
                              A_log=a_log, dt_bias=bias, lower_bound=lower_bound,
                              initial_state=initial, final_state=final)
        assert result is None
    else:
        # The lower-level interface accepts FP32 gates; CUDA's public API does not.
        from flash_kda_npu.ascendc import kda_fwd_ascendc
        kda_fwd_ascendc(
            q, k, v, g, beta.float(), scale=128 ** -0.5, out=out,
            A_log=a_log, bias=bias, lower_bound=lower_bound,
            initial_state=None if initial is None else initial.float(),
            output_final_state=final is not None, final_state=final)
    torch.npu.synchronize()
    from flash_kda_npu.ascendc import get_last_profile
    assert get_last_profile()["launch_counts"].get("kda_k2_persistent_loop", 0) == 1
    expected_out, expected_state = recurrent_reference(inputs, saved_initial, lower_bound=lower_bound)
    assert_close(out, expected_out)
    if final is not None:
        # CUDA gate sweeps use relative RMS: weak decay accumulates bf16
        # rounding across tokens, making per-element relative error unstable.
        if gate_value is not None:
            assert_relative_rms(final, expected_state)
        else:
            assert_close(final, expected_state)
    if initial is not None:
        torch.testing.assert_close(initial, saved_initial, rtol=0, atol=0)


@pytest.mark.parametrize("has_in,has_out", STATE_IO, ids=STATE_IDS)
@pytest.mark.parametrize("state_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("batch,steps,heads", [(1, 64, 1), (1, 256, 4), (2, 128, 4)])
def test_native_state_io(batch, steps, heads, state_dtype, has_in, has_out):
    check_native(batch, steps, heads, state_dtype, has_in, has_out)


@pytest.mark.parametrize("heads", [1, 4, 32, 96])
def test_native_head_mapping(heads):
    check_native(1, 64, heads)


@pytest.mark.parametrize("batch,steps,heads", [(4, 256, 1), (8, 128, 4)])
def test_native_batched(batch, steps, heads):
    check_native(batch, steps, heads)


@pytest.mark.parametrize("cores,before,after", [
    (20, 4096, 4192), (20, 12288, 12576),
    (24, 8192, 8384), (24, 12288, 12576),
    (32, 8192, 8384), (32, 12288, 12576),
])
def test_native_pre_gram_schedule(cores, before, after):
    """A small tail must not turn a few waves into thousands of tiny blocks."""
    from flash_kda_npu.ascendc import _pre_gram_unroll
    old_u = _pre_gram_unroll(before, cores)
    new_u = _pre_gram_unroll(after, cores)
    assert 32 <= old_u <= 88 and 32 <= new_u <= 88
    old_grid = (before + 2 * old_u - 1) // (2 * old_u)
    new_grid = (after + 2 * new_u - 1) // (2 * new_u)
    assert new_grid <= 2 * old_grid
    # Both fixed shapes already had efficient schedules; retain those
    # choices when startup and compute costs tie on this 24-core device.
    assert _pre_gram_unroll(8192, 24) == 57
    assert _pre_gram_unroll(12288, 24) == 64


@pytest.mark.parametrize("cores,heads,expected", [
    (20, 96, 40), (20, 128, 40), (20, 768, 256),
    (20, 32, 16), (24, 96, 24), (32, 96, 32),
])
def test_native_fixed_persistent_schedule(cores, heads, expected):
    from flash_kda_npu.ascendc import _persistent_fixed_blocks
    assert _persistent_fixed_blocks(heads, cores) == expected


@pytest.mark.parametrize("cores", [20, 24, 32])
def test_native_fixed_persistent_grid_coverage(cores):
    from flash_kda_npu.ascendc import PERSIST_MAXH, _persistent_fixed_blocks
    for heads in range(1, 1025):
        nb = _persistent_fixed_blocks(heads, cores)
        assert 1 <= nb <= heads
        assert (heads + nb - 1) // nb <= PERSIST_MAXH
        mapped = [head for block in range(nb) for head in range(block, heads, nb)]
        assert sorted(mapped) == list(range(heads))


@pytest.mark.parametrize("cores", [20, 24, 32])
def test_native_varlen_persistent_schedule(cores):
    from flash_kda_npu.ascendc import _persistent_varlen_blocks
    # A long sequence mixed with short sequences leaves single-head tails
    # in the old three-head stride map. Four heads retain more overlap.
    assert _persistent_varlen_blocks((21, 9, 32, 16, 5, 48), 96, cores, 4, 192) == 144
    # Equal-length and all-empty batches preserve the existing choice.
    assert _persistent_varlen_blocks((16,) * 8, 96, cores, 4, 256) == 256
    assert _persistent_varlen_blocks((0,) * 8, 96, cores, 4, 256) == 256
    for counts in ((0, 1, 0, 5, 1, 0, 2, 0), (1, 1000), (9, 2, 4)):
        for heads in (1, 3, 17, 96):
            bh = len(counts) * heads
            default = (bh + 3) // 4
            nb = _persistent_varlen_blocks(counts, heads, cores, 4, default)
            assert 1 <= nb <= bh and (bh + nb - 1) // nb <= 4


@pytest.mark.parametrize("counts,heads,default", [
    ((21, 9, 32, 16, 5, 48), 96, 144),
    ((0, 1, 1, 110, 1, 17), 32, 96),
])
def test_native_grouped_persistent_plan(counts, heads, default):
    from flash_kda_npu.ascendc import _persistent_grouped_plan
    nb, rows = _persistent_grouped_plan(counts, heads, 20, 4, default)
    assert rows and len(rows) == nb
    assert sorted(h for row in rows for h in row if h >= 0) == list(range(len(counts) * heads))
    assert all(len(row) == 4 and any(h >= 0 for h in row) for row in rows)
    work = [max(counts[h // heads] for h in row if h >= 0) for row in rows]
    assert work == sorted(work, reverse=True)
    assert _persistent_grouped_plan((16,) * 8, 96, 20, 4, 256) == (256, ())
    assert _persistent_grouped_plan((0, 1, 0, 5, 1, 0, 2, 0), 17, 20, 4, 34) == (34, ())


@pytest.mark.parametrize("unroll", [1, 3, 8])
@pytest.mark.parametrize("with_bias", [False, True])
def test_native_head_parameters(monkeypatch, unroll, with_bias):
    """Refresh cached metadata across heads, batches and the padded tail."""
    from flash_kda_npu.ascendc import CHUNK, kda_fwd_ascendc
    monkeypatch.setenv("KDA_PRE_UNROLL", str(unroll))
    inputs = list(make_inputs(2, CHUNK * 5, 3, "npu", gate_dtype=torch.float32))
    inputs[5] = torch.tensor([-1.0, 0.25, 1.0], device="npu")
    inputs[6] = torch.linspace(-0.5, 0.5, 3 * 128, device="npu").reshape(3, 128)
    if not with_bias:
        inputs[6].zero_()
    q, k, v, g, beta, a_log, bias = inputs
    out, state = kda_fwd_ascendc(
        q, k, v, g, beta.float(), A_log=a_log,
        bias=bias if with_bias else None, output_final_state=True)
    torch.npu.synchronize()
    expected_out, expected_state = recurrent_reference(inputs)
    assert_close(out, expected_out)
    assert_close(state, expected_state)


@pytest.mark.parametrize("blocks", [1, 2, 4])
def test_native_reused_handoff_slots(monkeypatch, blocks):
    """Exercise repeated handoffs with one, two or four heads per block."""
    monkeypatch.setenv("KDA_PERSIST_LOOP_BLOCKS", str(blocks))
    check_native(1, 512, 4, gate_dtype=torch.float32)


@pytest.mark.parametrize("state_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("gate_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("gate_value", [-8.0, -4.0, 0.0, 8.0])
def test_native_gate_sweep(state_dtype, gate_dtype, gate_value):
    check_native(1, 128, 2, state_dtype=state_dtype, gate_dtype=gate_dtype, gate_value=gate_value)


@pytest.mark.parametrize("lower_bound", [-1.0, -3.0])
def test_native_lower_bound(lower_bound):
    check_native(1, 128, 2, lower_bound=lower_bound)


@pytest.mark.parametrize("steps", [1024, 8192])
def test_native_multichunk(steps):
    check_native(1, steps, 1)


@pytest.mark.parametrize("steps", [131072, 1048576])
def test_native_long(request, steps):
    if not request.config.getoption("--run-long"):
        pytest.skip("use --run-long for the CUDA-scale long-sequence cases")
    check_native(1, steps, 1)


@pytest.mark.parametrize("chunk", [16, 32, 64])
def test_native_configured_chunk(chunk):
    from flash_kda_npu.ascendc import CHUNK, kda_fwd_ascendc
    if CHUNK != chunk:
        pytest.skip(f"requires a fresh process with KDA_CHUNK={chunk}")
    inputs = make_inputs(1, chunk * 3, 2, "npu", gate_dtype=torch.float32)
    q, k, v, g, beta, a_log, bias = inputs
    out, state = kda_fwd_ascendc(q, k, v, g, beta.float(),
                                    A_log=a_log, bias=bias, output_final_state=True)
    torch.npu.synchronize()
    expected_out, expected_state = recurrent_reference(inputs)
    assert_close(out, expected_out)
    assert_close(state, expected_state)


@pytest.mark.parametrize("initial_dtype,final_dtype", [
    (torch.float32, torch.bfloat16), (torch.bfloat16, torch.float32),
])
def test_native_rejects_mixed_state_dtypes(initial_dtype, final_dtype):
    inputs = make_inputs(2, 320, 3, "npu")
    q, k, v, g, beta, a_log, bias = inputs
    initial = make_state(2, 3, 128, initial_dtype, "npu")
    final = torch.full_like(initial, 7, dtype=final_dtype)
    out = torch.full_like(q, 11)
    with pytest.raises(RuntimeError, match="must have the same dtype"):
        flash_kda.fwd(q, k, v, g, beta, 128 ** -0.5, out, A_log=a_log,
                      dt_bias=bias, lower_bound=-5.0, initial_state=initial, final_state=final)
    assert bool((final == 7).all())
    assert bool((out == 11).all())


@pytest.mark.parametrize("steps", [1, 16, 17, 37, 63, 65, 97, 129])
@pytest.mark.parametrize("state_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("has_in,has_out", STATE_IO, ids=STATE_IDS)
def test_native_partial_state_io(steps, state_dtype, has_in, has_out):
    check_native(2, steps, 2, state_dtype, has_in, has_out)


@pytest.mark.parametrize("tail", range(1, 64))
@pytest.mark.parametrize("state_dtype", [torch.bfloat16, torch.float32])
def test_native_tail_direct_buffers(monkeypatch, tail, state_dtype):
    """Every tail size matches neutral padding, without packing caller buffers."""
    from flash_kda_npu import ascendc
    if tail >= ascendc.CHUNK:
        pytest.skip("tail must be smaller than the configured chunk")
    batch, heads = 2, 3
    steps = ascendc.CHUNK + tail
    inputs = make_inputs(batch, steps, heads, "npu")
    q, k, v, g, beta, a_log, bias = inputs
    initial = make_state(batch, heads, 128, state_dtype, "npu")
    guard = 256
    out_storage = torch.full((q.numel() + 2 * guard,), 11, dtype=q.dtype, device=q.device)
    out = out_storage[guard:-guard].view_as(q)
    state_storage = torch.full((initial.numel() + 2 * guard,), 7,
                               dtype=state_dtype, device=q.device)
    final = state_storage[guard:-guard].view_as(initial)
    native = ascendc.kda_fwd_ascendc
    calls = []

    def direct(*args, **kwargs):
        calls.append((args[0].data_ptr(), kwargs["out"].data_ptr(),
                      kwargs["final_state"].data_ptr()))
        assert kwargs.get("valid_lengths") is None
        return native(*args, **kwargs)

    monkeypatch.setattr(ascendc, "kda_fwd_ascendc", direct)
    flash_kda.fwd(q, k, v, g, beta, 128 ** -0.5, out, a_log, bias, -5.0,
                  initial_state=initial, final_state=final)
    assert calls == [(q.data_ptr(), out.data_ptr(), final.data_ptr())]
    assert ascendc.get_last_profile()["input_packing"] is False

    padded = 2 * ascendc.CHUNK
    packed = []
    for x in (q, k, v, g, beta):
        target = torch.zeros((batch, padded, *x.shape[2:]), dtype=x.dtype, device=x.device)
        target[:, :steps].copy_(x)
        packed.append(target)
    expected, expected_state = native(
        *packed[:3], packed[3].float(), packed[4].float(), scale=128 ** -0.5,
        initial_state=initial.float(), output_final_state=True,
        final_state=torch.empty_like(initial), A_log=a_log, bias=bias,
        valid_lengths=torch.full((batch,), steps, dtype=torch.int32, device=q.device))
    torch.npu.synchronize()
    torch.testing.assert_close(out, expected[:, :steps], rtol=0, atol=0)
    torch.testing.assert_close(final, expected_state, rtol=0, atol=0)
    for storage, sentinel in ((out_storage, 11), (state_storage, 7)):
        assert bool((storage[:guard] == sentinel).all())
        assert bool((storage[-guard:] == sentinel).all())


@pytest.mark.parametrize("unroll", [1, 3, 8])
@pytest.mark.parametrize("gate_value", [-8.0, 8.0])
def test_native_tail_unroll(monkeypatch, unroll, gate_value):
    from flash_kda_npu.ascendc import CHUNK
    monkeypatch.setenv("KDA_PRE_UNROLL", str(unroll))
    check_native(3, CHUNK * 5 + 1, 3, gate_value=gate_value)


def test_native_tail_long(request):
    if not request.config.getoption("--run-long"):
        pytest.skip("use --run-long for a long direct tail")
    check_native(1, 65537, 1)


@pytest.mark.parametrize("lengths", [
    [16], [16, 16], [4, 8, 12], [64, 128, 256], [17, 33, 65],
    [0, 1, 0, 63, 65, 0], [64, 64], [0, 0],
])
@pytest.mark.parametrize("state_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("has_in,has_out", STATE_IO, ids=STATE_IDS)
def test_native_varlen_state_io(lengths, state_dtype, has_in, has_out):
    inputs = make_inputs(1, sum(lengths), 2, "npu")
    q, k, v, g, beta, a_log, bias = inputs
    cu = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int64, device="npu")
    initial = make_state(len(lengths), 2, 128, state_dtype, "npu") if has_in else None
    saved = None if initial is None else initial.clone()
    final = torch.full((len(lengths), 2, 128, 128), float("nan"), dtype=state_dtype,
                       device="npu") if has_out else None
    out = torch.full_like(q, float("nan"))
    assert flash_kda.fwd(q, k, v, g, beta, 128 ** -0.5, out, a_log, bias, -5.0,
                         initial_state=initial, final_state=final, cu_seqlens=cu) is None
    torch.npu.synchronize()
    expected_out, expected_state = recurrent_reference(inputs, saved, lengths)
    assert_close(out, expected_out)
    if final is not None:
        assert_close(final, expected_state)
    if initial is not None:
        torch.testing.assert_close(initial, saved, rtol=0, atol=0)
    from flash_kda_npu.ascendc import CHUNK, get_last_profile
    profile = get_last_profile()
    if any(lengths):
        assert profile["launch_counts"].get("kda_k2_persistent_loop", 0) >= 1
    else:
        assert profile["launch_total"] == 0
    if "padded_tokens" in profile:
        assert profile["padded_tokens"] - sum(lengths) < len(lengths) * CHUNK


@pytest.mark.parametrize("offsets", [[1, 64, 128], [0, 64, 127], [0, 90, 64, 128]])
def test_native_invalid_boundaries_preserve_buffers(offsets):
    q, k, v, g, beta, a_log, bias = make_inputs(1, 128, 2, "npu")
    out = torch.full_like(q, 11)
    final = torch.full((len(offsets) - 1, 2, 128, 128), 7, dtype=torch.float32, device="npu")
    cu = torch.tensor(offsets, dtype=torch.int64, device="npu")
    with pytest.raises(RuntimeError, match="cu_seqlens must"):
        flash_kda.fwd(q, k, v, g, beta, 128 ** -0.5, out, a_log, bias, -5.0,
                      final_state=final, cu_seqlens=cu)
    assert bool((out == 11).all())
    assert bool((final == 7).all())


@pytest.mark.parametrize("gate_value", [-8.0, 8.0])
@pytest.mark.parametrize("state_dtype", [torch.bfloat16, torch.float32])
def test_native_varlen_extreme_gates(gate_value, state_dtype):
    lengths = [1, 17, 65]
    inputs = make_inputs(1, sum(lengths), 2, "npu")
    q, k, v, g, beta, a_log, bias = inputs
    g.fill_(gate_value)
    initial = make_state(3, 2, 128, state_dtype, "npu")
    final = torch.empty_like(initial)
    out = torch.empty_like(q)
    cu = torch.tensor([0, 1, 18, 83], dtype=torch.int64, device="npu")
    flash_kda.fwd(q, k, v, g, beta, 128 ** -0.5, out, a_log, bias, -5.0,
                  initial_state=initial, final_state=final, cu_seqlens=cu)
    expected_out, expected_state = recurrent_reference(inputs, initial, lengths)
    assert_close(out, expected_out)
    assert_relative_rms(final, expected_state)


def test_native_varlen_matches_independent_sequences():
    lengths = [17, 37, 65, 0, 17]
    inputs = make_inputs(1, sum(lengths), 2, "npu")
    q, k, v, g, beta, a_log, bias = inputs
    initial = make_state(5, 2, 128, torch.float32, "npu")
    final = torch.empty_like(initial)
    out = torch.empty_like(q)
    boundaries = [0, *torch.tensor(lengths).cumsum(0).tolist()]
    cu = torch.tensor(boundaries, dtype=torch.int64, device="npu")
    flash_kda.fwd(q, k, v, g, beta, 128 ** -0.5, out, a_log, bias, -5.0,
                  initial_state=initial, final_state=final, cu_seqlens=cu)
    for seq, (start, end) in enumerate(zip(boundaries, boundaries[1:])):
        if start == end:
            torch.testing.assert_close(final[seq], initial[seq], rtol=0, atol=0)
            continue
        parts = [x[:, start:end].contiguous() for x in (q, k, v, g, beta)]
        seq_out = torch.empty_like(parts[0])
        seq_final = torch.empty_like(initial[seq:seq + 1])
        flash_kda.fwd(*parts, 128 ** -0.5, seq_out, a_log, bias, -5.0,
                      initial_state=initial[seq:seq + 1], final_state=seq_final)
        torch.testing.assert_close(out[:, start:end], seq_out, rtol=0, atol=0)
        torch.testing.assert_close(final[seq:seq + 1], seq_final, rtol=0, atol=0)


@pytest.mark.parametrize("heads", [1, 3, 7, 17])
@pytest.mark.parametrize("state_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("unroll", [1, 3])
@pytest.mark.parametrize("head_map", ["auto", "stride", "grouped"])
def test_native_varlen_direct_buffers(monkeypatch, heads, state_dtype, unroll, head_map):
    """Mixed head schedules retire independently, with no full-input packing."""
    from flash_kda_npu import ascendc
    monkeypatch.setenv("KDA_PRE_UNROLL", str(unroll))
    monkeypatch.setenv("KDA_PERSIST_HEAD_MAP", head_map)
    lengths = [0, 1, 0, ascendc.CHUNK * 4 + 3, 7, 0, ascendc.CHUNK + 1, 0]
    steps = sum(lengths)
    q, k, v, g, beta, a_log, bias = make_inputs(1, steps, heads, "npu")
    initial = make_state(len(lengths), heads, 128, state_dtype, "npu")
    guard = 256
    out_storage = torch.full((q.numel() + guard * 2,), 11, device="npu", dtype=q.dtype)
    state_storage = torch.full((initial.numel() + guard * 2,), 7, device="npu", dtype=state_dtype)
    out = out_storage[guard:-guard].view_as(q)
    final = state_storage[guard:-guard].view_as(initial)
    cu = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], device="npu", dtype=torch.int64)
    calls = []
    native = ascendc.kda_fwd_ascendc

    def direct(*args, **kwargs):
        calls.append((args[0].data_ptr(), kwargs["out"].data_ptr(), kwargs["final_state"].data_ptr()))
        assert kwargs.get("sequence_offsets") == tuple(cu.cpu().tolist())
        assert kwargs.get("valid_lengths") is None
        return native(*args, **kwargs)

    monkeypatch.setattr(ascendc, "kda_fwd_ascendc", direct)
    flash_kda.fwd(q, k, v, g, beta, 128 ** -0.5, out, a_log, bias, -5.0,
                  initial_state=initial, final_state=final, cu_seqlens=cu)
    torch.npu.synchronize()
    assert calls == [(q.data_ptr(), out.data_ptr(), final.data_ptr())]
    profile = ascendc.get_last_profile()
    assert profile["packed_direct"] and not profile["input_packing"]
    assert profile["pre_unroll"] == unroll
    if head_map != "auto":
        assert profile["persistent_head_map"] == head_map
    assert profile["sequence_groups"] == 1
    assert profile["launch_counts"]["kda_pre_gram_mix"] == 1
    assert profile["launch_counts"]["kda_k2_persistent_loop"] == 1
    assert profile["padded_tokens"] == sum(
        (n + ascendc.CHUNK - 1) // ascendc.CHUNK * ascendc.CHUNK for n in lengths)
    expected, state = recurrent_reference((q, k, v, g, beta, a_log, bias), initial, lengths)
    assert_close(out, expected)
    assert_close(final, state)
    for storage, sentinel in ((out_storage, 11), (state_storage, 7)):
        assert bool((storage[:guard] == sentinel).all())
        assert bool((storage[-guard:] == sentinel).all())


def test_native_varlen_long(request):
    if not request.config.getoption("--run-long"):
        pytest.skip("use --run-long for a long packed sequence with partial chunks")
    lengths = [65537, 17, 63]
    inputs = make_inputs(1, sum(lengths), 1, "npu")
    q, k, v, g, beta, a_log, bias = inputs
    initial = make_state(3, 1, 128, torch.float32, "npu")
    final = torch.empty_like(initial)
    out = torch.empty_like(q)
    cu = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int64, device="npu")
    flash_kda.fwd(q, k, v, g, beta, 128 ** -0.5, out, a_log, bias, -5.0,
                  initial_state=initial, final_state=final, cu_seqlens=cu)
    expected_out, expected_state = recurrent_reference(inputs, initial, lengths)
    assert_close(out, expected_out)
    assert_close(final, expected_state)


@pytest.mark.parametrize("chunks", [1, 5, 128])
def test_native_final_state_rounding(chunks):
    from flash_kda_npu.ascendc import CHUNK, kda_fwd_ascendc
    q, k, v, g, beta, a_log, bias = make_inputs(
        1, CHUNK * chunks, 2, "npu", gate_dtype=torch.float32)
    kwargs = dict(A_log=a_log, bias=bias, output_final_state=True)
    out32, state32 = kda_fwd_ascendc(q, k, v, g, beta.float(), **kwargs)
    final16 = torch.empty_like(state32, dtype=torch.bfloat16)
    out16, state16 = kda_fwd_ascendc(
        q, k, v, g, beta.float(), final_state=final16, **kwargs)
    out_none, no_state = kda_fwd_ascendc(
        q, k, v, g, beta.float(), A_log=a_log, bias=bias)
    torch.npu.synchronize()
    assert state16.data_ptr() == final16.data_ptr()
    assert no_state is None
    torch.testing.assert_close(out16, out32, rtol=0, atol=0)
    torch.testing.assert_close(out_none, out32, rtol=0, atol=0)
    torch.testing.assert_close(state16, state32.bfloat16(), rtol=0, atol=0)


@pytest.mark.parametrize("bad_args,match", [
    ([], "expects 5 arguments"),
    ([b"\0"] + [b"\0" * 8] * 3 + [b"\0" * 4], "argument 0 expects 8 bytes"),
    ([b"\0" * 8] * 5, "argument 4 expects 4 bytes"),
])
def test_launcher_rejects_invalid_abi(bad_args, match):
    from flash_kda_npu.ascendc import _compile_all, launch_argsarray_engine
    stream = torch.npu.current_stream().npu_stream
    _compile_all()
    with pytest.raises(ValueError, match=match):
        launch_argsarray_engine("kda_solve_wu_wide", 1, stream, bad_args, 0)


def test_launcher_cache_identity():
    from flash_kda_npu.ascendc import _SOURCES, _defines, ROOT, rtc_compile
    torch.npu.current_stream()
    rel, name, sizes = next(entry for entry in _SOURCES if entry[1] == "kda_solve_wu_wide")
    source = _defines() + (ROOT / rel).read_text(encoding="utf-8-sig")
    first = rtc_compile(source, name, sizes)
    assert rtc_compile(source, name, sizes) == first
    with pytest.raises(ValueError, match="different source/options/ABI"):
        rtc_compile(source, name, sizes[:-1])
    with pytest.raises(ValueError, match="different source/options/ABI"):
        rtc_compile(source + "\n// changed source", name, sizes)


def test_native_device_switch():
    if torch.npu.device_count() < 2:
        pytest.skip("requires two NPU devices")
    from flash_kda_npu.ascendc import CHUNK, _COMPILED, kda_fwd_ascendc
    previous = torch.npu.current_device()
    try:
        for device in (0, 1, 0):
            inputs = make_inputs(1, CHUNK * 3, 2, f"npu:{device}", gate_dtype=torch.float32)
            q, k, v, g, beta, a_log, bias = inputs
            # Call with a different current device; the function must restore it.
            torch.npu.set_device(1 - device)
            out, state = kda_fwd_ascendc(
                q, k, v, g, beta.float(), A_log=a_log, bias=bias, output_final_state=True)
            assert torch.npu.current_device() == 1 - device
            torch.npu.synchronize(device)
            expected_out, expected_state = recurrent_reference(inputs)
            assert_close(out, expected_out)
            assert_close(state, expected_state)
        assert {key[1] for key in _COMPILED}.issuperset({0, 1})
    finally:
        torch.npu.set_device(previous)


def test_native_rejects_cross_device_inputs():
    if torch.npu.device_count() < 2:
        pytest.skip("requires two NPU devices")
    from flash_kda_npu.ascendc import CHUNK, kda_fwd_ascendc
    q, k, v, g, beta, a_log, bias = make_inputs(1, CHUNK, 1, "npu:0", gate_dtype=torch.float32)
    with pytest.raises(ValueError, match="all inputs must be on q's NPU device"):
        kda_fwd_ascendc(q, k.to("npu:1"), v, g, beta.float(), A_log=a_log, bias=bias)


@pytest.mark.parametrize("batch,steps,heads", [(1, 129, 96), (2, 65, 3)])
@pytest.mark.parametrize("state_dtype", [torch.bfloat16, torch.float32])
def test_native_bf16_gate_matches_fp32(batch, steps, heads, state_dtype):
    """Moving the exact BF16->FP32 cast into UB must preserve all results."""
    from flash_kda_npu.ascendc import kda_fwd_ascendc
    q, k, v, g, beta, a_log, bias = make_inputs(batch, steps, heads, "npu")
    initial = make_state(batch, heads, 128, torch.float32, "npu")
    final = torch.empty_like(initial, dtype=state_dtype)
    out = torch.empty_like(q)
    results = []
    for gate in (g.float(), g):
        kda_fwd_ascendc(q, k, v, gate, beta.float(), A_log=a_log, bias=bias,
                            initial_state=initial, final_state=final,
                            output_final_state=True, out=out)
        torch.npu.synchronize()
        results.append((out.clone(), final.clone()))
    assert torch.equal(results[0][0], results[1][0])
    assert torch.equal(results[0][1], results[1][1])


@pytest.mark.parametrize("steps", [64, 129])
@pytest.mark.parametrize("state_dtype", [torch.bfloat16, torch.float32])
def test_native_graph_replay_changed_inputs(steps, state_dtype):
    from flash_kda_npu import capture_fwd
    q, k, v, g, beta, a_log, bias = make_inputs(1, steps, 4, "npu")
    initial = make_state(1, 4, 128, state_dtype, "npu")
    final = torch.empty_like(initial)
    out = torch.empty_like(q)
    args = (q, k, v, g, beta, 128 ** -0.5, out, a_log, bias, -5.0)
    captured = capture_fwd(*args, initial_state=initial, final_state=final)
    for _ in range(2):
        q.mul_(0.75)
        g.add_(0.25)
        beta.add_(0.125)
        initial.mul_(0.5)
        flash_kda.fwd(*args, initial_state=initial, final_state=final)
        torch.npu.synchronize()
        expected, state = out.clone(), final.clone()
        out.fill_(float("nan")); final.fill_(float("nan"))
        captured.replay()
        torch.npu.synchronize()
        assert captured.out is out and captured.final_state is final
        assert torch.equal(out, expected)
        assert torch.equal(final, state)


def test_native_graph_private_workspaces():
    """Two captures must retain independent scratch and caller buffers."""
    from flash_kda_npu import capture_fwd
    captures, expected = [], []
    for factor in (0.5, 1.5):
        q, k, v, g, beta, a_log, bias = make_inputs(1, 65, 4, "npu")
        q.mul_(factor)
        out = torch.empty_like(q)
        args = (q, k, v, g, beta, 128 ** -0.5, out, a_log, bias, -5.0)
        flash_kda.fwd(*args)
        torch.npu.synchronize()
        expected.append(out.clone())
        captures.append(capture_fwd(*args))
    streams = [torch.npu.Stream(), torch.npu.Stream()]
    for captured, stream in zip(captures, streams):
        with torch.npu.stream(stream):
            captured.replay()
    torch.npu.synchronize()
    for captured, result in zip(captures, expected):
        assert torch.equal(captured.out, result)
        assert captured.final_state is None


@pytest.mark.parametrize("state_dtype", [torch.bfloat16, torch.float32])
def test_native_auto_graph_cache(monkeypatch, state_dtype):
    from flash_kda_npu import clear_auto_graph_cache
    from flash_kda_npu import auto_graph
    monkeypatch.setenv('KDA_AUTO_GRAPH_WARMUP', '1')
    monkeypatch.setenv('KDA_AUTO_GRAPH_CACHE_SIZE', '2')
    clear_auto_graph_cache()
    q, k, v, g, beta, a_log, bias = make_inputs(1, 65, 4, 'npu')
    initial = make_state(1, 4, 128, state_dtype, 'npu')
    final = torch.empty_like(initial)
    out = torch.empty_like(q)
    for step in range(4):
        q.mul_(0.75); g.add_(0.125); initial.mul_(0.5)
        for mode in ('0', '1'):
            monkeypatch.setenv('KDA_AUTO_GRAPH', mode)
            flash_kda.fwd(q, k, v, g, beta, 128 ** -0.5, out, a_log, bias, -5.,
                          initial_state=initial, final_state=final)
            torch.npu.synchronize()
            if mode == '0':
                expected, state = out.clone(), final.clone()
            else:
                assert torch.equal(out, expected)
                assert torch.equal(final, state)
    cache = auto_graph._LOCAL.cache
    assert len(cache) == 1 and next(iter(cache.values()))['graph'] is not None
    # A changed scalar or output address creates a separate entry and LRU eviction.
    for scale in (0.25, 0.5):
        other = torch.empty_like(out)
        flash_kda.fwd(q, k, v, g, beta, scale, other, a_log, bias, -5.,
                      initial_state=initial, final_state=final)
        torch.npu.synchronize()
        assert len(cache) <= 2
    clear_auto_graph_cache()
    assert not cache


def test_native_auto_graph_streams_and_bypass(monkeypatch):
    from flash_kda_npu import auto_graph, clear_auto_graph_cache
    monkeypatch.setenv('KDA_AUTO_GRAPH', '1')
    monkeypatch.setenv('KDA_AUTO_GRAPH_WARMUP', '1')
    clear_auto_graph_cache()
    q, k, v, g, beta, a_log, bias = make_inputs(1, 64, 4, 'npu')
    args = (q, k, v, g, beta, 128 ** -0.5, torch.empty_like(q), a_log, bias, -5.)
    torch.npu.synchronize()
    streams = [torch.npu.Stream(), torch.npu.Stream()]
    for stream in streams:
        with torch.npu.stream(stream):
            for _ in range(3):
                flash_kda.fwd(*args)
        stream.synchronize()  # shared output is used sequentially
    assert len(auto_graph._LOCAL.cache) == 2
    assert all(e['graph'] is not None for e in auto_graph._LOCAL.cache.values())
    before = len(auto_graph._LOCAL.cache)
    monkeypatch.setenv('KDA_PROFILE', '1')
    flash_kda.fwd(*args)
    assert len(auto_graph._LOCAL.cache) == before
    monkeypatch.setenv('KDA_PROFILE', '0')
    cu = torch.tensor([0, 32, 64], device='npu', dtype=torch.int64)
    flash_kda.fwd(*args, cu_seqlens=cu)
    assert len(auto_graph._LOCAL.cache) == before
    clear_auto_graph_cache()
