"""Mathematical validation for the exact KDA CP transition prototype."""

import importlib.util
import sys
from pathlib import Path

import torch
import torch.nn.functional as F


_CP_PATH = Path(__file__).parents[1] / "flash_kda" / "cp.py"
_SPEC = importlib.util.spec_from_file_location("flash_kda_cp", _CP_PATH)
assert _SPEC is not None and _SPEC.loader is not None
cp = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = cp
_SPEC.loader.exec_module(cp)


def _chunk(
    generator: torch.Generator,
    tokens: int,
    heads: int,
    key_dim: int,
    value_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    key = F.normalize(
        torch.randn((tokens, heads, key_dim), generator=generator, dtype=torch.float64),
        p=2,
        dim=-1,
    )
    value = 0.1 * torch.randn(
        (tokens, heads, value_dim), generator=generator, dtype=torch.float64
    )
    log_decay = -0.1 * torch.rand(
        (tokens, heads, key_dim), generator=generator, dtype=torch.float64
    )
    beta = 0.1 + 0.8 * torch.rand(
        (tokens, heads), generator=generator, dtype=torch.float64
    )
    return key, value, log_decay, beta


def _concatenate_chunks(
    chunks: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return tuple(torch.cat([chunk[index] for chunk in chunks], dim=0) for index in range(4))


def _assert_close(actual: torch.Tensor, expected: torch.Tensor) -> None:
    torch.testing.assert_close(actual, expected, rtol=1e-11, atol=1e-11)


def test_transition_matches_direct_kda_block_recurrence():
    generator = torch.Generator().manual_seed(20260814)
    heads, key_dim, value_dim = 2, 9, 7
    state = 0.1 * torch.randn(
        (heads, value_dim, key_dim), generator=generator, dtype=torch.float64
    )
    chunk = _chunk(generator, 13, heads, key_dim, value_dim)

    transition = cp.kda_chunk_transition(*chunk)
    _assert_close(cp.apply_transition(state, transition), cp.kda_chunk_update(state, *chunk))


def test_transition_composition_is_associative():
    generator = torch.Generator().manual_seed(20260815)
    chunks = [_chunk(generator, length, 2, 8, 6) for length in (16, 7, 11)]
    transitions = [cp.kda_chunk_transition(*chunk) for chunk in chunks]

    left = cp.compose_transitions(
        cp.compose_transitions(transitions[0], transitions[1]), transitions[2]
    )
    right = cp.compose_transitions(
        transitions[0], cp.compose_transitions(transitions[1], transitions[2])
    )

    _assert_close(left.matrix, right.matrix)
    _assert_close(left.bias, right.bias)


def test_segment_low_rank_accumulation_matches_dense_chunk_composition():
    generator = torch.Generator().manual_seed(20260816)
    chunks = [_chunk(generator, length, 2, 10, 7) for length in (16, 16, 16, 3)]
    dense = cp.compose_transition_sequence(
        [cp.kda_chunk_transition(*chunk) for chunk in chunks]
    )
    low_rank = cp.kda_segment_transition(*_concatenate_chunks(chunks))

    _assert_close(low_rank.matrix, dense.matrix)
    _assert_close(low_rank.bias, dense.bias)


def test_two_rank_cp_exclusive_prefix_matches_serial_recurrence():
    """Two logical CP ranks receive the exact start state for their segment."""
    generator = torch.Generator().manual_seed(20260817)
    heads, key_dim, value_dim = 2, 10, 5
    initial_state = 0.1 * torch.randn(
        (heads, value_dim, key_dim), generator=generator, dtype=torch.float64
    )
    chunks = [
        _chunk(generator, length, heads, key_dim, value_dim)
        for length in (16, 9, 16, 5)
    ]
    transitions = [cp.kda_chunk_transition(*chunk) for chunk in chunks]

    # Rank 0 owns chunks [0, 1]; rank 1 owns chunks [2, 3].
    local_transitions = [
        cp.compose_transition_sequence(transitions[:2]),
        cp.compose_transition_sequence(transitions[2:]),
    ]
    prefixes = cp.exclusive_prefix_transitions(local_transitions)
    rank0_start = cp.apply_transition(initial_state, prefixes[0])
    rank1_start = cp.apply_transition(initial_state, prefixes[1])

    serial = initial_state
    serial_after_rank0 = initial_state
    for index, chunk in enumerate(chunks):
        serial = cp.kda_chunk_update(serial, *chunk)
        if index == 1:
            serial_after_rank0 = serial

    rank0_end = cp.apply_transition(rank0_start, local_transitions[0])
    rank1_end = cp.apply_transition(rank1_start, local_transitions[1])
    full_transition = cp.compose_transition_sequence(transitions)

    _assert_close(rank0_end, serial_after_rank0)
    _assert_close(rank1_start, serial_after_rank0)
    _assert_close(rank1_end, serial)
    _assert_close(cp.apply_transition(initial_state, full_transition), serial)


def test_tree_prefix_scan_matches_serial_for_non_power_of_two():
    generator = torch.Generator().manual_seed(20260818)
    heads, key_dim, value_dim = 2, 9, 6
    initial_state = 0.1 * torch.randn(
        (heads, value_dim, key_dim), generator=generator, dtype=torch.float64
    )
    segments = [
        _concatenate_chunks(
            [_chunk(generator, length, heads, key_dim, value_dim) for length in lengths]
        )
        for lengths in ((16,), (5, 13), (9,), (16, 2), (7,))
    ]
    transitions = [cp.kda_segment_transition(*segment) for segment in segments]
    serial_prefixes = cp.exclusive_prefix_transitions(transitions)
    tree_prefixes = cp.exclusive_prefix_scan(transitions)

    assert tree_prefixes.matrix.shape[0] == 5
    for index, serial_prefix in enumerate(serial_prefixes):
        _assert_close(tree_prefixes.matrix[index], serial_prefix.matrix)
        _assert_close(tree_prefixes.bias[index], serial_prefix.bias)

    starts = cp.segment_start_states(initial_state, transitions)
    for index, serial_prefix in enumerate(serial_prefixes):
        _assert_close(starts[index], cp.apply_transition(initial_state, serial_prefix))


def test_cuda_triton_segment_matches_reference_with_tail_chunk():
    if not torch.cuda.is_available():
        return

    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(20260819)
    tokens, heads, key_dim, value_dim = 35, 2, 128, 128
    key = F.normalize(
        torch.randn(
            (tokens, heads, key_dim),
            generator=generator,
            device=device,
            dtype=torch.float32,
        ),
        p=2,
        dim=-1,
    ).to(torch.bfloat16)
    value = torch.randn(
        (tokens, heads, value_dim),
        generator=generator,
        device=device,
        dtype=torch.bfloat16,
    )
    log_decay = -0.1 * torch.rand(
        (tokens, heads, key_dim), generator=generator, device=device
    )
    beta = 0.1 + 0.8 * torch.rand(
        (tokens, heads), generator=generator, device=device
    )

    expected = cp.kda_segment_transition_reference(key, value, log_decay, beta)
    actual = cp.kda_segment_transition_triton(key, value, log_decay, beta)
    torch.testing.assert_close(actual.matrix, expected.matrix, rtol=3e-3, atol=3e-4)
    torch.testing.assert_close(actual.bias, expected.bias, rtol=3e-3, atol=3e-4)


def test_cuda_bf16_segment_scan_reconstructs_serial_state():
    """Exercise segment accumulation and a 3-rank scan at FlashKDA D=128."""
    if not torch.cuda.is_available():
        return

    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(20260819)
    heads, key_dim, value_dim = 4, 128, 128
    initial_state = (
        0.05
        * torch.randn(
            (heads, value_dim, key_dim),
            generator=generator,
            device=device,
            dtype=torch.float32,
        )
    ).to(torch.bfloat16)
    chunks = []
    for length in (16, 16, 9, 16):
        key = F.normalize(
            torch.randn(
                (length, heads, key_dim),
                generator=generator,
                device=device,
                dtype=torch.float32,
            ),
            p=2,
            dim=-1,
        ).to(torch.bfloat16)
        value = torch.randn(
            (length, heads, value_dim),
            generator=generator,
            device=device,
            dtype=torch.bfloat16,
        )
        log_decay = -0.1 * torch.rand(
            (length, heads, key_dim),
            generator=generator,
            device=device,
            dtype=torch.float32,
        )
        beta = 0.1 + 0.8 * torch.rand(
            (length, heads), generator=generator, device=device, dtype=torch.float32
        )
        chunks.append((key, value, log_decay, beta))

    segment_chunks = (chunks[:1], chunks[1:3], chunks[3:])
    transitions = [
        cp.kda_segment_transition(*_concatenate_chunks(segment))
        for segment in segment_chunks
    ]
    starts = cp.segment_start_states(initial_state, transitions)

    serial = initial_state
    expected_starts = []
    for segment in segment_chunks:
        expected_starts.append(serial.to(torch.float32))
        for chunk in segment:
            serial = cp.kda_chunk_update(serial, *chunk)

    for actual, expected in zip(starts, expected_starts):
        torch.testing.assert_close(actual, expected, rtol=5e-4, atol=5e-4)

    final_from_scan = cp.apply_transition(starts[-1], transitions[-1])
    torch.testing.assert_close(final_from_scan, serial, rtol=5e-4, atol=5e-4)
