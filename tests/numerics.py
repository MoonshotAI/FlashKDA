import torch


_COMPARISON_CHUNK_SIZE = 1 << 20


def _ordered_bfloat16(values):
    bits = values.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
    magnitude = bits & 0x7FFF
    return torch.where(bits & 0x8000 != 0, 0x8000 - magnitude, 0x8000 + magnitude)


def assert_bfloat16_close(actual, expected, *, max_ulps=1, msg=None):
    """Assert that BF16 tensors differ by at most ``max_ulps`` representable values."""
    if actual.shape != expected.shape:
        raise AssertionError(f"shape mismatch: {actual.shape} != {expected.shape}")
    if actual.dtype != torch.bfloat16 or expected.dtype != torch.bfloat16:
        raise AssertionError(
            f"expected bfloat16 tensors, got {actual.dtype} and {expected.dtype}"
        )
    if max_ulps < 0:
        raise ValueError("max_ulps must be non-negative")

    actual_flat = actual.reshape(-1)
    expected_flat = expected.reshape(-1)
    max_distance = torch.zeros((), dtype=torch.int32, device=actual.device)
    finite_mismatch_count = torch.zeros((), dtype=torch.int64, device=actual.device)
    nonfinite_mismatch_count = torch.zeros((), dtype=torch.int64, device=actual.device)

    for start in range(0, actual_flat.numel(), _COMPARISON_CHUNK_SIZE):
        end = start + _COMPARISON_CHUNK_SIZE
        actual_chunk = actual_flat[start:end]
        expected_chunk = expected_flat[start:end]
        equal = actual_chunk == expected_chunk
        finite_pair = torch.isfinite(actual_chunk) & torch.isfinite(expected_chunk)
        distance = (
            _ordered_bfloat16(actual_chunk) - _ordered_bfloat16(expected_chunk)
        ).abs()
        finite_mismatch = finite_pair & (distance > max_ulps)

        max_distance = torch.maximum(
            max_distance,
            distance.masked_fill(~finite_pair, 0).max(),
        )
        finite_mismatch_count += finite_mismatch.sum()
        nonfinite_mismatch_count += (~equal & ~finite_pair).sum()

    finite_mismatches = finite_mismatch_count.item()
    nonfinite_mismatches = nonfinite_mismatch_count.item()
    if finite_mismatches or nonfinite_mismatches:
        details = (
            f"bfloat16 tensors differ by up to {max_distance.item()} ULPs "
            f"(allowed {max_ulps}); {finite_mismatches} finite and "
            f"{nonfinite_mismatches} non-finite values are outside the tolerance"
        )
        raise AssertionError(f"{msg}: {details}" if msg else details)


def assert_matches_reference(actual, expected, *, msg):
    """Use a one-ULP BF16 comparison while keeping other dtypes bitwise exact."""
    if actual.dtype == torch.bfloat16 and expected.dtype == torch.bfloat16:
        assert_bfloat16_close(actual, expected, msg=msg)
    elif not torch.equal(actual, expected):
        raise AssertionError(msg)
