import pytest
import torch

from numerics import assert_bfloat16_close, assert_matches_reference


def test_assert_bfloat16_close_accepts_adjacent_values():
    expected = torch.tensor([1.0, -1.0, 0.0, 4.0], dtype=torch.bfloat16)
    actual = torch.tensor(
        [1.0078125, -1.0078125, -0.0, 4.03125], dtype=torch.bfloat16
    )

    assert_bfloat16_close(actual, expected)


def test_assert_bfloat16_close_rejects_two_ulp_difference():
    expected = torch.tensor([1.0], dtype=torch.bfloat16)
    actual = torch.tensor([1.015625], dtype=torch.bfloat16)

    with pytest.raises(AssertionError, match="2 ULPs"):
        assert_bfloat16_close(actual, expected)


def test_assert_bfloat16_close_rejects_nonmatching_nonfinite_values():
    expected = torch.tensor([float("inf"), 0.0], dtype=torch.bfloat16)
    actual = torch.tensor([float("-inf"), float("nan")], dtype=torch.bfloat16)

    with pytest.raises(AssertionError, match="non-finite"):
        assert_bfloat16_close(actual, expected)


def test_assert_matches_reference_keeps_float32_comparison_exact():
    expected = torch.tensor([1.0], dtype=torch.float32)
    actual = torch.nextafter(expected, torch.tensor([float("inf")]))

    with pytest.raises(AssertionError, match="state mismatch"):
        assert_matches_reference(actual, expected, msg="state mismatch")
