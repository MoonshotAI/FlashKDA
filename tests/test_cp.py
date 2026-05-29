#!/usr/bin/env python3
"""Test that fwd_cp produces results matching fwd (no CP) within bf16 tolerance."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from flash_kda import fwd, fwd_cp


def make_inputs(B, T, H, D, device="cuda"):
    q = torch.randn(B, T, H, D, device=device, dtype=torch.bfloat16)
    k = torch.randn(B, T, H, D, device=device, dtype=torch.bfloat16)
    v = torch.randn(B, T, H, D, device=device, dtype=torch.bfloat16)
    g = torch.randn(B, T, H, D, device=device, dtype=torch.bfloat16)
    beta = torch.randn(B, T, H, device=device, dtype=torch.bfloat16)
    A_log = torch.randn(H, device=device, dtype=torch.float32).abs().neg()
    dt_bias = torch.randn(H, D, device=device, dtype=torch.float32)
    scale = D ** -0.5
    lower_bound = -5.0
    return q, k, v, g, beta, scale, A_log, dt_bias, lower_bound


def test_cp_correctness_single_seq():
    """B=1, single long sequence — CP should engage and produce matching results."""
    B, T, H, D = 1, 8192, 16, 128
    q, k, v, g, beta, scale, A_log, dt_bias, lower_bound = make_inputs(B, T, H, D)

    out_ref = torch.empty_like(q)
    fwd(q, k, v, g, beta, scale, out_ref, A_log, dt_bias, lower_bound)

    out_cp = torch.empty_like(q)
    fwd_cp(q, k, v, g, beta, scale, out_cp, A_log, dt_bias, lower_bound, auto_cp=True)

    diff = (out_ref.float() - out_cp.float()).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    print(f"[single_seq] T={T}, H={H}: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")

    if torch.allclose(out_ref, out_cp, atol=0, rtol=0):
        print("  CP fallback or bit-identical — PASS")
        return True

    ok = max_diff < 0.1
    print(f"  CP engaged — {'PASS' if ok else 'FAIL'}")
    return ok


def test_cp_correctness_long_seq():
    """B=1, very long sequence — most likely to benefit from CP."""
    B, T, H, D = 1, 65536, 16, 128
    q, k, v, g, beta, scale, A_log, dt_bias, lower_bound = make_inputs(B, T, H, D)

    A_log = -torch.ones(H, device="cuda", dtype=torch.float32) * 0.1

    out_ref = torch.empty_like(q)
    fwd(q, k, v, g, beta, scale, out_ref, A_log, dt_bias, lower_bound)

    out_cp = torch.empty_like(q)
    fwd_cp(q, k, v, g, beta, scale, out_cp, A_log, dt_bias, lower_bound, auto_cp=True)

    diff = (out_ref.float() - out_cp.float()).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    rel_err = diff.sum() / out_ref.float().abs().sum()
    print(f"[long_seq] T={T}, H={H}: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}, rel_err={rel_err:.6f}")

    if torch.allclose(out_ref, out_cp, atol=0, rtol=0):
        print("  Bit-identical — PASS")
        return True

    ok = rel_err < 0.01
    print(f"  CP engaged — {'PASS' if ok else 'FAIL'}")
    return ok


def test_cp_final_state():
    """Verify final_state is correctly extracted from last sub-segment."""
    B, T, H, D = 1, 8192, 16, 128
    q, k, v, g, beta, scale, A_log, dt_bias, lower_bound = make_inputs(B, T, H, D)
    A_log = -torch.ones(H, device="cuda", dtype=torch.float32) * 0.1

    out_ref = torch.empty_like(q)
    fs_ref = torch.empty(B, H, D, D, dtype=torch.float32, device="cuda")
    fwd(q, k, v, g, beta, scale, out_ref, A_log, dt_bias, lower_bound, final_state=fs_ref)

    out_cp = torch.empty_like(q)
    fs_cp = torch.empty(B, H, D, D, dtype=torch.float32, device="cuda")
    fwd_cp(q, k, v, g, beta, scale, out_cp, A_log, dt_bias, lower_bound,
            final_state=fs_cp, auto_cp=True)

    diff = (fs_ref - fs_cp).abs()
    max_diff = diff.max().item()
    print(f"[final_state] max_diff={max_diff:.6f}")

    if torch.allclose(fs_ref, fs_cp, atol=0, rtol=0):
        print("  Bit-identical — PASS")
        return True

    ok = max_diff < 1.0
    print(f"  CP engaged — {'PASS' if ok else 'FAIL'}")
    return ok


def test_cp_with_initial_state():
    """Verify initial_state is correctly passed to first sub-segment."""
    B, T, H, D = 1, 8192, 16, 128
    q, k, v, g, beta, scale, A_log, dt_bias, lower_bound = make_inputs(B, T, H, D)
    A_log = -torch.ones(H, device="cuda", dtype=torch.float32) * 0.1
    h0 = torch.randn(B, H, D, D, dtype=torch.float32, device="cuda") * 0.01

    out_ref = torch.empty_like(q)
    fwd(q, k, v, g, beta, scale, out_ref, A_log, dt_bias, lower_bound, initial_state=h0)

    out_cp = torch.empty_like(q)
    fwd_cp(q, k, v, g, beta, scale, out_cp, A_log, dt_bias, lower_bound,
            initial_state=h0, auto_cp=True)

    diff = (out_ref.float() - out_cp.float()).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    print(f"[with_h0] max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")

    if torch.allclose(out_ref, out_cp, atol=0, rtol=0):
        print("  Bit-identical — PASS")
        return True

    ok = max_diff < 0.1
    print(f"  CP engaged — {'PASS' if ok else 'FAIL'}")
    return ok


def test_cp_disabled():
    """auto_cp=False should produce identical results to fwd()."""
    B, T, H, D = 1, 8192, 16, 128
    q, k, v, g, beta, scale, A_log, dt_bias, lower_bound = make_inputs(B, T, H, D)

    out_ref = torch.empty_like(q)
    fwd(q, k, v, g, beta, scale, out_ref, A_log, dt_bias, lower_bound)

    out_cp = torch.empty_like(q)
    fwd_cp(q, k, v, g, beta, scale, out_cp, A_log, dt_bias, lower_bound, auto_cp=False)

    assert torch.allclose(out_ref, out_cp, atol=0, rtol=0), "auto_cp=False should be identical to fwd()"
    print("[disabled] PASS — auto_cp=False produces identical output")
    return True


if __name__ == "__main__":
    torch.manual_seed(42)
    results = []
    results.append(test_cp_disabled())
    results.append(test_cp_correctness_single_seq())
    results.append(test_cp_correctness_long_seq())
    results.append(test_cp_final_state())
    results.append(test_cp_with_initial_state())

    print(f"\n{'='*40}")
    print(f"Results: {sum(results)}/{len(results)} passed")
    if not all(results):
        sys.exit(1)
