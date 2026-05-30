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
    A_log = torch.randn(H, device=device, dtype=torch.float32).abs().neg()  # negative
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

    # If CP didn't engage (fallback), outputs are identical
    if torch.allclose(out_ref, out_cp, atol=0, rtol=0):
        print("  CP did not engage (fallback to standard fwd) — outputs identical")
        return True

    # If CP engaged, expect small numerical diff due to bf16 state accumulation order
    ok = max_diff < 0.1
    print(f"  CP engaged — {'PASS' if ok else 'FAIL'}")
    return ok


def test_cp_correctness_long_seq():
    """B=1, very long sequence — most likely to benefit from CP."""
    B, T, H, D = 1, 65536, 16, 128
    q, k, v, g, beta, scale, A_log, dt_bias, lower_bound = make_inputs(B, T, H, D)

    # Make A_log more negative to ensure warmup converges
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
        print("  CP did not engage — outputs identical")
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
        print("  CP did not engage — final states identical")
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
        print("  CP did not engage — outputs identical")
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


def test_state_only_vs_fwd():
    """Verify state_only produces same final_state as fwd() for single segment."""
    from flash_kda import state_only

    B, T, H, D = 1, 8192, 16, 128
    q, k, v, g, beta, scale, A_log, dt_bias, lower_bound = make_inputs(B, T, H, D)
    A_log = -torch.ones(H, device="cuda", dtype=torch.float32) * 0.1

    # Reference: full fwd with final_state
    out_ref = torch.empty_like(q)
    fs_ref = torch.empty(B, H, D, D, dtype=torch.float32, device="cuda")
    fwd(q, k, v, g, beta, scale, out_ref, A_log, dt_bias, lower_bound, final_state=fs_ref)

    # state_only: process all chunks
    cu_seqlens = torch.tensor([0, T], dtype=torch.int64, device="cuda")
    num_chunks = (T + 15) // 16
    num_warmup = torch.tensor([num_chunks], dtype=torch.int32, device="cuda")
    fs_so = state_only(k, v, g, beta, A_log, dt_bias, lower_bound, cu_seqlens, num_warmup)

    diff = (fs_ref - fs_so).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    print(f"[state_only_vs_fwd] max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")

    ok = max_diff < 1e-4
    print(f"  {'PASS' if ok else 'FAIL'}")
    return ok


def test_state_only_warmup_subset():
    """Verify state_only with fewer warmup chunks gives reasonable approximation."""
    from flash_kda import state_only

    B, T, H, D = 1, 8192, 16, 128
    q, k, v, g, beta, scale, A_log, dt_bias, lower_bound = make_inputs(B, T, H, D)
    A_log = -torch.ones(H, device="cuda", dtype=torch.float32) * 0.5

    # Reference: full state
    out_ref = torch.empty_like(q)
    fs_ref = torch.empty(B, H, D, D, dtype=torch.float32, device="cuda")
    fwd(q, k, v, g, beta, scale, out_ref, A_log, dt_bias, lower_bound, final_state=fs_ref)

    # state_only with only last 32 chunks (out of 512)
    cu_seqlens = torch.tensor([0, T], dtype=torch.int64, device="cuda")
    num_warmup = torch.tensor([32], dtype=torch.int32, device="cuda")
    fs_so = state_only(k, v, g, beta, A_log, dt_bias, lower_bound, cu_seqlens, num_warmup)

    diff = (fs_ref - fs_so).abs()
    max_diff = diff.max().item()
    rel_err = diff.sum() / fs_ref.abs().sum()
    print(f"[state_only_warmup_subset] max_diff={max_diff:.6f}, rel_err={rel_err:.6f}")

    # With A_log=-0.5 and 32 chunks (512 tokens), decay = exp(512 * -0.5) ≈ 0
    # So warmup should converge and produce close result
    ok = rel_err < 0.05
    print(f"  {'PASS' if ok else 'FAIL'}")
    return ok


def test_state_only_multi_segment():
    """Verify state_only works with multiple segments (varlen)."""
    from flash_kda import state_only

    H, D = 16, 128
    seqlens = [2048, 4096, 1024]
    T_total = sum(seqlens)
    B = 1

    q = torch.randn(B, T_total, H, D, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(B, T_total, H, D, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(B, T_total, H, D, device="cuda", dtype=torch.bfloat16)
    g = torch.randn(B, T_total, H, D, device="cuda", dtype=torch.bfloat16)
    beta = torch.randn(B, T_total, H, device="cuda", dtype=torch.bfloat16)
    A_log = -torch.ones(H, device="cuda", dtype=torch.float32) * 0.1
    dt_bias = torch.randn(H, D, device="cuda", dtype=torch.float32)
    lower_bound = -5.0
    scale = D ** -0.5

    cu_seqlens = torch.tensor([0] + [sum(seqlens[:i+1]) for i in range(len(seqlens))],
                              dtype=torch.int64, device="cuda")
    N = len(seqlens)

    # Reference: fwd with varlen
    out_ref = torch.empty_like(q)
    fs_ref = torch.empty(N, H, D, D, dtype=torch.float32, device="cuda")
    fwd(q, k, v, g, beta, scale, out_ref, A_log, dt_bias, lower_bound,
        final_state=fs_ref, cu_seqlens=cu_seqlens)

    # state_only: all chunks per segment
    num_warmup_vals = [(s + 15) // 16 for s in seqlens]
    num_warmup = torch.tensor(num_warmup_vals, dtype=torch.int32, device="cuda")
    fs_so = state_only(k, v, g, beta, A_log, dt_bias, lower_bound, cu_seqlens, num_warmup)

    diff = (fs_ref - fs_so).abs()
    max_diff = diff.max().item()
    print(f"[state_only_multi_seg] max_diff={max_diff:.6f}")

    ok = max_diff < 1e-4
    print(f"  {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    torch.manual_seed(42)
    results = []
    results.append(test_cp_disabled())
    results.append(test_cp_correctness_single_seq())
    results.append(test_cp_correctness_long_seq())
    results.append(test_cp_final_state())
    results.append(test_cp_with_initial_state())
    results.append(test_state_only_vs_fwd())
    results.append(test_state_only_warmup_subset())
    results.append(test_state_only_multi_segment())

    print(f"\n{'='*40}")
    print(f"Results: {sum(results)}/{len(results)} passed")
    if not all(results):
        sys.exit(1)
