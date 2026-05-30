#!/usr/bin/env python3
"""Benchmark: no CP vs new CP (warmup + correct with CUDA scan)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from flash_kda import fwd, state_only
from flash_kda.cp import _calc_cp_seqs, get_warmup_chunks_cuda, correct_initial_states, CHUNK_SIZE


def bench(fn, warmup=10, iters=50):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def fwd_cp_new(q, k, v, g, beta, scale, out, A_log, dt_bias, lower_bound,
               cp_cu_seqlens, seq_map_r2c):
    """New CP: CUDA get_warmup_chunks + state_only + correct + single fwd."""
    cp_N = cp_cu_seqlens.numel() - 1

    num_warmup, fallback_mask = get_warmup_chunks_cuda(
        g, A_log, dt_bias, lower_bound, cp_cu_seqlens, CHUNK_SIZE
    )

    need_mt = fallback_mask.any().item()

    if need_mt:
        ht_buffer, mt_buffer = state_only(k, v, g, beta, A_log, dt_bias, lower_bound,
                                          cp_cu_seqlens, num_warmup, calc_mt=True)
    else:
        ht_buffer = state_only(k, v, g, beta, A_log, dt_bias, lower_bound,
                               cp_cu_seqlens, num_warmup, calc_mt=False)
        mt_buffer = None

    cp_h0 = correct_initial_states(None, ht_buffer, mt_buffer, fallback_mask, seq_map_r2c)

    fwd(q, k, v, g, beta, scale, out, A_log, dt_bias, lower_bound,
        initial_state=cp_h0, cu_seqlens=cp_cu_seqlens)


def run_bench(T, H=16, D=128, lower_bound=-5.0):
    B = 1
    device = "cuda"
    q = torch.randn(B, T, H, D, device=device, dtype=torch.bfloat16)
    k = torch.randn(B, T, H, D, device=device, dtype=torch.bfloat16)
    v = torch.randn(B, T, H, D, device=device, dtype=torch.bfloat16)
    g = torch.randn(B, T, H, D, device=device, dtype=torch.bfloat16)
    beta = torch.randn(B, T, H, device=device, dtype=torch.bfloat16)
    A_log = -torch.ones(H, device=device, dtype=torch.float32) * 0.1
    dt_bias = torch.randn(H, D, device=device, dtype=torch.float32)
    scale = D ** -0.5
    out = torch.empty_like(q)

    cu_seqlens = torch.tensor([0, T], dtype=torch.int64, device=device)
    use_cp, cp_cu_seqlens, seq_map_r2c, seq_map_c2r = _calc_cp_seqs(cu_seqlens, H)

    if not use_cp:
        print(f"  T={T:>6d}, H={H:>2d}: CP not engaged (sequence too short or H too large)")
        t_no_cp = bench(lambda: fwd(q, k, v, g, beta, scale, out, A_log, dt_bias, lower_bound))
        print(f"    no_cp={t_no_cp:.3f}ms")
        return

    cp_N = cp_cu_seqlens.numel() - 1

    def fn_no_cp():
        fwd(q, k, v, g, beta, scale, out, A_log, dt_bias, lower_bound)

    def fn_new_cp():
        fwd_cp_new(q, k, v, g, beta, scale, out, A_log, dt_bias, lower_bound,
                   cp_cu_seqlens, seq_map_r2c)

    t_no_cp = bench(fn_no_cp)
    t_new_cp = bench(fn_new_cp)

    print(f"  T={T:>6d}, H={H:>2d}, cp_N={cp_N:>3d} | "
          f"no_cp={t_no_cp:.3f}ms | "
          f"new_cp={t_new_cp:.3f}ms ({t_no_cp/t_new_cp:.2f}x)")


print("=" * 70)
print("FlashKDA CP Benchmark: no_cp vs new_cp (warmup + CUDA scan + correct)")
print("=" * 70)

torch.manual_seed(42)

print("\n--- Varying T (H=16, lb=-5.0) ---")
for T in [4096, 8192, 16384, 32768, 65536]:
    run_bench(T, H=16)

print("\n--- Varying H (T=65536, lb=-5.0) ---")
for H in [4, 8, 16, 32]:
    run_bench(65536, H=H)

print("\n--- Weak decay (H=16, lb=-1.0) ---")
for T in [4096, 8192, 16384, 32768, 65536]:
    run_bench(T, H=16, lower_bound=-1.0)

print("\n--- Very weak decay (H=16, lb=-0.5) ---")
for T in [4096, 8192, 16384, 32768, 65536]:
    run_bench(T, H=16, lower_bound=-0.5)
