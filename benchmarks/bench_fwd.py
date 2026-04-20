import torch
import flash_kda
import torch.nn.functional as F
import math
from fla.ops.kda import chunk_kda

def bench_fn(fn, warmup, iters, repeats):
    for _ in range(max(warmup, 1)):
        fn()
    torch.cuda.synchronize()

    all_ms = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        for i in range(iters):
            starts[i].record()
            fn()
            ends[i].record()
        torch.cuda.synchronize()
        all_ms.extend([s.elapsed_time(e) for s, e in zip(starts, ends)])

    xs = sorted(float(x) for x in all_ms)
    n = len(xs)
    mean = sum(xs) / n if n else float("nan")
    mn = xs[0] if xs else float("nan")
    mx = xs[-1] if xs else float("nan")
    return mean, mn, mx


def run_case(seq_lens, H, D, warmup, iters, repeats):
    device = torch.device("cuda")
    LOWER_BOUND = -5.0
    scale_float = 1.0 / math.sqrt(D)

    varlen = len(seq_lens) > 1
    T_total = sum(seq_lens)
    N = len(seq_lens)

    if varlen:
        cu_seqlens = torch.tensor(
            [0] + list(torch.cumsum(torch.tensor(seq_lens), dim=0).tolist()),
            dtype=torch.long, device=device,
        )
        print(f"varlen shape=[{T_total},{H},{D}] seq_lens={seq_lens} warmup={warmup} iters={iters} repeats={repeats}")
        extra = {"cu_seqlens": cu_seqlens}
    else:
        print(f"shape=[{T_total},{H},{D}] warmup={warmup} iters={iters} repeats={repeats}")
        extra = {}

    q = F.normalize(torch.randn((1, T_total, H, D), dtype=torch.float32, device=device), p=2, dim=-1).to(torch.bfloat16)
    k = F.normalize(torch.randn((1, T_total, H, D), dtype=torch.float32, device=device), p=2, dim=-1).to(torch.bfloat16)
    v = torch.randn((1, T_total, H, D), dtype=torch.bfloat16, device=device)
    g = torch.randn((1, T_total, H, D), dtype=torch.bfloat16, device=device)
    beta = torch.randn((1, T_total, H), dtype=torch.bfloat16, device=device)
    A_log = torch.rand(H, dtype=torch.float32, device=device)
    dt_bias = torch.rand(H, D, dtype=torch.float32, device=device)

    initial_state = torch.arange(N * H * D * D, dtype=torch.float32, device=device).reshape(N, H, D, D).to(torch.bfloat16)
    final_state = torch.zeros_like(initial_state)
    out = torch.zeros_like(q)
    scale = scale_float

    # --- flash_kda: bf16 state ---
    def run_flash_kda():
        flash_kda.fwd(q, k, v, g, beta, scale, out,
                      A_log=A_log, dt_bias=dt_bias, lower_bound=LOWER_BOUND,
                      initial_state=initial_state, final_state=final_state, **extra)

    mean, mn, mx = bench_fn(run_flash_kda, warmup, iters, repeats)
    print(f"  flash_kda (bf16 state) : mean={mean:.4f} ms, min={mn:.4f} ms, max={mx:.4f} ms")

    # --- flash_kda: no state ---
    def run_flash_kda_no_state():
        flash_kda.fwd(q, k, v, g, beta, scale, out,
                      A_log=A_log, dt_bias=dt_bias, lower_bound=LOWER_BOUND, **extra)

    mean, mn, mx = bench_fn(run_flash_kda_no_state, warmup, iters, repeats)
    print(f"  flash_kda (no state)   : mean={mean:.4f} ms, min={mn:.4f} ms, max={mx:.4f} ms")

    # --- flash_kda: fp32 state ---
    initial_state_fp32 = initial_state.float()
    final_state_fp32 = torch.zeros_like(initial_state_fp32)

    def run_flash_kda_fp32():
        flash_kda.fwd(q, k, v, g, beta, scale, out,
                      A_log=A_log, dt_bias=dt_bias, lower_bound=LOWER_BOUND,
                      initial_state=initial_state_fp32, final_state=final_state_fp32, **extra)

    mean, mn, mx = bench_fn(run_flash_kda_fp32, warmup, iters, repeats)
    print(f"  flash_kda (fp32 state) : mean={mean:.4f} ms, min={mn:.4f} ms, max={mx:.4f} ms")

    # --- chunk_kda ---
    h0_ck = initial_state.float()

    def run_chunk_kda():
        chunk_kda(
            q=q, k=k, v=v, g=g, beta=beta,
            scale=scale_float,
            initial_state=h0_ck,
            output_final_state=True,
            use_gate_in_kernel=True,
            use_qk_l2norm_in_kernel=True,
            use_beta_sigmoid_in_kernel=True,
            A_log=A_log, dt_bias=dt_bias,
            lower_bound=LOWER_BOUND,
            transpose_state_layout=True,
            **extra,
        )

    mean, mn, mx = bench_fn(run_chunk_kda, warmup, iters, repeats)
    print(f"  chunk_kda : mean={mean:.4f} ms, min={mn:.4f} ms, max={mx:.4f} ms")


FIXED_CASES = [
    [8192],
]

VARLEN_CASES = [
    [1300, 547, 2048, 963, 271, 3063],
    [1024] * 8,
]


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--warmup", type=int, default=30)
    p.add_argument("--iters", type=int, default=200)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--mode", choices=["fixed", "varlen", "all"], default="all")
    p.add_argument("--H", type=int, default=96)
    p.add_argument("--D", type=int, default=128)
    args = p.parse_args()

    cases = []
    if args.mode in ("fixed", "all"):
        cases.extend(FIXED_CASES)
    if args.mode in ("varlen", "all"):
        cases.extend(VARLEN_CASES)

    for seq_lens in cases:
        run_case(seq_lens, args.H, args.D, args.warmup, args.iters, args.repeats)


if __name__ == "__main__":
    main()
