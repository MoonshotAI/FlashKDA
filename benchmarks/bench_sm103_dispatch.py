"""Reproduce the three B300 K2 workloads through the public flash_kda.fwd API."""

import argparse
import os
import statistics


def bench_fn(fn, warmup, iters):
    import torch

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for start, end in zip(starts, ends):
        start.record()
        fn()
        end.record()
    torch.cuda.synchronize()
    times_us = [start.elapsed_time(end) * 1000 for start, end in zip(starts, ends)]
    return statistics.mean(times_us), statistics.median(times_us)


def run_case(B, T, expected_auto, warmup, iters, seed):
    import torch
    import torch.nn.functional as F
    import flash_kda

    H, D = 64, 128
    torch.manual_seed(seed)
    shape = (B, T, H, D)
    q = F.normalize(torch.randn(shape, dtype=torch.float32, device="cuda"), dim=-1).to(torch.bfloat16)
    k = F.normalize(torch.randn(shape, dtype=torch.float32, device="cuda"), dim=-1).to(torch.bfloat16)
    v = torch.randn(shape, dtype=torch.bfloat16, device="cuda")
    g = torch.randn_like(v)
    beta = torch.randn((B, T, H), dtype=torch.bfloat16, device="cuda")
    A_log = torch.rand(H, dtype=torch.float32, device="cuda")
    dt_bias = torch.rand(H, D, dtype=torch.float32, device="cuda")
    initial = torch.randn((B, H, D, D), dtype=torch.bfloat16, device="cuda")
    initial_copy = initial.clone()
    print(f"\nB={B} T={T} H={H} D={D} chunks={(T + 15) // 16} expected_auto={expected_auto}")

    previous = os.environ.get("FLASH_KDA_K2_IMPL")
    try:
        for mode in ("baseline", "v1a", "v1ae", "auto"):
            os.environ["FLASH_KDA_K2_IMPL"] = mode
            out = torch.full_like(q, float("nan"))
            final = torch.full_like(initial, float("nan"))

            def run():
                # Include the public wrapper's workspace allocation in each call.
                flash_kda.fwd(q, k, v, g, beta, D ** -0.5, out,
                              A_log=A_log, dt_bias=dt_bias, lower_bound=-5.0,
                              initial_state=initial, final_state=final)

            run()
            torch.cuda.synchronize()
            if mode == "baseline":
                if not (torch.isfinite(out).all() and torch.isfinite(final).all()):
                    raise RuntimeError("baseline produced non-finite results")
                baseline_out, baseline_final = out.clone(), final.clone()

            def check_results():
                if not torch.equal(initial, initial_copy):
                    raise RuntimeError(f"{mode}: initial_state changed")
                if not torch.equal(out, baseline_out):
                    raise RuntimeError(f"{mode}: output mismatch")
                if not torch.equal(final, baseline_final):
                    raise RuntimeError(f"{mode}: final_state mismatch")

            check_results()
            mean, median = bench_fn(run, warmup, iters)
            check_results()
            print(f"  {mode:8s} mean={mean:.3f} us median={median:.3f} us exact=PASS")
    finally:
        if previous is None:
            os.environ.pop("FLASH_KDA_K2_IMPL", None)
        else:
            os.environ["FLASH_KDA_K2_IMPL"] = previous


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.warmup < 0 or args.iters <= 0:
        parser.error("--warmup must be nonnegative and --iters must be positive")

    import torch

    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 3):
        parser.error("requires an SM103 GPU and FlashKDA built with V1a and V1aE")
    print(f"GPU={torch.cuda.get_device_name()} CC=10.3 "
          f"PyTorch={torch.__version__} CUDA={torch.version.cuda}")
    print(f"warmup={args.warmup} iters={args.iters} seed={args.seed}")
    with torch.inference_mode():
        for B, T, expected in ((1, 8192, "v1a"), (4, 2048, "v1ae"), (8, 1024, "v1ae")):
            run_case(B, T, expected, args.warmup, args.iters, args.seed)


if __name__ == "__main__":
    main()
