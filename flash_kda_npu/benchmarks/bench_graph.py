"""Compare regular forward and explicit fixed-shape NPU graph replay."""
import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch_npu

from flash_kda_npu.benchmarks.bench_fwd import bench_fn, positive_int
from flash_kda_npu import capture_fwd, fwd
from flash_kda_npu.ascendc import get_last_profile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--T', type=positive_int, default=256)
    parser.add_argument('--H', type=positive_int, default=96)
    parser.add_argument('--state', choices=['bf16', 'fp32', 'none'], default='fp32')
    parser.add_argument('--warmup', type=positive_int, default=30)
    parser.add_argument('--iters', type=positive_int, default=200)
    parser.add_argument('--repeats', type=positive_int, default=5)
    parser.add_argument('--json', type=Path)
    args = parser.parse_args()
    os.environ['KDA_PROFILE'] = '0'
    torch.manual_seed(42)
    shape = (1, args.T, args.H, 128)
    q, k, v, g = [torch.randn(shape, device='npu', dtype=torch.bfloat16) for _ in range(4)]
    beta = torch.randn(shape[:-1], device='npu', dtype=torch.bfloat16)
    a_log = torch.rand(args.H, device='npu')
    bias = torch.rand(args.H, 128, device='npu')
    initial = final = None
    if args.state != 'none':
        dtype = torch.float32 if args.state == 'fp32' else torch.bfloat16
        initial = torch.randn((1, args.H, 128, 128), device='npu', dtype=dtype)
        final = torch.empty_like(initial)
    out = torch.empty_like(q)
    inputs = (q, k, v, g, beta, 128 ** -0.5, out, a_log, bias)
    kwargs = dict(initial_state=initial, final_state=final)

    def run():
        fwd(*inputs, **kwargs)

    regular = bench_fn(run, args.warmup, args.iters, args.repeats)
    if not bool(torch.isfinite(out).all()) or (final is not None and not bool(torch.isfinite(final).all())):
        raise RuntimeError('nonfinite output/state')
    expected = out.clone()
    state = None if final is None else final.clone()
    setup_start = time.perf_counter()
    captured = capture_fwd(*inputs, **kwargs)
    torch.npu.synchronize()
    setup_ms = (time.perf_counter() - setup_start) * 1000
    replay = bench_fn(captured.replay, args.warmup, args.iters, args.repeats)
    assert torch.equal(out, expected)
    assert final is None or torch.equal(final, state)
    report = dict(
        device=torch.npu.get_device_name(0), T=args.T, H=args.H, state=args.state,
        warmup=args.warmup, iters=args.iters, repeats=args.repeats,
        bitwise_equal=True, setup_after_warmup_ms=setup_ms,
        regular=regular, graph_replay=replay,
        speedup=regular['mean_ms'] / replay['mean_ms'],
        compile=get_last_profile()['compile'],
    )
    print(json.dumps(report, indent=2))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
