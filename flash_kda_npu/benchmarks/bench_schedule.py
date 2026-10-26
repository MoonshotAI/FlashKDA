"""Sweep NPU scheduling overrides and verify bitwise-equivalent results.

Run from the repository root after sourcing CANN's set_env.sh:
  python -m flash_kda_npu.benchmarks.bench_schedule --json /tmp/schedule.json
Stage timings use synchronized host measurements; end-to-end timings use
NPU events with KDA_PROFILE disabled and retain the solve overlap.
"""
import argparse
import json
import os
import statistics
from pathlib import Path

import torch
import torch_npu
import flash_kda
from flash_kda_npu.benchmarks.bench_fwd import bench_fn
from flash_kda_npu.ascendc import _aic_core_count, get_last_profile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--json', type=Path, required=True)
    parser.add_argument('--iters', type=int, default=20)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--seq-lens', type=int, nargs='+')
    parser.add_argument('--heads', type=int, default=96)
    args = parser.parse_args()
    if args.iters < 1 or args.repeats < 1 or args.heads < 1:
        parser.error('iters, repeats and heads must be positive')
    if args.seq_lens is not None and (min(args.seq_lens) < 0 or sum(args.seq_lens) == 0):
        parser.error('sequence lengths must be nonnegative with positive total')
    for key in ('KDA_PRE_UNROLL', 'KDA_PERSIST_LOOP_BLOCKS', 'KDA_PERSIST_HEAD_MAP', 'KDA_PROFILE'):
        os.environ.pop(key, None)
    torch.manual_seed(42)
    cores = _aic_core_count(torch.device('npu:0'))
    print('cores', cores, flush=True)
    results = []
    cases = ([args.seq_lens] if args.seq_lens is not None else
             ([8192], [1300, 547, 2048, 963, 271, 3063], [1024] * 8))
    for lengths in cases:
        h = args.heads
        total = sum(lengths)
        shape = (1, total, h, 128)
        q, k, v, g = [torch.randn(shape, device='npu', dtype=torch.bfloat16) for _ in range(4)]
        beta = torch.randn(shape[:-1], device='npu', dtype=torch.bfloat16)
        a, bias = (torch.rand(h, device='npu'), torch.rand(h, 128, device='npu'))
        initial = torch.randn((len(lengths), h, 128, 128), device='npu')
        final = torch.empty_like(initial)
        out = torch.empty_like(q)
        cu = None
        if len(lengths) > 1:
            cu = torch.tensor(
                [0, *torch.tensor(lengths).cumsum(0).tolist()],
                device='npu', dtype=torch.int64,
            )

        def run():
            flash_kda.fwd(
                q, k, v, g, beta, 128 ** (-0.5), out, a, bias, -5.0,
                initial_state=initial, final_state=final, cu_seqlens=cu,
            )

        bh = len(lengths) * h
        block_counts = {(bh + mh - 1) // mh for mh in (1, 2, 3, 4)}
        block_counts.update(cores * wave for wave in range(1, 10)
                            if (bh + 3) // 4 <= cores * wave <= bh)
        configs = [(0, 0), (2, 0), (66, 0)]
        configs.extend((0, nb) for nb in sorted(block_counts))
        ref = None
        for u, nb in configs:
            os.environ['KDA_PRE_UNROLL'] = str(u)
            os.environ['KDA_PERSIST_LOOP_BLOCKS'] = str(nb)
            os.environ['KDA_PROFILE'] = '0'
            run()
            torch.npu.synchronize()
            if not (torch.isfinite(out).all() and torch.isfinite(final).all()):
                raise RuntimeError('nonfinite output/state')
            if ref is None:
                ref = (out.clone(), final.clone())
            if not (torch.equal(out, ref[0]) and torch.equal(final, ref[1])):
                raise AssertionError(f'output/state mismatch: {lengths=}, {u=}, {nb=}')
            timing = bench_fn(run, 5, args.iters, args.repeats)
            os.environ['KDA_PROFILE'] = '1'
            profiles = []
            for _ in range(5):
                run()
                profiles.append(get_last_profile())
            row = dict(
                lengths=lengths, heads=h, state_dtype="fp32",
                unroll_override=u, blocks_override=nb,
                pre_unroll=profiles[-1]['pre_unroll'],
                blocks=profiles[-1]['launch_blocks'],
                persistent_head_map=profiles[-1]['persistent_head_map'], bitwise_equal=True,
                warmup=5, iters=args.iters, repeats=args.repeats,
                **timing,
                **{key: statistics.median(p[key] for p in profiles)
                   for key in ('pre_gram_ms', 'solve_ms', 'k2_ms')})
            results.append(row)
            print(json.dumps(row), flush=True)
            args.json.parent.mkdir(parents=True, exist_ok=True)
            report = dict(
                cube_cores=cores, device=torch.npu.get_device_name(0), results=results,
            )
            args.json.write_text(json.dumps(report, indent=2) + '\n')
        del q, k, v, g, beta, a, bias, initial, final, out, ref
        torch.npu.empty_cache()


if __name__ == '__main__':
    main()
