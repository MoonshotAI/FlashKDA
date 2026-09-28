"""Benchmark automatic allocation against caller-owned workspace reuse.

The output buffer is preallocated in both modes so that the only Python API
difference under test is whether ``flash_kda.fwd`` allocates its workspace.
"""

import argparse
import gc
import json
import math
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch.profiler import ProfilerActivity, profile

import flash_kda


@dataclass(frozen=True)
class Case:
    name: str
    batch: int
    tokens: int
    heads: int
    seq_lens: Optional[Tuple[int, ...]] = None


CASES = (
    Case("fixed-t256-h1", batch=1, tokens=256, heads=1),
    Case("fixed-t2048-h8", batch=1, tokens=2048, heads=8),
    Case("batched-b4-t512-h8", batch=4, tokens=512, heads=8),
    Case(
        "varlen-17-33-65-257-h4",
        batch=1,
        tokens=372,
        heads=4,
        seq_lens=(17, 33, 65, 257),
    ),
    Case("fixed-t8192-h32", batch=1, tokens=8192, heads=32),
)

D = 128
LOWER_BOUND = -5.0


def percentile(values, percent):
    values = sorted(values)
    rank = math.ceil(percent / 100 * len(values)) - 1
    return values[max(0, rank)]


def make_inputs(case):
    torch.manual_seed(123)
    shape = (case.batch, case.tokens, case.heads, D)
    q = F.normalize(torch.randn(shape, device="cuda"), p=2, dim=-1).bfloat16()
    k = F.normalize(torch.randn(shape, device="cuda"), p=2, dim=-1).bfloat16()
    v = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    g = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    beta = torch.randn(
        (case.batch, case.tokens, case.heads),
        device="cuda",
        dtype=torch.bfloat16,
    )
    A_log = torch.rand(case.heads, device="cuda", dtype=torch.float32)
    dt_bias = torch.rand(case.heads, D, device="cuda", dtype=torch.float32)

    cu_seqlens = None
    if case.seq_lens is not None:
        if case.batch != 1 or sum(case.seq_lens) != case.tokens:
            raise ValueError(f"invalid varlen case: {case}")
        offsets = [0]
        for length in case.seq_lens:
            offsets.append(offsets[-1] + length)
        cu_seqlens = torch.tensor(offsets, device="cuda", dtype=torch.long)

    kwargs = {
        "q": q,
        "k": k,
        "v": v,
        "g": g,
        "beta": beta,
        "scale": 1.0 / math.sqrt(D),
        "out": torch.empty_like(q),
        "A_log": A_log,
        "dt_bias": dt_bias,
        "lower_bound": LOWER_BOUND,
    }
    if cu_seqlens is not None:
        kwargs["cu_seqlens"] = cu_seqlens
    return kwargs, cu_seqlens


def warmup(fn, calls):
    for _ in range(calls):
        fn()
    torch.cuda.synchronize()


def measure_host_enqueue(fn, iterations):
    samples = []
    for _ in range(iterations):
        start = time.perf_counter_ns()
        fn()
        samples.append((time.perf_counter_ns() - start) / 1_000)
    torch.cuda.synchronize()
    return {
        "p50_us": statistics.median(samples),
        "p95_us": percentile(samples, 95),
    }


def measure_batched_e2e(fn, repeats, calls_per_repeat):
    samples = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        start = time.perf_counter_ns()
        for _ in range(calls_per_repeat):
            fn()
        torch.cuda.synchronize()
        samples.append(
            (time.perf_counter_ns() - start) / 1_000 / calls_per_repeat
        )
    return {
        "p50_us_per_call": statistics.median(samples),
        "p95_us_per_call": percentile(samples, 95),
    }


def measure_stream_elapsed(fn, iterations):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    for _ in range(iterations):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1_000 / iterations


def count_empty_ops(fn, calls):
    with profile(activities=[ProfilerActivity.CPU]) as prof:
        for _ in range(calls):
            fn()
        torch.cuda.synchronize()
    return sum(
        event.count
        for event in prof.key_averages()
        if event.key.startswith("aten::empty")
    )


def measure_incremental_peak(fn, calls):
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    allocated_before = torch.cuda.memory_allocated()
    reserved_before = torch.cuda.memory_reserved()
    for _ in range(calls):
        fn()
    torch.cuda.synchronize()
    return {
        "allocated_bytes": torch.cuda.max_memory_allocated() - allocated_before,
        "reserved_bytes": torch.cuda.max_memory_reserved() - reserved_before,
    }


def check_cuda_graph(fn):
    expected = fn().clone()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = fn()
    graph.replay()
    torch.cuda.synchronize()
    exact = torch.equal(expected, captured)
    del graph, captured, expected
    return exact


def measure_mode(fn, args):
    warmup(fn, args.warmup)
    return {
        "host_enqueue": measure_host_enqueue(fn, args.host_iterations),
        "batched_e2e": measure_batched_e2e(
            fn, args.e2e_repeats, args.calls_per_repeat
        ),
        "stream_elapsed_mean_us": measure_stream_elapsed(
            fn, args.cuda_iterations
        ),
        "aten_empty_calls": count_empty_ops(fn, args.profile_calls),
        "profiled_calls": args.profile_calls,
        "incremental_peak": measure_incremental_peak(fn, args.memory_calls),
    }


def run_case(case, args):
    kwargs, cu_seqlens = make_inputs(case)
    workspace = flash_kda.allocate_workspace(kwargs["q"], cu_seqlens)

    def automatic():
        return flash_kda.fwd(**kwargs)

    def reused():
        return flash_kda.fwd(**kwargs, workspace=workspace)

    automatic_out = automatic().clone()
    reused_out = reused().clone()
    torch.cuda.synchronize()
    exact = torch.equal(automatic_out, reused_out)
    del automatic_out, reused_out

    modes = {
        "automatic": measure_mode(automatic, args),
        "reused": measure_mode(reused, args),
    }
    graph_exact = check_cuda_graph(reused)

    auto = modes["automatic"]
    reuse = modes["reused"]
    ratios = {
        "host_p50_auto_over_reuse": (
            auto["host_enqueue"]["p50_us"]
            / reuse["host_enqueue"]["p50_us"]
        ),
        "e2e_p50_auto_over_reuse": (
            auto["batched_e2e"]["p50_us_per_call"]
            / reuse["batched_e2e"]["p50_us_per_call"]
        ),
        "stream_auto_over_reuse": (
            auto["stream_elapsed_mean_us"]
            / reuse["stream_elapsed_mean_us"]
        ),
    }
    return {
        "case": asdict(case),
        "workspace_bytes": workspace.numel(),
        "automatic_equals_reused": exact,
        "reused_cuda_graph_exact": graph_exact,
        "modes": modes,
        "ratios": ratios,
    }


def format_bytes(value):
    return f"{value / (1024 * 1024):.2f} MiB"


def print_markdown(results):
    print()
    print(
        "| Case | Mode | Workspace | Host p50/p95 (us) | "
        "E2E p50/p95 (us) | Stream mean (us) | empty/call | Peak alloc |"
    )
    print("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for result in results:
        for mode_name, mode in result["modes"].items():
            host = mode["host_enqueue"]
            e2e = mode["batched_e2e"]
            empty_per_call = mode["aten_empty_calls"] / mode["profiled_calls"]
            print(
                f"| {result['case']['name']} | {mode_name} | "
                f"{format_bytes(result['workspace_bytes'])} | "
                f"{host['p50_us']:.2f}/{host['p95_us']:.2f} | "
                f"{e2e['p50_us_per_call']:.2f}/{e2e['p95_us_per_call']:.2f} | "
                f"{mode['stream_elapsed_mean_us']:.2f} | "
                f"{empty_per_call:.2f} | "
                f"{format_bytes(mode['incremental_peak']['allocated_bytes'])} |"
            )
    print()
    for result in results:
        ratios = result["ratios"]
        print(
            f"{result['case']['name']}: exact="
            f"{result['automatic_equals_reused']}, graph_exact="
            f"{result['reused_cuda_graph_exact']}, auto/reuse="
            f"host {ratios['host_p50_auto_over_reuse']:.3f}x, "
            f"e2e {ratios['e2e_p50_auto_over_reuse']:.3f}x, "
            f"stream {ratios['stream_auto_over_reuse']:.3f}x"
        )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cases",
        nargs="*",
        choices=[case.name for case in CASES],
        help="case names to run (default: all)",
    )
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--host-iterations", type=int, default=200)
    parser.add_argument("--e2e-repeats", type=int, default=30)
    parser.add_argument("--calls-per-repeat", type=int, default=10)
    parser.add_argument("--cuda-iterations", type=int, default=200)
    parser.add_argument("--profile-calls", type=int, default=25)
    parser.add_argument("--memory-calls", type=int, default=5)
    parser.add_argument("--json-out", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    selected = [case for case in CASES if not args.cases or case.name in args.cases]
    metadata = {
        "gpu": torch.cuda.get_device_name(),
        "compute_capability": list(torch.cuda.get_device_capability()),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "settings": {
            key: value
            for key, value in vars(args).items()
            if key not in {"cases", "json_out"}
        },
    }
    print(json.dumps(metadata, indent=2))

    results = []
    with torch.inference_mode():
        for case in selected:
            print(f"benchmarking {case.name}...", flush=True)
            results.append(run_case(case, args))
            gc.collect()
            torch.cuda.empty_cache()

    print_markdown(results)
    payload = {"metadata": metadata, "results": results}
    if args.json_out is not None:
        args.json_out.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"wrote {args.json_out}")


if __name__ == "__main__":
    main()
