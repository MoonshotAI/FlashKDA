# FlashKDA Benchmark on NVIDIA A800 (SM80)

> Test date: 2026-07-30
> GPU: NVIDIA A800-SXM4-80GB (SM80, Ampere)
> CUDA: 12.2
> PyTorch: 2.4+
> flash_kda: dual-arch branch `dual-arch-sm80-3way`

## Summary

FlashKDA SM80 forward kernel achieves **~1.8× speedup** over the fla Triton
`chunk_kda` implementation on A800 for the KDA prefill shape (T=8192, H=96,
D=128).

| Implementation | Mean | Min | Max | vs flash_kda |
|---|---|---|---|---|
| **flash_kda (SM80, bf16 state)** | **3.40 ms** | 3.10 ms | 4.46 ms | — |
| chunk_kda (fla Triton) | 6.13 ms | 6.09 ms | 6.52 ms | **1.80× slower** |
| chunk_gated_delta_rule (fla Triton) | 3.63 ms | 3.59 ms | 4.49 ms | 1.07× slower |

## Kernel Breakdown (PyTorch Profiler)

| Kernel | Time / call | Share |
|---|---|---|
| `_flash_kda_fwd_recurrence_sm80` (K2) | 2.35 ms | 58.8% |
| `_flash_kda_fwd_prepare_sm80` (K1) | 1.63 ms | 40.8% |

> Breakdown samples are from a co-tenancy rerun (e2e ~4.0 ms); clean-environment e2e is 3.40 ms.

> ncu is unavailable in the test environment (driver resource restricted);
> PyTorch Profiler is used for the kernel-level breakdown.

## Shared Memory / Occupancy

`SharedStorageK2` is specialized on `StateFP32` so the fp32 conversion scratch
buffer (64 KB) is only reserved when actually needed:

| Path | Shared Memory / CTA | Theoretical CTAs / SM (A800 164 KB) |
|---|---|---|
| bf16 state | **71.2 KB** | **2** |
| fp32 state | 96.0 KB | 1 |

Pipeline stages are unified to the values the kernel actually uses:
`kK2InputStages = 2`, `kK2OutputStages = 1` (previously declared as 3/2 in the
launch code while the kernel only ever used 2/1).

## Test Configuration

- Shape: `B=1, T=8192, H=96, D=128`
- `initial_state` / `final_state`: bf16
- Benchmark script: `benchmarks/bench_fwd.py --mode fixed --H 96 --D 128 --warmup 5 --iters 20 --repeats 3`
- `chunk_gated_delta_rule` is included as a reference point only; it implements
  Gated DeltaNet (scalar per-head gate), not KDA.

## Notes

- SM80 path uses cooperative copies and a 2-stage `cp.async` pipeline instead of
  TMA; numerical output is bit-exact against the torch reference for the tested
  shapes.
- `no state` and `fp32 state` configurations show the same min latency
  (~3.0–3.2 ms); occasional max outliers are first-iteration noise.
- `FLASH_KDA_CUDA_ARCHS=all` build requires CUDA 12.9+ for `compute_100a`;
  dual-arch (`80,90a`) build verified on CUDA 12.2.

## Correctness

- `tests/test_fwd.py`: 4 passed (`test_fwd`, `test_fwd_varlen`,
  `test_fwd_vs_fla`, `test_fwd_varlen_vs_fla`)
- Output matches `torch_ref` bit-exactly.
- vs fla `chunk_kda`: err_ratio ≈ 3–5e-3 (bf16 noise level).

## Re-run on 2026-08-17

- **Kernel breakdown reproduced** (PyTorch Profiler): K2 recurrence 2.35 ms (58.8%),
  K1 prepare 1.63 ms (40.8%) — identical to the table above.
- End-to-end ~4.0 ms (mean) under heavy GPU co-tenancy (~70 GB used by other
  tenants); clean-env figure remains ~3.40 ms.
- `compute-sanitizer --tool memcheck`: **0 errors** on the full A800 shape,
  both `fixed` and `varlen`.
- ncu/nsys remain unavailable in this environment (driver resource held,
  `perf_event_paranoid=4`, no Nsight Systems).
