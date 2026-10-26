# KDA forward benchmark (Ascend / A3)

Measured on 2026-10-21. Across the six cases below, FlashKDA is **8.32–9.69×
faster than FLA KDA** on the same A3 device. The latency ratio against FLA GDN is
**4.92–5.29×**; GDN uses a different recurrence and is included as a reference.

The case layouts and table columns follow the [CUDA H20 report](../BENCHMARK_H20.md).
These are same-device implementation comparisons, not an A3-versus-H20 comparison.

## Environment and settings

- Device: one A3 `Ascend910_9382`, 24 Cube cores and 48 Vector cores.
- CANN 9.1.0; PyTorch 2.9.0+cpu; `torch_npu` 2.9.0.post6.
- FLA 0.6.0, revision `15396e604df6a4699950d7f20457b391b0cd8e74`, matching
  the dependency pinned by `benchmarks/bench.sh`; Triton Ascend 3.2.2.
- FlashKDA runtime and benchmark sources match commit `a5c862c`; RTC target
  `Ascend910B3`, `KDA_CHUNK=64`, default scheduling, eager execution.
- All implementations use BF16 Q/K/V and FP32 initial/final states, with
  `B=1`, `T=8192`, `D=128`. Packed cases split the same total token count.
- Each implementation uses **5 warmups, 20 iterations × 3 repeats** per case;
  tables show the mean of 60 NPU Event samples. The CUDA H20 report uses a
  larger sample count (30 warmups, 200 iterations × 5 repeats).

`chunk_kda` enables `use_gate_in_kernel`, `use_qk_l2norm_in_kernel` and
`use_beta_sigmoid_in_kernel`, with `lower_bound=-5` and
`transpose_state_layout=True`. `FLA_FLASH_KDA=0` prevents FLA from dispatching
back to FlashKDA; the Ascend backend remains enabled. CANN's Python paths are
preserved so FLA detects the device's 192 KiB UB capacity.

`chunk_gated_delta_rule` uses Q/K normalization and the same state layout,
with a scalar gate `logsigmoid(randn)` of shape `[1,T,H]` and `sigmoid(beta_logits)`.
These activations are prepared outside timing. This differs from the CUDA
benchmark's raw normal GDN gate/beta inputs, which can overflow on long sequences.

## Results

### `T=8192`, `H=96`, `D=128`

| Case | `flash_kda` mean (ms) | `fla_chunk_kda` mean (ms) | Speedup vs `chunk_kda` | `fla_chunk_gdn` mean (ms) | Speedup vs `gdn` |
|------|----------------------:|--------------------------:|----------------------:|--------------------------:|------------------:|
| Fixed | 6.8877 | 66.7191 | 9.69× | 36.4053 | 5.29× |
| Varlen, `seq_lens`=[1300, 547, 2048, 963, 271, 3063] | 7.5781 | 66.8824 | 8.83× | 39.0659 | 5.16× |
| Varlen, `seq_lens`=`1024 x 8` | 7.5421 | 66.8049 | 8.86× | 38.3257 | 5.08× |

### `T=8192`, `H=64`, `D=128`

| Case | `flash_kda` mean (ms) | `fla_chunk_kda` mean (ms) | Speedup vs `chunk_kda` | `fla_chunk_gdn` mean (ms) | Speedup vs `gdn` |
|------|----------------------:|--------------------------:|----------------------:|--------------------------:|------------------:|
| Fixed | 4.8216 | 44.3018 | 9.19× | 24.6374 | 5.11× |
| Varlen, `seq_lens`=[1300, 547, 2048, 963, 271, 3063] | 5.3668 | 44.6427 | 8.32× | 26.4231 | 4.92× |
| Varlen, `seq_lens`=`1024 x 8` | 5.1427 | 44.4154 | 8.64× | 25.6862 | 4.99× |

Speedup is `FLA mean latency / FlashKDA mean latency`, computed before rounding.
Compilation and warmup are excluded. Input generation and initial-state setup
are outside timing; conversions, allocations, packed metadata processing and
kernel launches inside each public operator call remain inside the timed region.
NPU Event intervals are device-timeline measurements, not synchronized host
wall-clock measurements or whole-model inference latency.

All 18 implementation/case combinations passed the benchmark's untimed checks
for finite outputs and final states. Numerical accuracy and broader test coverage
are documented in the [NPU deep dive](docs/20261021-flashkda-npu-deep-dive.md).

## Reproduce

After [installing the NPU backend](README.md#installation), run from the repository
root. The first command installs the pinned FLA/Triton dependencies and measures
H=96; the second reuses that environment for H=64.

```bash
export ASCEND_HOME_PATH=/usr/local/Ascend/cann-9.1.0
source "$ASCEND_HOME_PATH/set_env.sh"
export ASCEND_RT_VISIBLE_DEVICES=6
export PYTHONPATH="$PWD:$PYTHONPATH"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export FLA_FLASH_KDA=0 KDA_AUTO_GRAPH=0 KDA_PROFILE=0 KDA_CHUNK=64
unset FLA_DISABLE_BACKEND_DISPATCH

bash flash_kda_npu/benchmarks/bench.sh \
  --mode all --H 96 --state fp32 --warmup 5 --iters 20 --repeats 3
python flash_kda_npu/benchmarks/bench_fwd.py \
  --mode all --H 64 --state fp32 --warmup 5 --iters 20 --repeats 3
```

Select an idle device and use the default scheduling settings. Ensure the imported
FLA source is the pinned revision if the environment also contains an editable
FLA checkout. The ratios above apply to this dependency version and these shapes;
other FLA revisions, sequence lengths and model workloads need separate measurement.
