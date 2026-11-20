# KDA forward benchmark (Ascend / 950)

Measured on 2026-11-16. Across the six cases below, FlashKDA is **3.64–6.19×
faster than FLA KDA** on the same 950 DT device. The latency ratio against FLA
GDN is **2.75–3.71×**; GDN uses a different recurrence and is included as a reference.

The case layouts and table columns follow the [A3 report](BENCHMARK_A3.md).

## Environment and settings

- Device: one `Ascend950DT_9582`, 32 Cube cores and 64 Vector cores.
- CANN 9.2.0; Python 3.12.15; PyTorch 2.10.0+cpu; `torch_npu` 2.10.0.
- RTC target `--npu-arch=dav-3510`, chunk=64, default scheduling, eager execution;
  `KDA_AUTO_GRAPH=0`, `KDA_PROFILE=0`.
- FLA 0.6.0, revision `15396e604df6a4699950d7f20457b391b0cd8e74`;
  Triton Ascend 3.2.2. The FLA state-update kernel uses
  `enable_ubuf_saving=False` and `unit_flag=False`.
- BF16 Q/K/V/g/beta, FP32 initial and final states; B=1, T=8192, D=128.
- Per implementation/case: **5 warmups, 20 iterations × 3 repeats**;
  tables show the mean of 60 NPU Event samples.

`chunk_kda` enables `use_gate_in_kernel`, `use_qk_l2norm_in_kernel` and
`use_beta_sigmoid_in_kernel`, with `lower_bound=-5` and
`transpose_state_layout=True`. `FLA_FLASH_KDA=0` prevents FLA from dispatching
back to FlashKDA.

`chunk_gated_delta_rule` uses Q/K normalization and the same state layout,
with a scalar `logsigmoid(randn)` gate and `sigmoid(beta_logits)`.
These activations are prepared outside timing.

## Results

### `T=8192`, `H=96`, `D=128`

| Case | FlashKDA mean (ms) | `fla_chunk_kda` mean (ms) | Speedup vs `chunk_kda` | `fla_chunk_gdn` mean (ms) | Speedup vs `gdn` |
| --- | ---: | ---: | ---: | ---: | ---: |
| Fixed | 5.0400 | 23.6050 | 4.68× | 18.6920 | 3.71× |
| Varlen, `seq_lens`=[1300, 547, 2048, 963, 271, 3063] | 6.5245 | 23.7227 | 3.64× | 19.1497 | 2.94× |
| Varlen, `seq_lens`=`1024 x 8` | 5.3334 | 23.6434 | 4.43× | 18.9572 | 3.55× |

### `T=8192`, `H=64`, `D=128`

| Case | FlashKDA mean (ms) | `fla_chunk_kda` mean (ms) | Speedup vs `chunk_kda` | `fla_chunk_gdn` mean (ms) | Speedup vs `gdn` |
| --- | ---: | ---: | ---: | ---: | ---: |
| Fixed | 3.4332 | 21.2669 | 6.19× | 12.4006 | 3.61× |
| Varlen, `seq_lens`=[1300, 547, 2048, 963, 271, 3063] | 4.6442 | 18.1423 | 3.91× | 12.7603 | 2.75× |
| Varlen, `seq_lens`=`1024 x 8` | 3.7200 | 20.7975 | 5.59× | 12.6606 | 3.40× |

Speedup is FLA mean latency / FlashKDA mean latency, computed before rounding.
Compilation, warmup, input generation and initial-state setup are excluded.
Conversions, allocations, packed metadata processing and kernel launches inside
each public operator call remain in the timed region. NPU Events measure
device-timeline intervals.

All 18 implementation/case combinations passed untimed checks for finite outputs
and final states. See the [950 optimization guide](docs/20261114-950-performance-accuracy.md)
for kernel design and default switches.

## Reproduce

After [installing the NPU backend](README.md#installation), run from the repository
root with the pinned FLA revision and Triton Ascend 3.2.2:

```bash
export ASCEND_HOME_PATH=/usr/local/Ascend/cann-9.2.0
source "$ASCEND_HOME_PATH/set_env.sh"
export ASCEND_RT_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export FLA_FLASH_KDA=0 KDA_AUTO_GRAPH=0 KDA_PROFILE=0 KDA_CHUNK=64
unset FLA_DISABLE_BACKEND_DISPATCH

python -m pip install --no-deps triton-ascend==3.2.2 \
  --extra-index-url https://mirrors.huaweicloud.com/ascend/repos/pypi
python -m pip install --no-deps \
  'https://github.com/fla-org/flash-linear-attention/archive/15396e604df6a4699950d7f20457b391b0cd8e74.tar.gz' einops

python - <<'PY'
import importlib
import sys
from flash_kda_npu.benchmarks.bench_fwd import main

state_update = importlib.import_module(
    "fla.ops.common.backends.triton_ascend.chunk_delta_h"
)
state_update._FWD_H_COMPILE.update(enable_ubuf_saving=False, unit_flag=False)
for heads in (96, 64):
    sys.argv = [
        "bench_fwd",
        "--mode", "all", "--H", str(heads), "--state", "fp32",
        "--warmup", "5", "--iters", "20", "--repeats", "3",
    ]
    main()
PY
```

Select an idle device. The reported ratios apply to these dependency versions,
compiler options and shapes.
