# FlashKDA Ascend NPU backend

High-performance KDA forward kernels for Ascend NPUs, implemented in AscendC.
The `flash_kda_npu.fwd` function has the same calling convention as CUDA's
`flash_kda.fwd` and supports fixed-length batches and packed variable-length sequences.

A3 and Ascend 950 share this backend and the `flash-kda-npu` distribution.
The active device selects the kernel implementation at runtime. Both families
use `import flash_kda_npu` and the same source installation command on x86_64
and aarch64 hosts.

See the [NPU implementation deep dive](docs/20261021-flashkda-npu-deep-dive.md)
for kernel design, accuracy results, data layouts and synchronization.

## Requirements

- Linux with an Ascend NPU. The repository includes A3 (`Ascend910_9382`) and
  950 (`Ascend950` family) implementations.
- A compatible CANN toolkit with RTC headers/runtime (`acl_rtc`) and a C++17
  compiler, plus a compatible PyTorch and `torch_npu` pair.
- For source builds: `setuptools>=64` and `wheel`.

| Device | CANN | PyTorch | `torch_npu` |
|---|---|---|---|
| A3 | 9.2.0 | 2.10.0+cpu | 2.10.0 |
| 950 DT | 9.2.0 | 2.10.0+cpu | 2.10.0 |

Both environments were tested with Python 3.12.

### Shared runtime and platform kernels

| Component | A3 | Ascend 950 |
|---|---|---|
| Public API | `flash_kda_npu.fwd` | `flash_kda_npu.fwd` |
| Host scheduling / validation | Shared Python implementation | Shared Python implementation |
| Kernel sources | `csrc/ascendc/` | Shared sources plus `csrc/ascendc950/` helpers and specialized kernels |
| RTC target | Runtime SoC via `--npu-soc` | `--npu-arch=dav-3510` |

`device_config.current_caps()` probes and caches the active device/context.
Source selection and workspace allocation use that same identity. 950-specific
VF/direct C/V switches apply only to Ascend 950; A3 retains its GM handoff path.
Compilation caches are scoped by device/context, so switching devices cannot reuse
another device's loaded kernel handles.

## Installation

The Ascend distribution is named **`flash-kda-npu`**, imported as
`flash_kda_npu`. Its build entry point is `flash_kda_npu/setup.py`, independent
of the CUDA package at the repository root. It also installs a `flash_kda`
compatibility entry point for FLA's existing FlashKDA backend.

Prepare CANN 9.2.0, PyTorch 2.10.0, `torch_npu` 2.10.0 and the build requirements
above, then source the CANN environment on either device:

```bash
export ASCEND_HOME_PATH=/usr/local/Ascend/cann-9.2.0
source "$ASCEND_HOME_PATH/set_env.sh"
```

### Install from GitHub

With Git installed, run:

```bash
python -m pip install --no-build-isolation "git+https://github.com/Ascend/FlashKDA.git@dev_npu#subdirectory=flash_kda_npu"
```

### Install from local source

Clone the repository, or use an existing local checkout:

```bash
git clone --branch dev_npu --single-branch https://github.com/Ascend/FlashKDA.git
cd FlashKDA
```

From the repository root, install the NPU package:

```bash
python -m pip install 'setuptools>=64' wheel
python -m pip install --no-build-isolation ./flash_kda_npu
```

For development, use an editable installation instead:

```bash
python -m pip install --no-build-isolation -e ./flash_kda_npu
```

Both installation methods build the host launcher `flash_kda_npu_C` for the
current machine using PyTorch's `BuildExtension`. `CXX` selects the host compiler;
PyTorch supplies the headers and ABI flags. CUDA and CUTLASS are not needed.
AscendC kernel sources are installed with the package and compiled through RTC
on first use. Warm up before measuring latency.

## Use with FLA

With an Ascend-enabled FLA build that includes `FlashKDABackend`, installing
`flash-kda-npu` is enough for FLA to discover the NPU implementation. The
integration was validated with FLA 0.6.0 at revision
[`15396e6`](https://github.com/fla-org/flash-linear-attention/tree/15396e604df6a4699950d7f20457b391b0cd8e74).
No FLA source changes or additional initialization imports are needed.

Run FLA applications outside this repository root, whose CUDA `flash_kda/`
source directory would shadow the installed NPU compatibility entry point.

Call `chunk_kda` with BF16 NPU inputs under inference mode:

```python
import torch
import torch_npu
from fla.ops.kda import chunk_kda

with torch.inference_mode():
    out, final_state = chunk_kda(
        q=q, k=k, v=v, g=g, beta=beta,
        initial_state=h0, output_final_state=True,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        use_beta_sigmoid_in_kernel=True,
        safe_gate=True, lower_bound=-5.0,
        state_v_first=True,
        A_log=A_log, dt_bias=dt_bias,
        cu_seqlens=cu_seqlens,
    )
```

This path requires `K=V=128`, equal query/value head counts, and unsigned
`sigmoid(beta)` (`allow_neg_eigval=False`). Pass raw gate and beta logits;
`A_log` and `dt_bias` are required FP32 tensors. FLA returns FP32 final states.
Fixed-length and packed inputs are supported, including tail and empty sequences.

Set `FLA_FLASH_KDA=0` to disable this backend. Enable INFO logging to see FLA's
`kda.chunk_kda -> flash_kda` dispatch message.

## Usage

```python
import torch
import torch_npu  # Register the NPU device backend.
import flash_kda_npu

B, T, H, D = 1, 64, 4, 128
q, k, v, g = [
    torch.randn((B, T, H, D), device="npu", dtype=torch.bfloat16)
    for _ in range(4)
]
beta = torch.randn((B, T, H), device="npu", dtype=torch.bfloat16)
A_log = torch.zeros(H, device="npu", dtype=torch.float32)
dt_bias = torch.zeros((H, D), device="npu", dtype=torch.float32)
out = torch.empty_like(v)
h0 = torch.zeros((B, H, D, D), device="npu", dtype=torch.float32)
h1 = torch.empty_like(h0)

flash_kda_npu.fwd(
    q, k, v, g, beta, D ** -0.5, out,
    A_log=A_log, dt_bias=dt_bias, lower_bound=-5.0,
    initial_state=h0, final_state=h1, cu_seqlens=None,
)
torch.npu.synchronize()
print(out.shape, h1.shape)
```

The function writes `out` and optional `final_state` in place and returns `None`.
Q/K normalization, gate activation and beta sigmoid run internally; pass raw
`g` and beta logits.

### API parameters

`D=128`. For fixed-length batches, `N=B`. For packed inputs, `B=1`, `T` is the
total token count, and `N=cu_seqlens.numel()-1`.

| Parameter | Dtype | Shape / meaning |
|---|---|---|
| `q`, `k`, `v`, `g` | BF16 | `[B,T,H,D]`; `g` is the raw gate input |
| `beta` | BF16 | `[B,T,H]`; raw logits, before sigmoid |
| `scale` | Python float | Output scaling factor, usually `D ** -0.5` |
| `out` | BF16 | `[B,T,H,D]`; caller-provided output buffer |
| `A_log` | FP32 | `[H]`; log-gate parameter |
| `dt_bias` | FP32 | `[H,D]`; gate bias |
| `lower_bound` | Python float | Gate lower bound; default `-5.0` |
| `initial_state` | BF16 / FP32 / `None` | `[N,H,D,D]`; `None` starts from zero |
| `final_state` | BF16 / FP32 / `None` | `[N,H,D,D]`; optional output buffer |
| `cu_seqlens` | INT64 / `None` | `[N+1]`; packed sequence boundaries on NPU |

All tensors must satisfy the following requirements:

- All tensors must be contiguous and on the same NPU; `K = V = 128`.
- `q/k/v/g/out` and `beta` use BF16; `A_log` and `dt_bias` use FP32.
- Initial and final states may be BF16 or FP32; their dtypes must match when
  both are supplied. Each batch element or packed sequence has its own state.
- Packed inputs use `B=1` and NPU INT64 `cu_seqlens`, starting at zero and ending
  at `T`, with nondecreasing boundaries. Empty sequences preserve the initial
  state, or return zero state.
- Sequence lengths need not be chunk-aligned. `KDA_CHUNK` supports 16, 32 and 64
  (default); set it before starting Python.

For repeated fixed-length calls with stable tensor addresses, `KDA_AUTO_GRAPH=1`
enables automatic graph capture and replay. Update inputs in place on the call's
stream; packed inputs bypass this cache. Explicit capture is available through
`flash_kda_npu.capture_fwd`. See the deep dive for graph ownership and
synchronization requirements.

## Performance

See [BENCHMARK_A3.md](BENCHMARK_A3.md) for A3 latency and speedups against FLA.
See [BENCHMARK_950.md](BENCHMARK_950.md) for the same case layouts on 950 and the
[950 KDA optimization guide](docs/20261114-950-performance-accuracy.md)
for the matrix layouts, register arithmetic, direct C/V exchange and default switches.
Each report records the environment used for its measurements; the historical A3
results use CANN 9.1.0 and PyTorch 2.9.0.

```bash
bash flash_kda_npu/benchmarks/bench.sh --skip-fla \
  --mode all --H 96 --state fp32 --warmup 5 --iters 20 --repeats 3
```

This runs FlashKDA with FP32 initial/final states for fixed-length and both packed
layouts. Repeat with `--H 64` for the other report shapes. The script installs the
NPU source in editable mode before benchmarking.

For FLA KDA/GDN comparisons, follow the reproduction commands in the device's
benchmark report. They specify the pinned FLA revision and Triton Ascend 3.2.2;
the 950 report also applies the required FLA state-update compiler settings.
Keep FLA backend dispatch enabled and set `FLA_FLASH_KDA=0` for those comparisons.
Use `--mode fixed` or `--mode varlen` to select one sequence layout.

## Tests

```bash
# Hardware correctness, run separately on A3 and 950:
python -m pytest -q flash_kda_npu/tests/
# Include the 128K/1M-token cases:
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q flash_kda_npu/tests/ --run-long
```

- `tests/test_fwd.py`: native correctness, input validation, partial/empty
  sequences, graph replay and device handling.
- `tests/test_fwd_full.py`: the CUDA-sized matrix through the public NPU API.
- `tests/cuda_cases.py`: shared test dimensions and state combinations mirrored
  from CUDA; no CUDA runtime dependency.

Run tests and benchmark scripts from the source checkout. The installed package
contains the runtime Python modules, native launcher and AscendC sources.

Numerical tests compare with an independent FP64 reference using error tolerances;
CUDA/NPU bitwise equality is not required. Gate-sweep final-state threshold
violations warn, matching CUDA's policy; nonfinite results fail. See the deep dive
for thresholds and measured coverage. NPU tests are skipped when no NPU is available.
The fixed-length `g=-8, bias=0` gate case has a known final-state RMS of 0.7083%
on 950 with chunk=64 and BF16 state, above its 0.5% warning threshold.
