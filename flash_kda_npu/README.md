# FlashKDA Ascend NPU backend

High-performance KDA forward kernels for Ascend NPUs, implemented in AscendC.
The backend uses the same `flash_kda.fwd` API as CUDA and supports fixed-length
batches and packed variable-length sequences.

See the [NPU implementation deep dive](docs/20261021-flashkda-npu-deep-dive.md)
for kernel design, accuracy results, data layouts and synchronization.

## Requirements

- Linux with an Ascend NPU; validated on A3 (`Ascend910_9382`).
- CANN 9.1.0 with RTC headers/runtime (`acl_rtc`) and a C++17 compiler.
- Compatible PyTorch and `torch_npu`; validated with PyTorch 2.9.0 and
  `torch_npu` 2.9.0.post6.

The RTC target currently defaults to `Ascend910B3`; it is not detected automatically.

## Installation

From the repository root, with PyTorch and `torch_npu` already installed:

```bash
export ASCEND_HOME_PATH=/usr/local/Ascend/cann-9.1.0
source "$ASCEND_HOME_PATH/set_env.sh"
pip install -v --no-build-isolation .
```

`ASCEND_HOME_PATH` enables the NPU `CppExtension`; setuptools compiles and installs
`flash_kda_npu_C` using PyTorch's `BuildExtension`, which also builds the CUDA
extension. CUDA-only wheels omit the NPU Python package, extension and RTC sources;
set `ASCEND_HOME_PATH` when building a wheel with NPU support.
For development, use an editable installation:

```bash
pip install -v -e . --no-build-isolation
```

The installation builds the launcher. AscendC kernel sources are included in the
package and compile through RTC on first use. Warm up before measuring latency.
`CXX` selects the host compiler; PyTorch supplies the Python headers and ABI flags.

## Usage

```python
import flash_kda

flash_kda.fwd(
    q, k, v, g, beta, scale, out,
    A_log=A_log, dt_bias=dt_bias, lower_bound=-5.0,
    initial_state=h0, final_state=h1, cu_seqlens=None,
)
```

The function writes `out` and optional `final_state` in place and returns `None`.
Q/K normalization, gate activation and beta sigmoid run internally; pass raw
`g` and beta logits. NPU tensors select the native AscendC backend.

See the shared [API parameter table](../README.md#kernel-api). On NPU:

- All tensors must be contiguous and on the same NPU; `K = V = 128`.
- `q/k/v/g/out` and `beta` use BF16; `A_log` and `dt_bias` use FP32.
- Initial and final states may be BF16 or FP32; their dtypes must match when
  both are supplied. Each batch element or packed sequence has its own state.
- Packed inputs use `B=1` and NPU INT64 `cu_seqlens`, starting at zero and ending
  at `T`. Empty sequences preserve the initial state, or return zero state.
- Sequence lengths need not be chunk-aligned. `KDA_CHUNK` supports 16, 32 and 64
  (default); set it before starting Python.

For repeated fixed-length calls with stable tensor addresses, `KDA_AUTO_GRAPH=1`
enables automatic graph capture and replay. Update inputs in place on the call's
stream; packed inputs bypass this cache. Explicit capture is available through
`flash_kda_npu.capture_fwd`. See the deep dive for graph ownership and
synchronization requirements.

## Performance

See [BENCHMARK_A3.md](BENCHMARK_A3.md) for A3 latency and speedups against FLA.

```bash
FLA_FLASH_KDA=0 bash flash_kda_npu/benchmarks/bench.sh --state fp32
```

The benchmark compares native FlashKDA with FLA KDA and GDN on NPU. The setup
script installs the compatible FLA revision and Triton Ascend 3.2.2; use
`--skip-fla` for native-only runs. Keep FLA backend dispatch enabled and set
`FLA_FLASH_KDA=0` when comparing against FLA, so it does not call FlashKDA itself.
Use `--mode fixed` or `--mode varlen` to select the sequence layout.

## Tests

```bash
python -m pytest -q flash_kda_npu/tests/
# Include the 128K/1M-token cases:
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q flash_kda_npu/tests/ --run-long
```

- `tests/test_fwd.py`: native correctness, input validation, partial/empty
  sequences, graph replay and device handling.
- `tests/test_fwd_full.py`: the CUDA-sized matrix through the public NPU API.
- `tests/cuda_cases.py`: shared test dimensions and state combinations mirrored
  from CUDA; no CUDA runtime dependency.

Numerical tests compare with an independent FP64 reference using error tolerances;
CUDA/NPU bitwise equality is not required. Gate-sweep final-state threshold
violations warn, matching CUDA's policy; nonfinite results fail. See the deep dive
for thresholds and measured coverage. NPU tests are skipped when no NPU is available.
