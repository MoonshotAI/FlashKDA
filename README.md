# FlashKDA

FlashKDA: Flash Kimi Delta Attention — high-performance KDA kernels built on CUTLASS

## News

- **2026-04-22** — Deep-Dive Blog: the design decisions behind FlashKDA v1, read it [here](docs/20260420-flashkda-v1-deep-dive.md).

## Requirements
- SM90 and above
- CUDA 12.9 and above
- PyTorch 2.4 and above

## Installation
```bash
git clone https://github.com/MoonshotAI/FlashKDA.git flash-kda
cd flash-kda
git submodule update --init --recursive
pip install -v --no-build-isolation .
```

By default, the build detects the current CUDA device and compiles for that architecture. For wheel or CI builds, compile all supported architectures explicitly:

```bash
FLASH_KDA_CUDA_ARCHS=all pip install -v --no-build-isolation .
```

Supported values are `auto` (default), `all`, or a comma-separated arch list such as `90a,100a`.

### Optional SM103 K2 register state

On SM103 (B300), build the V1a and V1aE specializations with:

```bash
FLASH_KDA_CUDA_ARCHS=103a FLASH_KDA_ENABLE_V1A=1 FLASH_KDA_ENABLE_V1AE=1 \
  pip install -v --no-build-isolation .
```

`FLASH_KDA_ENABLE_V1A` enables persistent register state; `FLASH_KDA_ENABLE_V1AE`
enables the same recurrence with a one-time shared-memory/TMA final-state store.
Both flags default to off and may be enabled independently. The optimization
changes recurrent-state dataflow, preserving the existing MMA math.

Set `FLASH_KDA_K2_IMPL=auto` at runtime to select by workload, or use `baseline`,
`v1a`, or `v1ae` explicitly. Unset defaults to `baseline`, even when both
specializations are built. V1a/V1aE require SM103, fixed lengths, and both BF16
`initial_state` and `final_state`. Unsupported or uncompiled explicit selections
raise an error; `auto` falls back to baseline when its preferred path is unavailable.

The H=64, B1–B8 thresholds in [`csrc/k2_dispatch.h`](csrc/k2_dispatch.h) are
empirical B300 policy. For B5–B7, `>320` chunks selecting V1a is a conservative,
unvalidated fallback, not a measured crossover. At H=64, B8 with `>=320` chunks
and all B>8 workloads use baseline. For H!=64, supported calls use V1a at
`>=128` chunks and baseline below that. A chunk is 16 tokens; the policy counts
`ceil(T/16)` tiles per sequence, including a partial final tile. These thresholds
are not claims for other GPUs.

## Using FlashKDA as an FLA backend

Once installed, FlashKDA is auto-dispatched from `flash-linear-attention`'s `chunk_kda`. See [fla-org/flash-linear-attention#852](https://github.com/fla-org/flash-linear-attention/pull/852) for integration details.

**Requirements**

1. Install `flash-linear-attention >= 0.5.0`:
   ```bash
   pip install -U flash-linear-attention
   ```
2. Call `chunk_kda` under `torch.inference_mode()` 
   ```python
   import torch
   from fla.ops.kda import chunk_kda

   with torch.inference_mode():
       out, final_state = chunk_kda(
           q=q, k=k, v=v, g=g, beta=beta,
           scale=scale,
           initial_state=h0,
           output_final_state=True,
           use_gate_in_kernel=True,
           use_qk_l2norm_in_kernel=True,
           use_beta_sigmoid_in_kernel=True,
           safe_gate=True,
           A_log=A_log, dt_bias=dt_bias,
           lower_bound=lower_bound,
           transpose_state_layout=True,
           cu_seqlens=cu_seqlens,
       )
   ```

**Opt out:** set `FLA_FLASH_KDA=0` to fall back to the Triton path.

**Debug dispatch:** add `logging.basicConfig(level=logging.INFO)` to see `[FLA Backend] kda.chunk_kda -> flashkda` on hit, or `... rejected: <reason>` on miss.

## Performance

See [BENCHMARK_H20.md](BENCHMARK_H20.md).

After the SM103 build above, compare all four K2 selections on B1/T8192,
B4/T2048, and B8/T1024 (H=64, D=128):

```bash
python benchmarks/bench_sm103_dispatch.py --warmup 30 --iters 200 --seed 42
```

The script uses identical seeded inputs, checks exact output/final-state equality
and unchanged initial state, and prints mean/median CUDA-event latency in
microseconds plus the expected AUTO selection. Timing calls the public
`flash_kda.fwd` wrapper, including its workspace allocation; checks are untimed.

## Tests

```bash
bash tests/test.sh
```

- `tests/test_fwd.py` — correctness tests (exact match against the torch reference; compared with `flash-linear-attention`)

Host-only K2 policy tests require Python and a C++17 GCC/Clang compiler, with
no PyTorch or CUDA dependency (`CXX` may name the compiler executable):

```bash
python tests/test_k2_dispatch.py
```

The runner compiles into a temporary directory and also supports pytest.
After building both specializations, opt into SM103 parity checks with:

```bash
FLASH_KDA_TEST_SM103=1 python -m pytest tests/test_sm103_k2.py -q
```

These GPU tests skip by default and on other architectures. Run the existing
full regression separately under `FLASH_KDA_K2_IMPL=baseline` and `auto` using
`python -m pytest tests/test_fwd_full.py -q`.


## Kernel API

### `flash_kda.fwd`

```python
flash_kda.fwd(q, k, v, g, beta, scale, out, A_log, dt_bias, lower_bound,
              initial_state=None, final_state=None, cu_seqlens=None)
```

**Parameters:**

| Parameter | Dtype | Shape | Description |
|---|---|---|---|
| `q` | bf16 | `[B, T, H, K]` | Query |
| `k` | bf16 | `[B, T, H, K]` | Key |
| `v` | bf16 | `[B, T, H, V]` | Value |
| `g` | bf16 | `[B, T, H, K]` | Gate before activation |
| `beta` | bf16 | `[B, T, H]` | Beta logits (pre-activation; sigmoid applied internally) |
| `scale` | float | scalar | scaling factor |
| `out` | bf16 | `[B, T, H, V]` | Output tensor |
| `A_log` | fp32 | `[H]` | Log-gate parameter |
| `dt_bias` | fp32 | `[H, K]` | Gate bias |
| `lower_bound` | float | scalar | Gate lower bound (range from -5.0 to 0) |
| `initial_state` | bf16/fp32/None | `[B, H, V, K]` or `[N, H, V, K]` | (optional) Initial recurrent state |
| `final_state` | bf16/fp32/None | `[B, H, V, K]` or `[N, H, V, K]` | (optional, output) Final recurrent state |
| `cu_seqlens` | int64 | `[N+1]` | (optional) Cumulative sequence lengths for variable-length batching |

- Currently requires `K = V = 128`.
- `initial_state` / `final_state` accept `None` (stateless), bf16, or fp32 tensors. When both are provided, their dtypes must match.
- When `cu_seqlens` is provided, `B` must be 1, `T` is the total length across all sequences, and `initial_state` / `final_state` have shape `[N, H, V, K]`.
- When `cu_seqlens` is `None`, each batch element is treated as an independent sequence, and the state shape is `[B, H, V, K]`.

## Development

To set up IntelliSense (clangd) for the CUDA/C++ sources, run:

```bash
bash setup_clangd.sh
```

This generates a `.clangd` file with the correct repository paths and installs the global clangd `config.yaml` to `~/.config/clangd/`.

## Citation

```bibtex
@misc{flashkda2026,
      title={FlashKDA: Flash Kimi Delta Attention},
      author={Yutian Chen, Zhiyuan Li, Yucheng Wang, Ming Wei},
      year={2026},
      publisher = {GitHub},
      howpublished = {\url{https://github.com/MoonshotAI/FlashKDA}},
}
```
