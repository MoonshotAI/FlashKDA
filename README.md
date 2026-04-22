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
pip install -v .
```

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

## Tests

```bash
bash tests/test.sh
```

- `tests/test_fwd.py` — correctness tests (exact match against the torch reference; compared with `flash-linear-attention`)


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
