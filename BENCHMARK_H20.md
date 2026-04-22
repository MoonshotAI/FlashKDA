# KDA forward benchmark (Hopper / H20)

- Generated: 2026-04-22

- Command: `python benchmarks/generate_benchmark_hopper_h20_md.py`

- Benchmark settings: `warmup=30`, `iters=200`, `repeats=5`

- `fla_chunk_kda` configuration: `use_gate_in_kernel=True`, `use_qk_l2norm_in_kernel=True`, `use_beta_sigmoid_in_kernel=True`, `lower_bound=-5`, `transpose_state_layout=True`
- `fla_chunk_gated_delta_rule` configuration: scalar per-head gate `g` of shape `(1, T, H)`, `use_qk_l2norm_in_kernel=True`, `transpose_state_layout=True`

### `T=8192`, `H=96`, `D=128`

| Case | `flash_kda` mean (ms) | `fla_chunk_kda` mean (ms) | Speedup vs `chunk_kda` | `fla_chunk_gdn` mean (ms) | Speedup vs `gdn` |
|------|----------------------:|----------------------:|--------:|----------------------:|--------:|
| Fixed | 2.6220 | 4.8388 | 1.85× | 3.1985 | 1.22× |
| Varlen, `seq_lens`=[1300, 547, 2048, 963, 271, 3063] | 2.3449 | 4.8291 | 2.06× | 3.0541 | 1.30× |
| Varlen, `seq_lens`=`1024 x 8` | 2.0432 | 4.6723 | 2.29× | 2.9117 | 1.43× |

### `T=8192`, `H=64`, `D=128`

| Case | `flash_kda` mean (ms) | `fla_chunk_kda` mean (ms) | Speedup vs `chunk_kda` | `fla_chunk_gdn` mean (ms) | Speedup vs `gdn` |
|------|----------------------:|----------------------:|--------:|----------------------:|--------:|
| Fixed | 1.6217 | 3.1659 | 1.95× | 2.0062 | 1.24× |
| Varlen, `seq_lens`=[1300, 547, 2048, 963, 271, 3063] | 1.7060 | 3.2551 | 1.91× | 1.9986 | 1.17× |
| Varlen, `seq_lens`=`1024 x 8` | 1.3951 | 3.2175 | 2.31× | 1.9568 | 1.40× |
