# KDA forward benchmark (Blackwell / GB200)

- Generated: 2026-05-26

- Command: `python benchmarks/generate_benchmark_md.py -o BENCHMARK_GB200.md --device-label Blackwell / GB200`

- Benchmark settings: `warmup=30`, `iters=200`, `repeats=5`

- `fla_chunk_kda` configuration: `use_gate_in_kernel=True`, `use_qk_l2norm_in_kernel=True`, `use_beta_sigmoid_in_kernel=True`, `lower_bound=-5`, `transpose_state_layout=True`
- `fla_chunk_gated_delta_rule` configuration: scalar per-head gate `g` of shape `(1, T, H)`, `use_qk_l2norm_in_kernel=True`, `transpose_state_layout=True`

### `T=8192`, `H=96`, `D=128`

| Case | `flash_kda` mean (ms) | `fla_chunk_kda` mean (ms) | Speedup vs `chunk_kda` | `fla_chunk_gdn` mean (ms) | Speedup vs `gdn` |
|------|----------------------:|----------------------:|--------:|----------------------:|--------:|
| Fixed | 1.0087 | 2.3271 | 2.31× | 1.2792 | 1.27× |
| Varlen, `seq_lens`=[1300, 547, 2048, 963, 271, 3063] | 0.8597 | 2.3340 | 2.71× | 1.2962 | 1.51× |
| Varlen, `seq_lens`=`1024 x 8` | 0.7064 | 2.3105 | 3.27× | 1.2744 | 1.80× |

### `T=8192`, `H=64`, `D=128`

| Case | `flash_kda` mean (ms) | `fla_chunk_kda` mean (ms) | Speedup vs `chunk_kda` | `fla_chunk_gdn` mean (ms) | Speedup vs `gdn` |
|------|----------------------:|----------------------:|--------:|----------------------:|--------:|
| Fixed | 0.9247 | 1.5764 | 1.70× | 0.8857 | 0.96× |
| Varlen, `seq_lens`=[1300, 547, 2048, 963, 271, 3063] | 0.6553 | 1.5843 | 2.42× | 0.9053 | 1.38× |
| Varlen, `seq_lens`=`1024 x 8` | 0.4811 | 1.5422 | 3.21× | 0.8683 | 1.80× |
