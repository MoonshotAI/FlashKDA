# KDA training benchmark (fwd / fwd+bwd) (Blackwell / RTX 5090)

- Generated: 2026-08-08

- Command: `python benchmarks/generate_train_benchmark_md.py`

- Benchmark settings: `warmup=30`, `iters=200`, `repeats=5`

- `fla_chunk_kda` configuration: `use_gate_in_kernel=True`, `use_qk_l2norm_in_kernel=True`, post-sigmoid `beta`, `lower_bound=-5`, fp32 `initial_state`
- `flash_kda_train` configuration: `flash_kda.train.chunk_kda_train_fwd`/`chunk_kda_train_bwd`, `use_gate_in_kernel=True`, post-sigmoid `beta`, `lower_bound=-5`, fp32 `initial_state`, `chunk_size=64`; q/k l2-normalized inside the timed region (matches `use_qk_l2norm_in_kernel`)

### `T=8192`, `H=96`, `D=128`

| Case | `flash_kda_train` fwd (ms) | `fla_chunk_kda` fwd (ms) | fwd speedup | `flash_kda_train` fwd+bwd (ms) | `fla_chunk_kda` fwd+bwd (ms) | fwd+bwd speedup |
|------|------------------:|------------------:|--------:|------------------:|------------------:|--------:|
| Fixed | 6.1385 | 5.4185 | 0.88× | 21.2520 | 23.4677 | 1.10× |
| Varlen, `seq_lens`=[1300, 547, 2048, 963, 271, 3063] | 6.1990 | 5.4671 | 0.88× | 21.3378 | 23.0464 | 1.08× |
| Varlen, `seq_lens`=`1024 x 8` | 6.1546 | 5.4465 | 0.88× | 21.1676 | 23.0427 | 1.09× |

### `T=8192`, `H=64`, `D=128`

| Case | `flash_kda_train` fwd (ms) | `fla_chunk_kda` fwd (ms) | fwd speedup | `flash_kda_train` fwd+bwd (ms) | `fla_chunk_kda` fwd+bwd (ms) | fwd+bwd speedup |
|------|------------------:|------------------:|--------:|------------------:|------------------:|--------:|
| Fixed | 4.0432 | 3.5908 | 0.89× | 13.9893 | 15.3251 | 1.10× |
| Varlen, `seq_lens`=[1300, 547, 2048, 963, 271, 3063] | 4.1186 | 3.5776 | 0.87× | 14.1548 | 15.0392 | 1.06× |
| Varlen, `seq_lens`=`1024 x 8` | 4.0600 | 3.5983 | 0.89× | 13.9681 | 15.1306 | 1.08× |
