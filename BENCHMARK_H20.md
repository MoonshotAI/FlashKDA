# KDA forward benchmark (Hopper / H20)

- Generated: 2026-04-20
- Command: `python benchmarks/generate_benchmark_hopper_h20_md.py`
- Benchmark settings: `warmup=30`, `iters=200`, `repeats=5`
- `fla_chunk_kda` configuration: `use_gate_in_kernel=True`, `use_qk_l2norm_in_kernel=True`, `use_beta_sigmoid_in_kernel=True`, `lower_bound=-5`, `transpose_state_layout=True`

### `T=8192`, `H=96`, `D=128`


| Case                                                 | `flash_kda` mean (ms) | `fla_chunk_kda` mean (ms) | Speedup |
| ---------------------------------------------------- | --------------------- | ------------------------- | ------- |
| Fixed                                                | 2.6219                | 4.5052                    | 1.72×   |
| Varlen, `seq_lens`=[1300, 547, 2048, 963, 271, 3063] | 2.3420                | 4.5717                    | 1.95×   |
| Varlen, `seq_lens`=`1024 x 8`                        | 2.0100                | 4.4668                    | 2.22×   |


### `T=8192`, `H=64`, `D=128`


| Case                                                 | `flash_kda` mean (ms) | `fla_chunk_kda` mean (ms) | Speedup |
| ---------------------------------------------------- | --------------------- | ------------------------- | ------- |
| Fixed                                                | 1.6199                | 2.9587                    | 1.83×   |
| Varlen, `seq_lens`=[1300, 547, 2048, 963, 271, 3063] | 1.7027                | 3.0595                    | 1.80×   |
| Varlen, `seq_lens`=`1024 x 8`                        | 1.3930                | 3.0412                    | 2.18×   |


