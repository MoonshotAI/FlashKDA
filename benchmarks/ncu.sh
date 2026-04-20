rm -rf report.ncu-rep
rm -rf report_varlen.ncu-rep
set -e
pip install -e .
ncu --set full --kernel-name-base function -k "regex:_flash_kda_fwd_(prepare|recurrence)" --clock-control none --import-source yes --source-folders . --export report.ncu-rep python benchmarks/bench_fwd.py --mode fixed --warmup 0 --iters 5 --repeats 1
ncu --set full --kernel-name-base function -k "regex:_flash_kda_fwd_(prepare|recurrence)" --clock-control none --import-source yes --source-folders . --export report_varlen.ncu-rep python benchmarks/bench_fwd.py --mode varlen --warmup 0 --iters 5 --repeats 1
