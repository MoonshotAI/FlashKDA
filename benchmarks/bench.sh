set -e
pip install -e .
pip install flash-linear-attention matplotlib
python benchmarks/bench_fwd.py
