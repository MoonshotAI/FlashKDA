set -e
pip install -e .
pip install flash-linear-attention matplotlib
python tests/test_fwd.py
