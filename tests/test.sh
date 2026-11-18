set -e
pip install -e .
pip install "flash-linear-attention>=0.5.1,<0.6" matplotlib
python tests/test_fwd.py
