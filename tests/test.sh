set -e
pip install -e .
pip install "flash-linear-attention>=0.5.0" matplotlib pytest
pytest -q tests/test_python_api.py
python tests/test_fwd.py
