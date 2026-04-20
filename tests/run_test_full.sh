set -e
pip install -e .
pip install pytest pytest-xdist
cd tests && FLASH_KDA_DIST_GPU=1 pytest test_fwd_full.py -x -v -n 16
