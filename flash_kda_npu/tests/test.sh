#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."

if [ -f "${ASCEND_HOME_PATH:-/usr/local/Ascend/ascend-toolkit/latest}/set_env.sh" ]; then
    source "${ASCEND_HOME_PATH:-/usr/local/Ascend/ascend-toolkit/latest}/set_env.sh"
fi

python -m pip install -e ./flash_kda_npu --no-build-isolation --no-deps
python -m pip install pytest
python -m pytest flash_kda_npu/tests/test_fwd.py \
    flash_kda_npu/tests/test_fwd_full.py::test_cuda_smoke -x -v "$@"
