#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."

if [ -f /usr/local/Ascend/ascend-toolkit/latest/set_env.sh ]; then
    source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh
fi

pip install -e . --no-build-isolation
pip install pytest
python -m pytest flash_kda_npu/tests/ -x -v "$@"
