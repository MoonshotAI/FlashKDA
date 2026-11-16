#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/../.."

# Ascend hosts need the CANN runtime on LD_LIBRARY_PATH before torch can
# even import: torch_npu auto-loads as a device backend and its _C ext fails
# to resolve the runtime symbols otherwise.
if [ -f "${ASCEND_HOME_PATH:-/usr/local/Ascend/ascend-toolkit/latest}/set_env.sh" ]; then
    source "${ASCEND_HOME_PATH:-/usr/local/Ascend/ascend-toolkit/latest}/set_env.sh"
fi

python -m pip install -e ./flash_kda_npu --no-build-isolation --no-deps -q
case " $* " in
    *" --skip-fla "*) ;;
    *)
        # Use the Ascend index for FLA's pinned Triton dependency.
        python -m pip install --no-deps \
            --extra-index-url https://mirrors.huaweicloud.com/ascend/repos/pypi \
            "${TRITON_ASCEND_WHEEL:-triton-ascend==3.2.2}"
        # 0.5.2 has no Ascend GLA output backend used by chunk_kda.
        python -m pip install --no-deps \
            'https://github.com/fla-org/flash-linear-attention/archive/15396e604df6a4699950d7f20457b391b0cd8e74.tar.gz' \
            einops
        ;;
esac
FLA_FLASH_KDA=0 python -m flash_kda_npu.benchmarks.bench_fwd "$@"
