set -e
cd "$(dirname "$0")/../.."

# Ascend hosts need the CANN runtime on LD_LIBRARY_PATH before torch can
# even import: torch_npu auto-loads as a device backend and its _C ext fails
# to resolve the runtime symbols otherwise.
if [ -f /usr/local/Ascend/ascend-toolkit/latest/set_env.sh ]; then
    source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh
fi

python -m pip install -e . --no-build-isolation -q
case " $* " in
    *" --skip-fla "*) ;;
    *)
        # PyPI only carries Triton Ascend 3.2.0. The official Ascend index
        # supplies 3.2.2, required by FLA, without replacing torch.
        python -m pip install --no-deps \
            --index-url https://mirrors.huaweicloud.com/ascend/repos/pypi \
            "${TRITON_ASCEND_WHEEL:-triton-ascend==3.2.2}"
        # 0.5.2 has no Ascend GLA output backend used by chunk_kda.
        python -m pip install --no-deps \
            'https://github.com/fla-org/flash-linear-attention/archive/15396e604df6a4699950d7f20457b391b0cd8e74.tar.gz' \
            einops
        ;;
esac
python flash_kda_npu/benchmarks/bench_fwd.py "$@"
