import os
import subprocess
from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension, CUDA_HOME

this_dir = os.path.dirname(os.path.abspath(__file__))
subprocess.run(["git", "submodule", "update", "--init", "cutlass"])


def is_flag_set(flag: str) -> bool:
    return os.getenv(flag, "FALSE").lower() in ["true", "1", "y", "yes"]


def get_nvcc_thread_args():
    nvcc_threads = os.getenv("NVCC_THREADS") or "32"
    return ["--threads", nvcc_threads]


SUPPORTED_CUDA_ARCHS = ["90a", "100a", "103a", "120a"]


def detect_cuda_arch():
    import torch

    if not torch.cuda.is_available():
        return None

    major, minor = torch.cuda.get_device_capability(torch.cuda.current_device())
    return f"{major}{minor}a"


def get_arch_flags():
    assert CUDA_HOME is not None, "PyTorch must be compiled with CUDA support"

    requested = os.getenv("FLASH_KDA_CUDA_ARCHS", "auto").lower()
    if requested == "auto":
        arch = detect_cuda_arch()
        if arch is None:
            raise RuntimeError(
                "FLASH_KDA_CUDA_ARCHS=auto requires a visible CUDA device. "
                "Set FLASH_KDA_CUDA_ARCHS=all to build all supported archs."
            )
        archs = [arch]
    elif requested == "all":
        archs = SUPPORTED_CUDA_ARCHS
    else:
        archs = [arch.strip() for arch in requested.split(",") if arch.strip()]

    flags = []
    for arch in archs:
        flags.extend(["-gencode", f"arch=compute_{arch},code=sm_{arch}"])
    return flags


ext_modules = [
    CUDAExtension(
        name='flash_kda_C',
        sources=[
            'csrc/flash_kda.cpp',
            'csrc/smxx/fwd_launch.cu',
        ],
        include_dirs=[
            os.path.join(this_dir, 'cutlass', 'include'),
            os.path.join(this_dir, 'cutlass', 'examples', 'common'),
            os.path.join(this_dir, 'cutlass', 'tools', 'util', 'include'),
            os.path.join(this_dir, 'csrc'),
        ],
        extra_compile_args={
            'cxx': ['-O3', '-Wno-psabi'],
            'nvcc': [
                '-O3',
                '-U__CUDA_NO_HALF_OPERATORS__',
                '-U__CUDA_NO_HALF_CONVERSIONS__',
                '-U__CUDA_NO_HALF2_OPERATORS__',
                '-U__CUDA_NO_BFLOAT16_CONVERSIONS__',
                '--expt-relaxed-constexpr',
                '--expt-extended-lambda',
                '--use_fast_math',
                '--ptxas-options=-v,--register-usage-level=10,--warn-on-spills',
                '-lineinfo',
                *get_nvcc_thread_args(),
                *get_arch_flags(),
            ],
        },
    )
]
cmdclass = {"build_ext": BuildExtension}

rev = os.getenv("FLASH_KDA_VERSION_SUFFIX", "")
if not rev:
    try:
        cmd = ["git", "rev-parse", "--short", "HEAD"]
        rev = "+" + subprocess.check_output(cmd, cwd=this_dir).decode("ascii").rstrip()
    except Exception:
        rev = ""

setup(
    name='flash_kda',
    version='0.0.1' + rev,
    description='FlashKDA: Flash Kimi Delta Attention',
    ext_modules=ext_modules,
    packages=['flash_kda'],
    cmdclass=cmdclass,
    zip_safe=False,
)
