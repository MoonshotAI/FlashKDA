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


def get_arch_flags():
    assert CUDA_HOME is not None, "PyTorch must be compiled with CUDA support"
    DISABLE_SM90 = is_flag_set("FLASH_KDA_DISABLE_SM90")
    arch_flags = []
    if not DISABLE_SM90:
        arch_flags.extend(["-gencode", "arch=compute_90a,code=sm_90a"])
    return arch_flags


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
