import os
import subprocess
from setuptools import setup
from torch.utils.cpp_extension import CppExtension, CUDAExtension, BuildExtension, CUDA_HOME

this_dir = os.path.dirname(os.path.abspath(__file__))


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


def should_build_cuda():
    requested = os.getenv("FLASH_KDA_BUILD_CUDA", "auto").lower()
    if requested in ("0", "false", "no", "off"):
        return False
    if requested in ("1", "true", "yes", "on"):
        if CUDA_HOME is None:
            raise RuntimeError("FLASH_KDA_BUILD_CUDA=1 requires a CUDA toolkit")
        return True
    if requested != "auto":
        raise ValueError("FLASH_KDA_BUILD_CUDA must be auto, 0, or 1")
    if CUDA_HOME is None:
        return False

    import torch

    return torch.cuda.is_available() or os.getenv("FLASH_KDA_CUDA_ARCHS", "auto").lower() != "auto"


def should_build_cpu():
    requested = os.getenv("FLASH_KDA_BUILD_CPU", "1").lower()
    if requested in ("0", "false", "no", "off"):
        return False
    if requested in ("1", "true", "yes", "on"):
        return True
    raise ValueError("FLASH_KDA_BUILD_CPU must be 0 or 1")


ext_modules = []
if should_build_cpu():
    cpu_compile_args = ['/O2'] if os.name == 'nt' else ['-O3']
    ext_modules.append(
        CppExtension(
            name='flash_kda_cpu_C',
            sources=['csrc/cpu/torch_bindings.cpp'],
            extra_compile_args=cpu_compile_args,
        )
    )

if should_build_cuda():
    subprocess.run(["git", "submodule", "update", "--init", "cutlass"], check=True)
    ext_modules.append(
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
    )

cmdclass = {"build_ext": BuildExtension} if ext_modules else {}

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
