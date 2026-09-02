"""Dev-mode per-stage extension loader.

Builds a single training stage's sources into its own torch extension with its
own build directory, so stages can be developed and tested in parallel without
rebuilding (or locking) the main `flash_kda_train_C` extension. At integration
time the same `.cu` sources are merged into `setup.py`'s `train_sources`.
"""

import os

import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_stage(name: str, sources: list[str], verbose: bool = False):
    from torch.utils.cpp_extension import load

    os.environ.setdefault("CUDA_HOME", "/usr/local/cuda")
    os.environ["PATH"] = "/usr/local/cuda/bin:" + os.environ.get("PATH", "")

    build_directory = os.path.join(_REPO_ROOT, "build_dev", name)
    os.makedirs(build_directory, exist_ok=True)

    major, minor = torch.cuda.get_device_capability(torch.cuda.current_device())
    arch = f"{major}{minor}a"

    return load(
        name=f"flash_kda_train_{name}",
        sources=[os.path.join(_REPO_ROOT, s) for s in sources],
        extra_include_paths=[
            os.path.join(_REPO_ROOT, "cutlass", "include"),
            os.path.join(_REPO_ROOT, "csrc"),
            os.path.join(_REPO_ROOT, "csrc", "train"),
        ],
        extra_cflags=["-O3", "-Wno-psabi"],
        extra_cuda_cflags=[
            "-O3",
            "-U__CUDA_NO_HALF_OPERATORS__",
            "-U__CUDA_NO_HALF_CONVERSIONS__",
            "-U__CUDA_NO_HALF2_OPERATORS__",
            "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
            "--expt-relaxed-constexpr",
            "--expt-extended-lambda",
            "--use_fast_math",
            "-lineinfo",
            "-gencode", f"arch=compute_{arch},code=sm_{arch}",
        ],
        build_directory=build_directory,
        verbose=verbose,
    )
