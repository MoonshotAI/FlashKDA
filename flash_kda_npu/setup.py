import os
import platform
from pathlib import Path

from setuptools import setup
from setuptools.command.build_py import build_py
from torch.utils.cpp_extension import BuildExtension, CppExtension


this_dir = Path(__file__).resolve().parent
ascend_home = os.environ.get("ASCEND_HOME_PATH")
if not ascend_home:
    raise RuntimeError(
        "Source CANN's set_env.sh to set ASCEND_HOME_PATH before building."
    )
ascend_platform = Path(ascend_home) / (platform.machine() + "-linux")


class NpuBuildPy(build_py):
    def find_package_modules(self, package, package_dir):
        return [
            module for module in super().find_package_modules(package, package_dir)
            if module[1] != "setup"
        ]


setup(
    name="flash-kda-npu",
    version=os.environ.get("FLASH_KDA_VERSION", "0.0.1"),
    description="FlashKDA: Flash Kimi Delta Attention for Ascend NPUs",
    long_description=(this_dir / "README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    packages=["flash_kda_npu", "flash_kda"],
    package_dir={"flash_kda_npu": ".", "flash_kda": "compat/flash_kda"},
    package_data={
        "flash_kda_npu": [
            "csrc/ascendc/*.cpp",
            "csrc/ascendc950/*.cpp",
            "csrc/ascendc950/*.h",
        ],
    },
    include_package_data=False,
    ext_modules=[
        CppExtension(
            name="flash_kda_npu_C",
            sources=["csrc/flash_kda.cpp"],
            include_dirs=[str(ascend_platform / "include")],
            library_dirs=[str(ascend_platform / "lib64")],
            libraries=["ascendcl", "acl_rtc"],
            extra_compile_args={"cxx": ["-O3", "-std=c++17"]},
        ),
    ],
    cmdclass={"build_ext": BuildExtension, "build_py": NpuBuildPy},
    zip_safe=False,
)
