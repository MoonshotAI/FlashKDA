"""Host-only dispatch test: python tests/test_k2_dispatch.py (no torch/CUDA)."""

import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


class TestK2Dispatch(unittest.TestCase):
    def test_dispatch(self):
        compiler = shutil.which(os.environ.get("CXX", "c++"))
        if compiler is None:
            self.skipTest("requires a C++17 GCC/Clang compiler; set CXX to its executable")
        root = Path(__file__).resolve().parents[1]
        env = os.environ.copy()
        env["PATH"] = str(Path(compiler).parent) + os.pathsep + env.get("PATH", "")
        with tempfile.TemporaryDirectory(prefix="flashkda-dispatch-") as tmp:
            executable = Path(tmp) / ("dispatch.exe" if os.name == "nt" else "dispatch")
            build = subprocess.run(
                [compiler, "-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror",
                 "-I", str(root / "csrc"), str(root / "tests/test_k2_dispatch.cpp"),
                 "-o", str(executable)],
                capture_output=True, text=True, env=env, timeout=60,
            )
            self.assertEqual(build.returncode, 0, build.stdout + build.stderr)
            run = subprocess.run(
                [str(executable)], capture_output=True, text=True, env=env, timeout=10,
            )
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
            self.assertIn("K2 dispatch tests passed", run.stdout)
            print(run.stdout.strip())


if __name__ == "__main__":
    unittest.main()
