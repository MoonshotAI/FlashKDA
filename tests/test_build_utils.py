import subprocess
from unittest import mock

import pytest

from build_utils import update_cutlass_submodule


def test_update_cutlass_submodule_runs_from_repository(tmp_path):
    with mock.patch("build_utils.subprocess.run") as run:
        update_cutlass_submodule(str(tmp_path))

    run.assert_called_once_with(
        ["git", "submodule", "update", "--init", "cutlass"],
        cwd=str(tmp_path),
        check=True,
    )


def test_update_cutlass_submodule_reports_git_failure(tmp_path):
    error = subprocess.CalledProcessError(128, ["git", "submodule"])

    with mock.patch("build_utils.subprocess.run", side_effect=error):
        with pytest.raises(
            RuntimeError,
            match=r"Failed to initialize the CUTLASS submodule "
            r"\(Git exited with status 128\).*git submodule update",
        ):
            update_cutlass_submodule(str(tmp_path))


def test_update_cutlass_submodule_reports_missing_git(tmp_path):
    with mock.patch(
        "build_utils.subprocess.run",
        side_effect=FileNotFoundError("git"),
    ):
        with pytest.raises(
            RuntimeError,
            match=r"CUTLASS submodule because Git was not found",
        ):
            update_cutlass_submodule(str(tmp_path))
