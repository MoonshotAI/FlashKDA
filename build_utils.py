import subprocess


def update_cutlass_submodule(repo_dir: str) -> None:
    command = ["git", "submodule", "update", "--init", "cutlass"]
    try:
        subprocess.run(command, cwd=repo_dir, check=True)
    except FileNotFoundError as exc:
        raise RuntimeError(
            "Failed to initialize the CUTLASS submodule because Git was not found. "
            "Install Git and run `git submodule update --init --recursive`."
        ) from exc
    except (OSError, subprocess.CalledProcessError) as exc:
        detail = (
            f" (Git exited with status {exc.returncode})"
            if isinstance(exc, subprocess.CalledProcessError)
            else ""
        )
        raise RuntimeError(
            "Failed to initialize the CUTLASS submodule"
            f"{detail}. Run `git submodule update --init --recursive` "
            "from the FlashKDA repository and retry the installation."
        ) from exc
