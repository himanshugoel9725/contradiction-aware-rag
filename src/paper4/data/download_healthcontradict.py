"""
Download HealthContradict dataset.

Clones the GitHub repo (tinaboya/HealthContradict) to data_raw/healthcontradict/.
If git lfs is available, pulls LFS objects for dataset_ready.jsonl.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

HC_REPO_URL = "https://github.com/tinaboya/HealthContradict.git"


def download_healthcontradict(
    output_dir: str | Path = "./data_raw",
    force: bool = False,
) -> Path:
    """
    Clone HealthContradict repo.

    Args:
        output_dir: Parent directory. Creates healthcontradict/ subdirectory.
        force: If True, delete existing and re-clone.

    Returns:
        Path to the cloned directory.
    """
    output_dir = Path(output_dir)
    clone_dir = output_dir / "healthcontradict"

    if clone_dir.exists() and not force:
        print(f"[HealthContradict] Already cloned at {clone_dir}")
        return clone_dir

    if clone_dir.exists() and force:
        shutil.rmtree(clone_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Clone
    print(f"[HealthContradict] Cloning {HC_REPO_URL} ...")
    subprocess.run(
        ["git", "clone", "--depth", "1", HC_REPO_URL, str(clone_dir)],
        check=True,
    )

    # Try git lfs pull for dataset_ready.jsonl
    if shutil.which("git-lfs") or shutil.which("git"):
        try:
            print("[HealthContradict] Attempting git lfs pull ...")
            subprocess.run(
                ["git", "lfs", "pull"],
                cwd=str(clone_dir),
                check=True,
                timeout=120,
            )
            print("[HealthContradict] LFS pull complete")
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired) as e:
            print(f"[HealthContradict] LFS pull failed ({e}); will use manual assembly fallback")

    # Verify
    key_files = ["query-all.jsonl", "doc-all-stance.csv", "doc-contradict.csv", "dataset_ready.jsonl"]
    found = [f for f in key_files if (clone_dir / f).exists() or list(clone_dir.rglob(f))]
    print(f"[HealthContradict] Found files: {found}")

    return clone_dir


if __name__ == "__main__":
    download_healthcontradict(force="--force" in sys.argv)
