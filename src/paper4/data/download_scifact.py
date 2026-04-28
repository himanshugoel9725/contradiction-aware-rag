"""
Download SciFact dataset.

Downloads the official release tarball from S3, extracts to data_raw/scifact/.

Source: https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz
"""

from __future__ import annotations

import hashlib
import os
import sys
import tarfile
from pathlib import Path
from urllib.request import urlretrieve

SCIFACT_URL = "https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz"


def download_scifact(
    output_dir: str | Path = "./data_raw",
    force: bool = False,
) -> Path:
    """
    Download and extract SciFact dataset.

    Args:
        output_dir: Where to extract (creates scifact/ subdirectory).
        force: If True, re-download even if already present.

    Returns:
        Path to extracted directory.
    """
    output_dir = Path(output_dir)
    extract_dir = output_dir / "scifact"
    tarball_path = output_dir / "scifact_data.tar.gz"

    # Check if already extracted
    if (extract_dir / "corpus.jsonl").exists() and not force:
        print(f"[SciFact] Already extracted at {extract_dir}")
        return extract_dir

    output_dir.mkdir(parents=True, exist_ok=True)

    # Download
    print(f"[SciFact] Downloading from {SCIFACT_URL} ...")
    urlretrieve(SCIFACT_URL, tarball_path)
    print(f"[SciFact] Downloaded to {tarball_path} ({tarball_path.stat().st_size / 1e6:.1f} MB)")

    # Extract
    print(f"[SciFact] Extracting to {extract_dir} ...")
    extract_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tarball_path, "r:gz") as tar:
        # Security: validate member names to prevent path traversal
        for member in tar.getmembers():
            member_path = os.path.normpath(member.name)
            if member_path.startswith("..") or os.path.isabs(member_path):
                raise ValueError(f"Unsafe path in tarball: {member.name}")

        tar.extractall(path=output_dir, filter="data")

    # The tarball may extract to data/ — move files to scifact/
    data_subdir = output_dir / "data"
    if data_subdir.exists() and data_subdir != extract_dir:
        for item in data_subdir.iterdir():
            target = extract_dir / item.name
            if not target.exists():
                item.rename(target)
        if data_subdir.exists() and not any(data_subdir.iterdir()):
            data_subdir.rmdir()

    # Clean up tarball
    tarball_path.unlink(missing_ok=True)

    # Verify expected files
    expected = ["corpus.jsonl", "claims_dev.jsonl"]
    for fname in expected:
        path = extract_dir / fname
        if not path.exists():
            # Check subdirectories
            matches = list(extract_dir.rglob(fname))
            if matches:
                matches[0].rename(path)
            else:
                print(f"[SciFact] WARNING: Expected file not found: {fname}")

    print(f"[SciFact] Extraction complete: {list(extract_dir.iterdir())}")
    return extract_dir


if __name__ == "__main__":
    download_scifact(force="--force" in sys.argv)
