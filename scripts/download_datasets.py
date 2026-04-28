#!/usr/bin/env python3
"""
Download all datasets.

USAGE:
    python scripts/download_datasets.py
    python scripts/download_datasets.py --force
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import typer

from src.paper4.data.download_scifact import download_scifact
from src.paper4.data.download_healthcontradict import download_healthcontradict

app = typer.Typer(help="Download all datasets")


@app.command()
def download(
    output_dir: str = typer.Option("./data_raw", "--output-dir", "-o"),
    force: bool = typer.Option(False, "--force", help="Re-download even if present"),
) -> None:
    """Download SciFact and HealthContradict datasets."""
    print("=" * 50)
    print("  Dataset Download")
    print("=" * 50)

    print("\n[1/2] SciFact")
    try:
        download_scifact(output_dir, force=force)
    except Exception as e:
        print(f"  ERROR: {e}")

    print("\n[2/2] HealthContradict")
    try:
        download_healthcontradict(output_dir, force=force)
    except Exception as e:
        print(f"  ERROR: {e}")

    print("\n[Done] Check data_raw/ for downloaded files.")


if __name__ == "__main__":
    app()
