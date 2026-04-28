#!/usr/bin/env python3
"""
Package a reproducibility bundle for a completed experiment.

USAGE:
    python scripts/package_reproducibility.py --run-id <run_id>

Creates a zip containing: config, per_example.jsonl, metrics.json, figures,
env_report.json, and the SQLite cache (for exact response reproduction).
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import typer  # noqa: E402

app = typer.Typer(help="Package reproducibility bundle")


@app.command()
def package(
    run_id: str = typer.Option(..., "--run-id", "-r"),
    runs_dir: str = typer.Option("./runs", "--runs-dir"),
    include_cache: bool = typer.Option(True, "--include-cache/--no-cache"),
) -> None:
    """Create a reproducibility zip bundle."""
    run_dir = Path(runs_dir) / run_id

    if not run_dir.exists():
        print(f"[ERROR] Run directory not found: {run_dir}")
        raise typer.Exit(1)

    bundle_dir = run_dir / "_repro_bundle"
    bundle_dir.mkdir(exist_ok=True)

    # Copy run artifacts
    for artifact in ["config.yaml", "per_example.jsonl", "metrics.json"]:
        src = run_dir / artifact
        if src.exists():
            shutil.copy2(src, bundle_dir / artifact)

    # Copy figures
    fig_dir = run_dir / "figures"
    if fig_dir.exists():
        shutil.copytree(fig_dir, bundle_dir / "figures", dirs_exist_ok=True)

    # Copy env report
    env_report = Path("logs/env_report.json")
    if env_report.exists():
        shutil.copy2(env_report, bundle_dir / "env_report.json")

    # Copy requirements
    for req in ["requirements.lock.txt", "requirements-dev.lock.txt"]:
        src = Path(req)
        if src.exists():
            shutil.copy2(src, bundle_dir / req)

    # Optionally copy cache
    if include_cache:
        cache_src = Path("cache/openai_cache.db")
        if cache_src.exists():
            shutil.copy2(cache_src, bundle_dir / "openai_cache.db")

    # Create zip
    zip_path = run_dir / f"{run_id}_repro"
    shutil.make_archive(str(zip_path), "zip", bundle_dir)

    # Clean up staging dir
    shutil.rmtree(bundle_dir)

    print(f"[Repro] Bundle created: {zip_path}.zip")
    print(f"[Repro] Contents: config, results, metrics, figures, env report, cache")


if __name__ == "__main__":
    app()
