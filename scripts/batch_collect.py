#!/usr/bin/env python3
"""
Collect results from completed OpenAI Batch API jobs.

USAGE:
    python scripts/batch_collect.py --run-id exp2_3_batch
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import typer  # noqa: E402

app = typer.Typer(help="Collect Batch API results")


@app.command()
def collect(
    run_id: str = typer.Option(..., "--run-id", "-r"),
    runs_dir: str = typer.Option("./runs", "--runs-dir"),
) -> None:
    """Collect results from a completed batch job."""
    run_dir = Path(runs_dir) / run_id
    batch_info_path = run_dir / "batch_info.json"

    if not batch_info_path.exists():
        print(f"[ERROR] No batch_info.json in {run_dir}")
        print(f"  Submit a batch first: python scripts/batch_submit.py")
        raise typer.Exit(1)

    print(f"[Batch] Collecting results for {run_id}")
    print(f"[Batch] NOTE: This is a placeholder — implement batch result retrieval")

    # TODO: Read batch job ID from batch_info.json
    # TODO: Check batch status via OpenAI API
    # TODO: Download results
    # TODO: Parse into per_example.jsonl format
    # TODO: Populate the SQLite cache with batch responses


if __name__ == "__main__":
    app()
