#!/usr/bin/env python3
"""
Submit OpenAI Batch API jobs for paper-tier experiments.

WHY: Full experiments with hundreds/thousands of examples are cheaper via
Batch API (50% discount) and don't hit rate limits.

USAGE:
    python scripts/batch_submit.py --config configs/experiment_2_3.yaml --run-id exp2_3_batch
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import typer  # noqa: E402
import yaml  # noqa: E402

app = typer.Typer(help="Submit Batch API jobs")


@app.command()
def submit(
    config: str = typer.Option(..., "--config", "-c"),
    run_id: str = typer.Option(None, "--run-id", "-r"),
) -> None:
    """Submit a batch job to OpenAI Batch API."""
    with open(config, "r") as f:
        cfg = yaml.safe_load(f)

    experiment_name = cfg.get("experiment_name", Path(config).stem)
    if run_id is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        run_id = f"{experiment_name}_batch_{timestamp}"

    run_dir = Path("runs") / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Batch] Preparing batch job for {experiment_name}")
    print(f"[Batch] Run directory: {run_dir}")
    print(f"[Batch] NOTE: This is a placeholder — implement batch request JSONL generation")
    print(f"[Batch]       based on the pipeline stages needed for this config.")

    # Save config
    with open(run_dir / "config.yaml", "w") as f:
        yaml.dump(cfg, f)

    # TODO: Generate batch request JSONL file
    # TODO: Upload to OpenAI Files API
    # TODO: Create batch job
    # TODO: Save batch job ID to run_dir/batch_info.json

    print(f"[Batch] Batch submission placeholder complete.")


if __name__ == "__main__":
    app()
