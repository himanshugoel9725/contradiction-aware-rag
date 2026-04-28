#!/usr/bin/env python3
"""
Prepare datasets — normalize raw data to unified JSONL schema.

USAGE:
    python scripts/prepare_dataset.py
    python scripts/prepare_dataset.py --dataset scifact
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import typer

from src.paper4.data.data_loaders import (
    load_scifact,
    load_healthcontradict,
    load_manconcorpus,
    save_dataset_jsonl,
)

app = typer.Typer(help="Prepare datasets for pipeline")


def _sha256_file(path: Path) -> str:
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


@app.command()
def prepare(
    data_raw: str = typer.Option("./data_raw", "--data-raw"),
    data_processed: str = typer.Option("./data_processed", "--data-processed"),
    dataset: str = typer.Option("all", "--dataset", "-d", help="scifact, healthcontradict, manconcorpus, or all"),
) -> None:
    """Normalize raw datasets to unified JSONL."""
    print("=" * 50)
    print("  Dataset Preparation")
    print("=" * 50)

    loaders = {
        "scifact": load_scifact,
        "healthcontradict": load_healthcontradict,
        "manconcorpus": load_manconcorpus,
    }

    datasets_to_process = list(loaders.keys()) if dataset == "all" else [dataset]

    for name in datasets_to_process:
        if name not in loaders:
            print(f"[WARN] Unknown dataset: {name}")
            continue

        print(f"\n[{name}] Loading...")
        try:
            examples = loaders[name](data_raw)
            output_path = Path(data_processed) / f"{name}.jsonl"
            save_dataset_jsonl(examples, output_path)

            # Checksum
            checksum = _sha256_file(output_path)
            print(f"[{name}] {len(examples)} examples → {output_path}")
            print(f"[{name}] SHA-256: {checksum}")

        except FileNotFoundError as e:
            print(f"[{name}] Skipped: {e}")
        except Exception as e:
            print(f"[{name}] ERROR: {e}")

    print("\n[Done] Check data_processed/ for normalized JSONL files.")


if __name__ == "__main__":
    app()
