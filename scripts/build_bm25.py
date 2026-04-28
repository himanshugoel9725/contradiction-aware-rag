#!/usr/bin/env python3
"""
Build BM25 index from processed dataset.

USAGE:
    python scripts/build_bm25.py --dataset healthcontradict
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import typer

from src.paper4.retrieval.bm25_index import BM25Index
from src.paper4.data.data_loaders import load_dataset_jsonl

app = typer.Typer(help="Build BM25 index")


@app.command()
def build(
    dataset: str = typer.Option("healthcontradict", "--dataset", "-d"),
    data_processed: str = typer.Option("./data_processed"),
    output_dir: str = typer.Option("./models", "--output-dir"),
) -> None:
    """Build BM25 index from a processed dataset."""
    dataset_path = Path(data_processed) / f"{dataset}.jsonl"
    if not dataset_path.exists():
        print(f"[ERROR] Dataset not found: {dataset_path}")
        raise typer.Exit(1)

    print(f"[BM25] Loading {dataset_path}...")
    examples = load_dataset_jsonl(dataset_path)

    # Extract all unique documents
    doc_map: dict[str, dict] = {}
    for ex in examples:
        for doc in ex.documents:
            if doc.doc_id not in doc_map:
                doc_map[doc.doc_id] = {
                    "doc_id": doc.doc_id,
                    "title": doc.title,
                    "text": doc.text,
                }

    documents = list(doc_map.values())
    print(f"[BM25] {len(documents)} unique documents from {len(examples)} examples")

    index = BM25Index()
    index.build(documents)

    output_path = Path(output_dir) / f"bm25_{dataset}.pkl"
    index.save(output_path)
    print(f"[BM25] Done! Index saved to {output_path}")


if __name__ == "__main__":
    app()
