#!/usr/bin/env python3
"""
Build FAISS embedding index from processed dataset.

USAGE:
    python scripts/build_embeddings_index.py --dataset healthcontradict
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Must be imported first — loads .env and sets offline env vars
import src.paper4.utils.env_setup  # noqa: F401, E402

import typer

from src.paper4.retrieval.embedding_index import EmbeddingIndex
from src.paper4.llm.openai_client import CachedOpenAIClient
from src.paper4.data.data_loaders import load_dataset_jsonl

app = typer.Typer(help="Build FAISS embedding index")


@app.command()
def build(
    dataset: str = typer.Option("healthcontradict", "--dataset", "-d"),
    data_processed: str = typer.Option("./data_processed"),
    output_dir: str = typer.Option("./models", "--output-dir"),
    batch_size: int = typer.Option(100, "--batch-size"),
) -> None:
    """Build FAISS embedding index using OpenAI text-embedding-3-large."""
    dataset_path = Path(data_processed) / f"{dataset}.jsonl"
    if not dataset_path.exists():
        print(f"[ERROR] Dataset not found: {dataset_path}")
        raise typer.Exit(1)

    print(f"[Embeddings] Loading {dataset_path}...")
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
    print(f"[Embeddings] {len(documents)} unique documents from {len(examples)} examples")

    client = CachedOpenAIClient(cache_dir="./cache")
    index = EmbeddingIndex()
    index.build(documents, client, batch_size=batch_size)

    output_path = Path(output_dir) / f"embeddings_{dataset}"
    index.save(output_path)

    client.print_cost_summary()
    print(f"[Embeddings] Done! Index saved to {output_path}")


if __name__ == "__main__":
    app()
