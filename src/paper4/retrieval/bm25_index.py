"""
BM25 sparse retrieval index.

WHY: BM25 is the standard sparse baseline. It complements dense embeddings
by excelling at exact keyword matches (e.g. drug names, gene symbols).

HOW: Uses rank-bm25 library. The index is serialized to disk with pickle
so it only needs to be built once per corpus.
"""

from __future__ import annotations

import json
import pickle
import re
from pathlib import Path
from typing import Any

from rank_bm25 import BM25Okapi


def _tokenize(text: str) -> list[str]:
    """Simple whitespace + lowercased tokenizer. Strips punctuation."""
    return re.findall(r"\w+", text.lower())


class BM25Index:
    """
    BM25 retrieval index for a document corpus.

    Attributes:
        doc_ids: Ordered list of document IDs matching the index.
        doc_texts: Original document texts (for display in results).
    """

    def __init__(self) -> None:
        self._index: BM25Okapi | None = None
        self.doc_ids: list[str] = []
        self.doc_texts: list[str] = []
        self.doc_titles: list[str] = []

    def build(self, documents: list[dict[str, Any]]) -> None:
        """
        Build the BM25 index from a list of documents.

        Args:
            documents: List of dicts with at least 'doc_id' and 'text' keys.
        """
        self.doc_ids = [d["doc_id"] for d in documents]
        self.doc_texts = [d.get("text", "") for d in documents]
        self.doc_titles = [d.get("title", "") for d in documents]

        tokenized = [_tokenize(text) for text in self.doc_texts]
        self._index = BM25Okapi(tokenized)
        print(f"[BM25] Built index with {len(self.doc_ids)} documents")

    def search(self, query: str, top_k: int = 10) -> list[dict[str, Any]]:
        """
        Retrieve top-k documents for a query.

        Returns:
            List of dicts with doc_id, title, text, score — sorted by score descending.
        """
        if self._index is None:
            raise RuntimeError("BM25 index not built. Call build() first.")

        tokenized_query = _tokenize(query)
        scores = self._index.get_scores(tokenized_query)

        # Get top-k indices
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        results = []
        for idx in top_indices:
            results.append({
                "doc_id": self.doc_ids[idx],
                "title": self.doc_titles[idx],
                "text": self.doc_texts[idx],
                "score": float(scores[idx]),
                "source": "bm25",
            })
        return results

    def save(self, path: str | Path) -> None:
        """Serialize index to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "index": self._index,
                    "doc_ids": self.doc_ids,
                    "doc_texts": self.doc_texts,
                    "doc_titles": self.doc_titles,
                },
                f,
            )
        print(f"[BM25] Saved index to {path}")

    def load(self, path: str | Path) -> None:
        """Load index from disk."""
        path = Path(path)
        with open(path, "rb") as f:
            data = pickle.load(f)  # noqa: S301
        self._index = data["index"]
        self.doc_ids = data["doc_ids"]
        self.doc_texts = data["doc_texts"]
        self.doc_titles = data.get("doc_titles", [""] * len(self.doc_ids))
        print(f"[BM25] Loaded index with {len(self.doc_ids)} documents")

    @property
    def size(self) -> int:
        return len(self.doc_ids)
