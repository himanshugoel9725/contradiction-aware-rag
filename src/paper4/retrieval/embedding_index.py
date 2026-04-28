"""
Dense embedding retrieval using OpenAI text-embedding-3-large + FAISS.

WHY: Dense embeddings capture semantic similarity beyond keyword overlap.
Combined with BM25 (sparse), they form a hybrid retrieval system that
covers both exact-match and meaning-based search.

HOW:
    1. Embed all documents via CachedOpenAIClient (cached in SQLite).
    2. Store vectors in a FAISS IndexFlatIP (inner product = cosine on normalized vecs).
    3. At query time, embed the query and search the FAISS index.
    4. Supports save/load so embeddings are computed once per corpus.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import faiss
import numpy as np


class EmbeddingIndex:
    """
    FAISS-based dense retrieval index backed by OpenAI embeddings.

    The client is injected so the index doesn't own the caching/cost logic.
    """

    def __init__(self, dimension: int = 3072) -> None:
        """
        Args:
            dimension: Embedding dimension. text-embedding-3-large = 3072.
        """
        self.dimension = dimension
        self._index: faiss.IndexFlatIP | None = None
        self.doc_ids: list[str] = []
        self.doc_texts: list[str] = []
        self.doc_titles: list[str] = []

    def build(
        self,
        documents: list[dict[str, Any]],
        client: Any,
        batch_size: int = 100,
    ) -> None:
        """
        Embed all documents and build the FAISS index.

        Args:
            documents: List of dicts with 'doc_id', 'text', optional 'title'.
            client: A CachedOpenAIClient instance (for embeddings API).
            batch_size: How many texts to embed per API call.
        """
        self.doc_ids = [d["doc_id"] for d in documents]
        self.doc_texts = [d.get("text", "") for d in documents]
        self.doc_titles = [d.get("title", "") for d in documents]

        print(f"[EmbeddingIndex] Embedding {len(documents)} documents ...")
        all_embeddings = client.embed(self.doc_texts, batch_size=batch_size)

        # Normalize for cosine similarity via inner product
        matrix = np.array(all_embeddings, dtype=np.float32)
        faiss.normalize_L2(matrix)

        self._index = faiss.IndexFlatIP(self.dimension)
        self._index.add(matrix)
        print(f"[EmbeddingIndex] Built FAISS index with {self._index.ntotal} vectors")

    def search(
        self,
        query: str,
        client: Any,
        top_k: int = 10,
    ) -> list[dict[str, Any]]:
        """
        Retrieve top-k documents for a query.

        Args:
            query: Natural language query text.
            client: CachedOpenAIClient for embedding the query.
            top_k: Number of results.

        Returns:
            List of dicts with doc_id, title, text, score.
        """
        if self._index is None:
            raise RuntimeError("Embedding index not built. Call build() first.")

        query_vec = np.array(client.embed([query]), dtype=np.float32)
        faiss.normalize_L2(query_vec)

        scores, indices = self._index.search(query_vec, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:  # FAISS returns -1 for missing
                continue
            results.append({
                "doc_id": self.doc_ids[idx],
                "title": self.doc_titles[idx],
                "text": self.doc_texts[idx],
                "score": float(score),
                "source": "embedding",
            })
        return results

    def save(self, path: str | Path) -> None:
        """Save FAISS index + metadata to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Save FAISS index
        faiss.write_index(self._index, str(path.with_suffix(".faiss")))

        # Save metadata
        meta = {
            "doc_ids": self.doc_ids,
            "doc_texts": self.doc_texts,
            "doc_titles": self.doc_titles,
            "dimension": self.dimension,
        }
        with open(path.with_suffix(".meta.json"), "w") as f:
            json.dump(meta, f)
        print(f"[EmbeddingIndex] Saved to {path}")

    def load(self, path: str | Path) -> None:
        """Load FAISS index + metadata from disk."""
        path = Path(path)

        self._index = faiss.read_index(str(path.with_suffix(".faiss")))

        with open(path.with_suffix(".meta.json"), "r") as f:
            meta = json.load(f)
        self.doc_ids = meta["doc_ids"]
        self.doc_texts = meta["doc_texts"]
        self.doc_titles = meta.get("doc_titles", [""] * len(self.doc_ids))
        self.dimension = meta.get("dimension", 3072)
        print(f"[EmbeddingIndex] Loaded {self._index.ntotal} vectors from {path}")

    @property
    def size(self) -> int:
        return self._index.ntotal if self._index else 0

    def search_mmr(
        self,
        query: str,
        client: Any,
        top_k: int = 10,
        lambda_param: float = 0.5,
    ) -> list[dict[str, Any]]:
        """
        Retrieve top-k documents using Maximal Marginal Relevance re-ranking.

        Retrieves top_k*3 candidates by cosine similarity then applies MMR to
        balance relevance (to query) against redundancy (among selected docs).

        Args:
            query: Natural language query text.
            client: CachedOpenAIClient for embedding the query.
            top_k: Number of results to return.
            lambda_param: MMR trade-off — 1.0 = pure relevance, 0.0 = pure diversity.

        Returns:
            List of top_k dicts with doc_id, title, text, score, source="mmr".
        """
        if self._index is None:
            raise RuntimeError("Embedding index not built. Call build() first.")

        candidates = self.search(query, client, top_k=top_k * 3)
        if not candidates:
            return []

        # Build query embedding
        query_vec = client.embed([query])[0]

        # Build doc embeddings dict from stored vectors
        doc_embeddings: dict[str, list[float]] = {}
        for doc in candidates:
            idx = self.doc_ids.index(doc["doc_id"])
            # Retrieve stored vector from the FAISS index
            vec = faiss.rev_swig_ptr(self._index.get_xb(), self._index.ntotal * self.dimension)
            vec_matrix = np.frombuffer(vec, dtype=np.float32).reshape(self._index.ntotal, self.dimension)
            doc_embeddings[doc["doc_id"]] = vec_matrix[idx].tolist()

        reranked = mmr_rerank(
            results=candidates,
            query_embedding=query_vec,
            doc_embeddings=doc_embeddings,
            top_k=top_k,
            mmr_lambda=lambda_param,
        )
        for doc in reranked:
            doc["source"] = "mmr"
        return reranked


# ── Hybrid retrieval ─────────────────────────────────────────────────────────


def hybrid_search(
    query: str,
    bm25_index: Any,
    embedding_index: "EmbeddingIndex",
    client: Any,
    top_k: int = 10,
    bm25_weight: float = 0.5,
    embedding_weight: float = 0.5,
) -> list[dict[str, Any]]:
    """
    Combine BM25 and embedding results via weighted reciprocal rank fusion.

    Args:
        query: Natural language query.
        bm25_index: BM25Index instance.
        embedding_index: EmbeddingIndex instance.
        client: CachedOpenAIClient for embedding the query.
        top_k: Number of results to return.
        bm25_weight: Weight for BM25 scores.
        embedding_weight: Weight for embedding scores.

    Returns:
        Deduplicated, re-ranked list of documents.
    """
    k = 60  # RRF constant

    bm25_results = bm25_index.search(query, top_k=top_k * 2)
    emb_results = embedding_index.search(query, client, top_k=top_k * 2)

    # Reciprocal rank fusion
    rrf_scores: dict[str, float] = {}
    doc_map: dict[str, dict] = {}

    for rank, doc in enumerate(bm25_results):
        doc_id = doc["doc_id"]
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + bm25_weight / (k + rank + 1)
        doc_map[doc_id] = doc

    for rank, doc in enumerate(emb_results):
        doc_id = doc["doc_id"]
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + embedding_weight / (k + rank + 1)
        if doc_id not in doc_map:
            doc_map[doc_id] = doc

    # Sort by fused score
    sorted_ids = sorted(rrf_scores, key=lambda x: rrf_scores[x], reverse=True)[:top_k]

    results = []
    for doc_id in sorted_ids:
        doc = doc_map[doc_id].copy()
        doc["score"] = rrf_scores[doc_id]
        doc["source"] = "hybrid"
        results.append(doc)

    return results


def mmr_rerank(
    results: list[dict[str, Any]],
    query_embedding: list[float],
    doc_embeddings: dict[str, list[float]],
    top_k: int = 10,
    mmr_lambda: float = 0.5,
) -> list[dict[str, Any]]:
    """
    Maximal Marginal Relevance re-ranking for diversity.

    Balances relevance (to query) with diversity (among selected docs).
    """
    if not results:
        return []

    q_vec = np.array(query_embedding, dtype=np.float32)
    q_vec = q_vec / (np.linalg.norm(q_vec) + 1e-10)

    selected: list[dict] = []
    remaining = list(results)

    while remaining and len(selected) < top_k:
        best_score = -float("inf")
        best_idx = 0

        for i, doc in enumerate(remaining):
            doc_vec = np.array(doc_embeddings.get(doc["doc_id"], [0.0] * len(query_embedding)), dtype=np.float32)
            doc_vec = doc_vec / (np.linalg.norm(doc_vec) + 1e-10)

            relevance = float(np.dot(q_vec, doc_vec))

            # Max similarity to already-selected
            max_sim = 0.0
            for sel in selected:
                sel_vec = np.array(doc_embeddings.get(sel["doc_id"], [0.0] * len(query_embedding)), dtype=np.float32)
                sel_vec = sel_vec / (np.linalg.norm(sel_vec) + 1e-10)
                sim = float(np.dot(doc_vec, sel_vec))
                max_sim = max(max_sim, sim)

            mmr_score = mmr_lambda * relevance - (1 - mmr_lambda) * max_sim
            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = i

        doc = remaining.pop(best_idx)
        doc["score"] = best_score
        doc["source"] = "mmr"
        selected.append(doc)

    return selected
