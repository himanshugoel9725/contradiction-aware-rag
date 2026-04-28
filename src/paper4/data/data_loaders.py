"""
Unified data schema and dataset-specific loaders.

WHY: HealthContradict, SciFact, and ManConCorpus each have different formats.
This module normalizes all of them into a single Pydantic-validated schema so
every downstream module (retrieval, pipeline, eval) can work with one format.

UNIFIED SCHEMA (Section 4.2 of the spec):
    Each example is a dict with:
        - example_id: str           Globally unique ID (dataset_prefix + original_id)
        - dataset: str              Source dataset name
        - claim: str                The biomedical claim or question
        - documents: list[Document] Retrieved/gold documents
        - gold_label: str | None    SUPPORT / CONTRADICT / NOT_ENOUGH_INFO / None
        - metadata: dict            Dataset-specific extra fields
"""

from __future__ import annotations

import csv
import json
import os
import tarfile
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


# ── Unified Schema ────────────────────────────────────────────────────────────


class Document(BaseModel):
    """A single document (abstract, passage, or evidence sentence)."""

    doc_id: str
    title: str = ""
    text: str
    label: str | None = None  # SUPPORT / CONTRADICT / NOT_ENOUGH_INFO
    metadata: dict[str, Any] = Field(default_factory=dict)


class Example(BaseModel):
    """
    One example in the unified dataset. Every dataset is normalized to this shape.
    """

    example_id: str
    dataset: str
    claim: str
    documents: list[Document] = Field(default_factory=list)
    gold_label: str | None = None  # Overall label if available
    metadata: dict[str, Any] = Field(default_factory=dict)


# ── SciFact Loader ────────────────────────────────────────────────────────────


def load_scifact(data_raw_dir: str | Path) -> list[Example]:
    """
    Load SciFact dataset from extracted tarball.

    Expected structure:
        data_raw_dir/scifact/claims_dev.jsonl
        data_raw_dir/scifact/corpus.jsonl

    SciFact format:
        claim: {id, claim, evidence: {doc_id: [{label, sentences}]}, cited_doc_ids}
        corpus doc: {doc_id, title, abstract: string[], structured}

    Key decisions:
        - abstract is a list of strings → joined with space
        - Only dev set used (test labels not public)
        - Gold label taken from evidence dict (SUPPORT or CONTRADICT)
        - Claims without evidence get gold_label=None
    """
    data_raw_dir = Path(data_raw_dir)
    claims_path = data_raw_dir / "scifact" / "claims_dev.jsonl"
    corpus_path = data_raw_dir / "scifact" / "corpus.jsonl"

    # Load corpus
    corpus: dict[str, dict] = {}
    with open(corpus_path, "r") as f:
        for line in f:
            doc = json.loads(line)
            doc_id = str(doc["doc_id"])
            # abstract is a list of sentences → join
            abstract_text = " ".join(doc.get("abstract", []))
            corpus[doc_id] = {
                "title": doc.get("title", ""),
                "text": abstract_text,
                "structured": doc.get("structured", False),
            }

    # Load claims and pair with evidence
    examples: list[Example] = []
    with open(claims_path, "r") as f:
        for line in f:
            claim_data = json.loads(line)
            claim_id = str(claim_data["id"])
            claim_text = claim_data["claim"]

            evidence_dict = claim_data.get("evidence", {})
            documents: list[Document] = []
            labels_seen: set[str] = set()

            for doc_id_str, evidence_list in evidence_dict.items():
                doc_id_str = str(doc_id_str)
                corpus_doc = corpus.get(doc_id_str, {})

                for ev in evidence_list:
                    label = ev.get("label", "NOT_ENOUGH_INFO")
                    labels_seen.add(label)

                    evidence_sentences = ev.get("sentences", [])
                    documents.append(
                        Document(
                            doc_id=f"scifact_{doc_id_str}",
                            title=corpus_doc.get("title", ""),
                            text=corpus_doc.get("text", ""),
                            label=label,
                            metadata={
                                "evidence_sentence_ids": evidence_sentences,
                                "structured": corpus_doc.get("structured", False),
                            },
                        )
                    )

            # Also add cited docs without explicit evidence
            for cited_id in claim_data.get("cited_doc_ids", []):
                cited_id_str = str(cited_id)
                if cited_id_str not in evidence_dict:
                    corpus_doc = corpus.get(cited_id_str, {})
                    if corpus_doc:
                        documents.append(
                            Document(
                                doc_id=f"scifact_{cited_id_str}",
                                title=corpus_doc.get("title", ""),
                                text=corpus_doc.get("text", ""),
                                label=None,
                            )
                        )

            # Determine overall gold label
            if "CONTRADICT" in labels_seen and "SUPPORT" in labels_seen:
                gold_label = "CONTRADICT"  # mixed evidence → contradiction
            elif "CONTRADICT" in labels_seen:
                gold_label = "CONTRADICT"
            elif "SUPPORT" in labels_seen:
                gold_label = "SUPPORT"
            else:
                gold_label = None

            examples.append(
                Example(
                    example_id=f"scifact_{claim_id}",
                    dataset="scifact",
                    claim=claim_text,
                    documents=documents,
                    gold_label=gold_label,
                    metadata={"cited_doc_ids": claim_data.get("cited_doc_ids", [])},
                )
            )

    return examples


# ── HealthContradict Loader ───────────────────────────────────────────────────


def load_healthcontradict(data_raw_dir: str | Path) -> list[Example]:
    """
    Load HealthContradict dataset.

    Expected structure (after git clone):
        data_raw_dir/healthcontradict/
            dataset/
                dataset_ready.jsonl     — pre-assembled dataset (if available via LFS)
            query/
                query-all.jsonl         — topics with (topic_id, description, query_stance)
            doc/
                doc-all-stance.csv      — all documents with stance labels
                doc-contradict.csv      — specifically contradicting document pairs

    Strategy:
        1. Try loading dataset_ready.jsonl first (easiest path)
        2. Fall back to assembling from query + doc files
    """
    data_raw_dir = Path(data_raw_dir)
    base = data_raw_dir / "healthcontradict"

    # Try the pre-assembled file first
    ready_path = base / "dataset" / "dataset_ready.jsonl"
    if ready_path.exists() and ready_path.stat().st_size > 100:
        return _load_healthcontradict_ready(ready_path)

    # Fall back to manual assembly
    return _load_healthcontradict_manual(base)


def _load_healthcontradict_ready(path: Path) -> list[Example]:
    """Load from the pre-assembled dataset_ready.jsonl."""
    examples: list[Example] = []
    with open(path, "r") as f:
        for i, line in enumerate(f):
            rec = json.loads(line)
            topic_id = str(rec.get("topic_id", rec.get("id", i)))
            instance_id = str(rec.get("instance_id", i))

            # Build documents from the record
            documents: list[Document] = []
            
            # Check if it's the newer multi-doc format (documents/docs array) or older pair format (doc_a/doc_b)
            if "documents" in rec or "docs" in rec:
                for j, doc in enumerate(rec.get("documents", rec.get("docs", []))):
                    documents.append(
                        Document(
                            doc_id=f"hc_{topic_id}_doc{j}",
                            title=doc.get("title", ""),
                            text=doc.get("text", doc.get("content", "")),
                            label=doc.get("stance", doc.get("label", None)),
                            metadata={k: v for k, v in doc.items() if k not in ("title", "text", "content", "stance", "label")},
                        )
                    )
            else:
                # Handle doc_a/doc_b pair format (the actual format of dataset_ready.jsonl)
                for j, doc_key in enumerate(["doc_a", "doc_b"]):
                    if doc_key in rec and rec[doc_key]:
                        # doc_b (j=1) is always the CONTRADICT side; each instance
                        # in dataset_ready.jsonl is a contradictory pair by design.
                        stance = "CONTRADICT" if j == 1 else "SUPPORT"
                        documents.append(
                            Document(
                                doc_id=f"hc_{topic_id}_doc{j}",
                                title="",
                                text=rec[doc_key],
                                label=stance,
                                metadata={"pair_type": rec.get("pair_type", "")},
                            )
                        )

            examples.append(
                Example(
                    example_id=f"hc_{topic_id}_{instance_id}",
                    dataset="healthcontradict",
                    claim=rec.get("query", rec.get("description", rec.get("claim", ""))),
                    documents=documents,
                    gold_label="CONTRADICT" if any(d.label == "CONTRADICT" for d in documents) else None,
                    metadata={k: v for k, v in rec.items() if k not in ("documents", "docs", "query", "description", "claim", "doc_a", "doc_b", "pair_type")},
                )
            )
    return examples


def _load_healthcontradict_manual(base: Path) -> list[Example]:
    """Assemble from query-all.jsonl + doc-all-stance.csv."""
    # Load queries
    queries: dict[str, dict] = {}
    query_path = base / "query" / "query-all.jsonl"
    if query_path.exists():
        with open(query_path, "r") as f:
            for line in f:
                q = json.loads(line)
                topic_id = str(q.get("topic_id", q.get("id", "")))
                queries[topic_id] = q

    # Load documents with stance labels
    docs_by_topic: dict[str, list[dict]] = {}

    doc_dir = base / "doc"
    for csv_name in ["doc-all-stance.csv", "doc-contradict.csv"]:
        csv_path = doc_dir / csv_name
        if not csv_path.exists():
            continue
        with open(csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                topic_id = str(row.get("topic_id", row.get("query_id", "")))
                if topic_id not in docs_by_topic:
                    docs_by_topic[topic_id] = []
                docs_by_topic[topic_id].append(row)

    # Combine
    examples: list[Example] = []
    all_topic_ids = set(queries.keys()) | set(docs_by_topic.keys())

    for topic_id in sorted(all_topic_ids):
        query_info = queries.get(topic_id, {})
        claim = query_info.get("description", query_info.get("query", f"Topic {topic_id}"))

        documents: list[Document] = []
        for j, doc_row in enumerate(docs_by_topic.get(topic_id, [])):
            stance = doc_row.get("stance", doc_row.get("label", None))
            if stance:
                stance = stance.upper().strip()

            documents.append(
                Document(
                    doc_id=f"hc_{topic_id}_doc{j}",
                    title=doc_row.get("title", ""),
                    text=doc_row.get("text", doc_row.get("content", doc_row.get("abstract", ""))),
                    label=stance,
                    metadata={k: v for k, v in doc_row.items() if k not in ("title", "text", "content", "abstract", "stance", "label")},
                )
            )

        has_contradict = any(d.label == "CONTRADICT" for d in documents)
        examples.append(
            Example(
                example_id=f"hc_{topic_id}",
                dataset="healthcontradict",
                claim=claim,
                documents=documents,
                gold_label="CONTRADICT" if has_contradict else None,
                metadata={"query_stance": query_info.get("query_stance", None)},
            )
        )

    return examples


# ── ManConCorpus Loader ───────────────────────────────────────────────────────

# Label mapping per spec: Excitatory → SUPPORT, Inhibitory → CONTRADICT, Neutral → NOT_ENOUGH_INFO
_MANCON_LABEL_MAP = {
    "excitatory": "SUPPORT",
    "inhibitory": "CONTRADICT",
    "neutral": "NOT_ENOUGH_INFO",
}


def load_manconcorpus(data_raw_dir: str | Path) -> list[Example]:
    """
    Load ManConCorpus dataset.

    Label mapping: Excitatory→SUPPORT, Inhibitory→CONTRADICT, Neutral→NOT_ENOUGH_INFO

    NOTE: ManConCorpus is a secondary dataset — loaded opportunistically.
    If files are not present, returns an empty list.
    """
    data_raw_dir = Path(data_raw_dir)
    base = data_raw_dir / "manconcorpus"

    if not base.exists():
        return []

    examples: list[Example] = []
    # Try common file patterns
    for jsonl_file in sorted(base.glob("*.jsonl")):
        with open(jsonl_file, "r") as f:
            for i, line in enumerate(f):
                rec = json.loads(line)
                raw_label = rec.get("label", rec.get("relation_type", "neutral")).lower().strip()
                label = _MANCON_LABEL_MAP.get(raw_label, "NOT_ENOUGH_INFO")

                examples.append(
                    Example(
                        example_id=f"mancon_{i}",
                        dataset="manconcorpus",
                        claim=rec.get("claim", rec.get("sentence1", rec.get("text", ""))),
                        documents=[
                            Document(
                                doc_id=f"mancon_{i}_doc0",
                                text=rec.get("evidence", rec.get("sentence2", rec.get("abstract", ""))),
                                label=label,
                            )
                        ],
                        gold_label=label,
                        metadata={k: v for k, v in rec.items()
                                  if k not in ("claim", "evidence", "label", "relation_type", "sentence1", "sentence2", "text", "abstract")},
                    )
                )
    return examples


# ── Normalization orchestrator ────────────────────────────────────────────────


def load_all_datasets(
    data_raw_dir: str | Path = "./data_raw",
) -> dict[str, list[Example]]:
    """
    Load all available datasets and return them keyed by name.

    Returns:
        {"scifact": [...], "healthcontradict": [...], "manconcorpus": [...]}
        Missing datasets return empty lists.
    """
    data_raw_dir = Path(data_raw_dir)
    results: dict[str, list[Example]] = {}

    for name, loader in [
        ("scifact", load_scifact),
        ("healthcontradict", load_healthcontradict),
        ("manconcorpus", load_manconcorpus),
    ]:
        try:
            examples = loader(data_raw_dir)
            results[name] = examples
            print(f"[Data] Loaded {name}: {len(examples)} examples")
        except FileNotFoundError as e:
            print(f"[Data] Skipping {name}: {e}")
            results[name] = []

    return results


def save_dataset_jsonl(examples: list[Example], output_path: str | Path) -> None:
    """Write examples to JSONL, one per line."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for ex in examples:
            f.write(ex.model_dump_json() + "\n")
    print(f"[Data] Wrote {len(examples)} examples to {output_path}")


def load_dataset_jsonl(input_path: str | Path) -> list[Example]:
    """Read examples from a normalized JSONL file."""
    examples: list[Example] = []
    with open(input_path, "r") as f:
        for line in f:
            if line.strip():
                examples.append(Example.model_validate_json(line))
    return examples
