#!/usr/bin/env python3
"""
Prepare SciFact data for the contradiction-aware RAG pipeline.

Converts SciFact claims_dev.jsonl + corpus.jsonl into two files:
    data_processed/scifact_corpus.jsonl     — flat list of documents
    data_processed/scifact_contradiction_pairs.jsonl — unified schema records

UNIFIED SCHEMA (matches HealthContradict format):
    example_id: str       "sf_<claim_id>"
    dataset: str          "scifact"
    claim: str            The claim text
    gold_label: str       "CONTRADICT" | "SUPPORT" | "NOT_ENOUGH_INFO"
    documents: list[dict] [{doc_id, text, label, title, gold_stance_for_doc}]
    metadata: dict        {claim_id, cited_doc_ids, …}

Only claims that have at least one CONTRADICT evidence entry are included
(the dataset is not filtered to contradiction-only, but all annotated pairs
are kept including SUPPORT ones, so the pipeline can be evaluated end-to-end).

USAGE:
    python scripts/prepare_scifact.py
    python scripts/prepare_scifact.py --claims-path data_raw/scifact/claims_train.jsonl
    python scripts/prepare_scifact.py --max-claims 100
"""

from __future__ import annotations

import json
from pathlib import Path

import typer

app = typer.Typer(help="Prepare SciFact dataset for contradiction RAG pipeline")

_LABEL_MAP = {
    "CONTRADICT": "CONTRADICT",
    "SUPPORT": "SUPPORT",
    "NOT_ENOUGH_INFO": "NOT_ENOUGH_INFO",
}


def _load_corpus(corpus_path: str) -> dict[int, dict]:
    """Load corpus.jsonl into {doc_id_int: {doc_id, title, abstract}} dict."""
    corpus: dict[int, dict] = {}
    with open(corpus_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            doc_id = int(rec["doc_id"])
            abstract_raw = rec.get("abstract", "")
            abstract_str = " ".join(abstract_raw) if isinstance(abstract_raw, list) else abstract_raw
            corpus[doc_id] = {
                "doc_id": str(doc_id),
                "title": rec.get("title", ""),
                "abstract": abstract_str,
                "text": (rec.get("title", "") + " " + abstract_str).strip(),
            }
    return corpus


def _infer_claim_label(evidence: dict) -> str:
    """
    Infer overall claim label from evidence dict.
    SciFact evidence structure: {doc_id_str: {"label": ..., "sentences": [...]}}
    Priority: CONTRADICT > SUPPORT > NOT_ENOUGH_INFO
    """
    labels = set()
    for doc_evidence in evidence.values():
        # doc_evidence is a list of sentence-level dicts: [{"sentences": [...], "label": "..."}, ...]
        entries = doc_evidence if isinstance(doc_evidence, list) else [doc_evidence]
        for entry in entries:
            lbl = entry.get("label", "NOT_ENOUGH_INFO")
            labels.add(lbl)

    if "CONTRADICT" in labels:
        return "CONTRADICT"
    if "SUPPORT" in labels:
        return "SUPPORT"
    return "NOT_ENOUGH_INFO"


@app.command()
def prepare(
    claims_path: str = typer.Option(
        "data_raw/scifact/claims_dev.jsonl",
        "--claims-path",
        help="Path to SciFact claims JSONL (dev or train)",
    ),
    corpus_path: str = typer.Option(
        "data_raw/scifact/corpus.jsonl",
        "--corpus-path",
        help="Path to SciFact corpus JSONL",
    ),
    output_dir: str = typer.Option(
        "data_processed",
        "--output-dir",
        help="Directory to write output files",
    ),
    max_claims: int = typer.Option(
        0,
        "--max-claims",
        help="Limit number of claims (0 = all)",
    ),
    contradictions_only: bool = typer.Option(
        False,
        "--contradictions-only",
        help="Only output claims with at least one CONTRADICT evidence entry",
    ),
) -> None:
    """Convert SciFact claims + corpus into unified pipeline format."""
    if not Path(claims_path).exists():
        print(f"[ERROR] Claims not found: {claims_path}")
        raise typer.Exit(1)
    if not Path(corpus_path).exists():
        print(f"[ERROR] Corpus not found: {corpus_path}")
        raise typer.Exit(1)

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print(f"[SciFact] Loading corpus from {corpus_path}...")
    corpus = _load_corpus(corpus_path)
    print(f"[SciFact] Corpus: {len(corpus)} documents")

    # Write flat corpus file
    corpus_out_path = Path(output_dir) / "scifact_corpus.jsonl"
    with open(corpus_out_path, "w") as f:
        for doc in corpus.values():
            f.write(json.dumps(doc) + "\n")
    print(f"[SciFact] Written corpus: {corpus_out_path}")

    print(f"[SciFact] Loading claims from {claims_path}...")
    claims = []
    with open(claims_path) as f:
        for line in f:
            line = line.strip()
            if line:
                claims.append(json.loads(line))

    if max_claims > 0:
        claims = claims[:max_claims]
    print(f"[SciFact] {len(claims)} claims to process")

    # Build unified schema records
    records = []
    n_skipped = 0
    n_no_corpus = 0

    for raw_claim in claims:
        claim_id = raw_claim.get("id", raw_claim.get("claim_id", ""))
        claim_text = raw_claim.get("claim", "")
        evidence: dict = raw_claim.get("evidence", {})
        cited_doc_ids: list = raw_claim.get("cited_doc_ids", [])

        # SciFact format: evidence is {doc_id_str: {label, sentences}}
        # For claims without evidence (e.g., test set), use cited_doc_ids without labels
        if not evidence and not cited_doc_ids:
            n_skipped += 1
            continue

        gold_label = _infer_claim_label(evidence) if evidence else "NOT_ENOUGH_INFO"

        if contradictions_only and gold_label != "CONTRADICT":
            n_skipped += 1
            continue

        # Build document list from evidence (prefer evidence keys, fall back to cited_doc_ids)
        evidence_doc_ids = [int(k) for k in evidence.keys()] if evidence else []
        all_cited_ids = list(dict.fromkeys(evidence_doc_ids + [int(d) for d in cited_doc_ids]))

        documents = []
        for doc_id_int in all_cited_ids:
            if doc_id_int not in corpus:
                n_no_corpus += 1
                continue

            doc = corpus[doc_id_int].copy()
            doc_evidence = evidence.get(str(doc_id_int), {})
            # doc_evidence may be a list of {sentences, label} entries (SciFact format)
            if isinstance(doc_evidence, list):
                entry_labels = [e.get("label", "NOT_ENOUGH_INFO") for e in doc_evidence]
                if "CONTRADICT" in entry_labels:
                    doc_label = "CONTRADICT"
                elif "SUPPORT" in entry_labels:
                    doc_label = "SUPPORT"
                else:
                    doc_label = "NOT_ENOUGH_INFO"
                evidence_sentences = sorted(set(s for e in doc_evidence for s in e.get("sentences", [])))
            else:
                doc_label = doc_evidence.get("label", "NOT_ENOUGH_INFO")
                evidence_sentences = doc_evidence.get("sentences", [])

            doc["label"] = _LABEL_MAP.get(doc_label, "NOT_ENOUGH_INFO")
            doc["gold_stance_for_doc"] = doc["label"]

            # Include highlighted evidence sentences as a hint in metadata
            if evidence_sentences:
                abstract_sents = corpus[doc_id_int].get("abstract", "").split(". ")
                selected_sents = [
                    abstract_sents[i] for i in evidence_sentences
                    if 0 <= i < len(abstract_sents)
                ]
                doc["evidence_sentences"] = selected_sents

            documents.append(doc)

        if not documents:
            n_skipped += 1
            continue

        record = {
            "example_id": f"sf_{claim_id}",
            "dataset": "scifact",
            "claim": claim_text,
            "gold_label": gold_label,
            "documents": documents,
            "metadata": {
                "claim_id": claim_id,
                "cited_doc_ids": [str(d) for d in all_cited_ids],
                "has_contradiction": gold_label == "CONTRADICT",
            },
        }
        records.append(record)

    print(f"[SciFact] {len(records)} records built  (skipped={n_skipped}, no_corpus={n_no_corpus})")

    # Write contradiction pairs
    out_path = Path(output_dir) / "scifact_contradiction_pairs.jsonl"
    with open(out_path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    print(f"[SciFact] Written: {out_path}")

    # Stats
    label_counts: dict[str, int] = {}
    for rec in records:
        label_counts[rec["gold_label"]] = label_counts.get(rec["gold_label"], 0) + 1
    for lbl, count in sorted(label_counts.items()):
        print(f"  {lbl}: {count}")


if __name__ == "__main__":
    app()
