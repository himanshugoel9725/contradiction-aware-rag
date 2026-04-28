"""
Test data loaders — schema validation, label mapping, edge cases.
"""

import json
import tempfile
from pathlib import Path

import pytest
from src.paper4.data.data_loaders import (
    Example,
    Document,
    load_scifact,
    load_healthcontradict,
    load_manconcorpus,
    save_dataset_jsonl,
    load_dataset_jsonl,
)


def test_example_schema():
    """Example model validates correct data."""
    ex = Example(
        example_id="test_1",
        dataset="test",
        claim="Aspirin reduces heart attacks.",
        documents=[
            Document(doc_id="d1", text="Study shows aspirin helps.", label="SUPPORT"),
        ],
        gold_label="SUPPORT",
    )
    assert ex.example_id == "test_1"
    assert len(ex.documents) == 1
    assert ex.documents[0].label == "SUPPORT"


def test_example_roundtrip_jsonl():
    """Write and read examples via JSONL."""
    examples = [
        Example(
            example_id=f"ex_{i}",
            dataset="test",
            claim=f"Claim {i}",
            gold_label="SUPPORT" if i % 2 == 0 else "CONTRADICT",
        )
        for i in range(5)
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.jsonl"
        save_dataset_jsonl(examples, path)

        loaded = load_dataset_jsonl(path)
        assert len(loaded) == 5
        assert loaded[0].example_id == "ex_0"
        assert loaded[2].gold_label == "SUPPORT"
        assert loaded[3].gold_label == "CONTRADICT"


def test_manconcorpus_label_mapping():
    """ManConCorpus labels map correctly: Excitatory→SUPPORT, etc."""
    from src.paper4.data.data_loaders import _MANCON_LABEL_MAP

    assert _MANCON_LABEL_MAP["excitatory"] == "SUPPORT"
    assert _MANCON_LABEL_MAP["inhibitory"] == "CONTRADICT"
    assert _MANCON_LABEL_MAP["neutral"] == "NOT_ENOUGH_INFO"


def test_scifact_loader_with_fixture():
    """Test SciFact loader with minimal fixture data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        scifact_dir = Path(tmpdir) / "scifact"
        scifact_dir.mkdir()

        # Create minimal corpus
        corpus = [
            {"doc_id": 1, "title": "Test Paper", "abstract": ["Sentence one.", "Sentence two."], "structured": False},
        ]
        with open(scifact_dir / "corpus.jsonl", "w") as f:
            for doc in corpus:
                f.write(json.dumps(doc) + "\n")

        # Create minimal claims
        claims = [
            {
                "id": 100,
                "claim": "Test claim about medicine.",
                "evidence": {"1": [{"label": "SUPPORT", "sentences": [0]}]},
                "cited_doc_ids": [1],
            },
        ]
        with open(scifact_dir / "claims_dev.jsonl", "w") as f:
            for claim in claims:
                f.write(json.dumps(claim) + "\n")

        examples = load_scifact(tmpdir)
        assert len(examples) == 1
        assert examples[0].example_id == "scifact_100"
        assert examples[0].gold_label == "SUPPORT"
        assert len(examples[0].documents) == 1
        assert "Sentence one. Sentence two." in examples[0].documents[0].text


def test_empty_manconcorpus():
    """ManConCorpus returns empty list when directory doesn't exist."""
    with tempfile.TemporaryDirectory() as tmpdir:
        result = load_manconcorpus(tmpdir)
        assert result == []
