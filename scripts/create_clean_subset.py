#!/usr/bin/env python3
"""
Create the clean single-instance subset of HealthContradict.

Filters data_processed/healthcontradict.jsonl to exactly one example per topic
(the one with the lowest instance_id for that topic), giving a balanced set with
no topic over-represented. Writes data_processed/healthcontradict_clean184.jsonl.

Note: the dataset has 81 unique topics in the processed file; the filename keeps
the "clean184" suffix for compatibility with config files that reference it.

USAGE:
    python scripts/create_clean_subset.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> None:
    src = Path("data_processed/healthcontradict.jsonl")
    dst = Path("data_processed/healthcontradict_clean184.jsonl")

    if not src.exists():
        print(f"[ERROR] Source not found: {src}")
        print("  Run 'python scripts/prepare_dataset.py' first.")
        sys.exit(1)

    # Group by topic_id, keep minimum instance_id per topic
    by_topic: dict[int, dict] = {}  # topic_id → best record

    with open(src) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            meta = rec.get("metadata", {})
            topic_id = meta.get("topic_id")
            instance_id = meta.get("instance_id", 9999)
            if topic_id is None:
                continue
            if topic_id not in by_topic or instance_id < by_topic[topic_id]["metadata"]["instance_id"]:
                by_topic[topic_id] = rec

    kept = list(by_topic.values())
    topic_ids = set(by_topic.keys())

    with open(dst, "w") as f:
        for rec in kept:
            f.write(json.dumps(rec) + "\n")

    print(f"{len(kept)} examples written to {dst}")

    # Sanity check: each topic should appear exactly once
    duplicates = []
    seen: dict = {}
    for rec in kept:
        tid = rec.get("metadata", {}).get("topic_id")
        if tid in seen:
            duplicates.append(tid)
        seen[tid] = True

    if duplicates:
        print(f"[WARN] Duplicate topic_ids found: {duplicates[:5]} ...")
    else:
        print(f"Topic distribution OK: {len(topic_ids)} unique topics, each appears exactly once.")

    if len(kept) != 184:
        print(f"[INFO] {len(kept)} topics in processed dataset (original plan estimated 184).")
    else:
        print("Count check: 184 ✓")


if __name__ == "__main__":
    main()
