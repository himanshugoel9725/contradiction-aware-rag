#!/usr/bin/env python3
"""
Re-apply post-generation checks to an existing per_example.jsonl run.

WHY: After fixing postcheck.py (tighter VCS keywords, better citation regex,
stance_label fallback), this script re-computes the postcheck fields in-place
so you get corrected metrics without re-running the expensive pipeline.

USAGE:
    python scripts/recheck.py --run-id experiment_2_3_generation_strategies_20260423_063021
    python scripts/recheck.py --run-id <id> [--runs-dir ./runs]

Writes a new per_example.jsonl (replaces the old one) with fresh postcheck
values, then prints a summary of before/after VCS/EAA counts so you can verify
the fix had the intended effect.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import src.paper4.utils.env_setup  # noqa: F401, E402

import typer  # noqa: E402

from src.paper4.pipeline.postcheck import run_postchecks  # noqa: E402

app = typer.Typer(help="Re-apply fixed postcheck to an existing run")


def _get_strategy(result: dict) -> str:
    """Best-effort: extract strategy from synthesis field, default to 'C'."""
    synth = result.get("synthesis", {})
    if isinstance(synth, dict):
        return synth.get("strategy", "C")
    return "C"


@app.command()
def recheck(
    run_id: str = typer.Option(..., "--run-id", "-r"),
    runs_dir: str = typer.Option("./runs", "--runs-dir"),
) -> None:
    run_dir = Path(runs_dir) / run_id
    per_example_path = run_dir / "per_example.jsonl"

    if not per_example_path.exists():
        print(f"[ERROR] Not found: {per_example_path}")
        raise typer.Exit(1)

    # Load
    results = []
    with open(per_example_path) as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))

    print(f"[Recheck] Loaded {len(results)} examples from {per_example_path}")

    # Before counts
    before_vcs = sum(
        1 for r in results
        if r.get("postcheck", {}).get("stance_coverage", {}).get("response_mentions_both", False)
    )
    before_eaa_cited = sum(
        1 for r in results
        if r.get("postcheck", {}).get("citations", {}).get("citation_count", 0) > 0
    )
    print(f"[Recheck] BEFORE — VCS=1 count: {before_vcs}/{len(results)}, examples with any citations: {before_eaa_cited}/{len(results)}")

    # Backup original
    backup_path = per_example_path.with_suffix(".jsonl.pre_recheck")
    if not backup_path.exists():
        shutil.copy2(per_example_path, backup_path)
        print(f"[Recheck] Backup saved to {backup_path.name}")
    else:
        print(f"[Recheck] Backup already exists — skipping backup")

    # Re-apply postcheck
    updated = []
    for r in results:
        if "error" in r:
            updated.append(r)
            continue

        synth = r.get("synthesis", {})
        response_text = synth.get("text", "") if isinstance(synth, dict) else ""
        docs = r.get("selected_documents", [])
        strategy = _get_strategy(r)

        if response_text:
            r["postcheck"] = run_postchecks(response_text, docs, strategy)

        updated.append(r)

    # Write
    with open(per_example_path, "w") as f:
        for r in updated:
            f.write(json.dumps(r) + "\n")

    # After counts
    after_vcs = sum(
        1 for r in updated
        if r.get("postcheck", {}).get("stance_coverage", {}).get("response_mentions_both", False)
    )
    after_eaa_cited = sum(
        1 for r in updated
        if r.get("postcheck", {}).get("citations", {}).get("citation_count", 0) > 0
    )
    print(f"[Recheck] AFTER  — VCS=1 count: {after_vcs}/{len(updated)}, examples with any citations: {after_eaa_cited}/{len(updated)}")
    print(f"[Recheck] Done. Run evaluate.py to recompute metrics.json.")


if __name__ == "__main__":
    app()
