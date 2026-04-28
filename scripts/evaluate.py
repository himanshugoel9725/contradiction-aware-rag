#!/usr/bin/env python3
"""
Evaluate a completed pipeline run — compute all metrics with bootstrap CIs.

USAGE:
    python scripts/evaluate.py --run-id smoke_test_20250101_120000
    python scripts/evaluate.py --run-dir runs/gap1_strategy_A
    python scripts/evaluate.py --run-dir runs/gap1_strategy_A --run-judge
    python scripts/evaluate.py --run-dir runs/gap1_strategy_A --no-judge

Reads per_example.jsonl from the run directory, computes CBR/VCS/EAA/EQU, and
optionally runs the LLM judge for CAS + semantic EAA.

CHECKPOINTING: Judge results are written to per_example_judged.jsonl after each
example so the run can be interrupted and resumed safely. All LLM calls are
cached in SQLite so resuming is free.
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import src.paper4.utils.env_setup  # noqa: F401, E402

import typer  # noqa: E402
import numpy as np  # noqa: E402
from tqdm import tqdm  # noqa: E402

from src.paper4.eval.metrics import compute_all_metrics, compute_cbr  # noqa: E402
from src.paper4.eval.bootstrap import bootstrap_ci, bootstrap_metric  # noqa: E402
from src.paper4.llm.anthropic_client import CachedAnthropicClient, BudgetExhaustedError  # noqa: E402
from src.paper4.eval.judge import LLMJudge  # noqa: E402

app = typer.Typer(help="Evaluate a pipeline run")


@app.command()
def evaluate(
    run_id: Optional[str] = typer.Option(None, "--run-id", "-r", help="Run identifier (resolves to ./runs/<run_id>)"),
    run_dir: Optional[str] = typer.Option(None, "--run-dir", help="Direct path to run directory"),
    bootstrap: int = typer.Option(1000, "--bootstrap", "-b", help="Number of bootstrap resamples"),
    runs_dir: str = typer.Option("./runs", "--runs-dir", help="Base directory for runs (used with --run-id)"),
    run_judge: bool = typer.Option(False, "--run-judge", help="Run LLM judge for CAS + semantic EAA"),
    no_judge: bool = typer.Option(False, "--no-judge", help="Skip LLM judge (fast evaluation only)"),
    max_budget: float = typer.Option(0.0, "--max-budget", help="Max API spend for judge in USD (0 = unlimited)"),
) -> None:
    """Compute all metrics for a completed run."""
    # Resolve run directory
    if run_dir:
        resolved_dir = Path(run_dir)
        resolved_run_id = resolved_dir.name
    elif run_id:
        resolved_dir = Path(runs_dir) / run_id
        resolved_run_id = run_id
    else:
        print("[ERROR] Provide either --run-id or --run-dir")
        raise typer.Exit(1)

    per_example_path = resolved_dir / "per_example.jsonl"
    if not per_example_path.exists():
        print(f"[ERROR] Not found: {per_example_path}")
        raise typer.Exit(1)

    # Load results
    results = []
    with open(per_example_path, "r") as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))

    print(f"[Eval] Loaded {len(results)} examples from {per_example_path}")

    valid_results = [r for r in results if "error" not in r]
    error_count = len(results) - len(valid_results)
    if error_count > 0:
        print(f"[Eval] Skipping {error_count} errored examples")

    # ── Compute base metrics ──────────────────────────────────────
    metrics = compute_all_metrics(valid_results)
    print(f"\n[Eval] Raw metrics:")
    for metric_name, metric_value in metrics.items():
        if isinstance(metric_value, dict):
            main_key = next((k for k in metric_value if k in (metric_name, "point_estimate")), None)
            if main_key:
                print(f"  {metric_name}: {metric_value[main_key]:.4f}")

    # Bootstrap CIs
    print(f"\n[Eval] Computing bootstrap CIs ({bootstrap} resamples)...")

    cbr_fn = lambda res: compute_cbr(res).get("cbr", 0.0)
    cbr_ci = bootstrap_metric(valid_results, cbr_fn, n_resamples=bootstrap)

    vcs_scores = []
    eaa_scores = []
    for r in valid_results:
        pc = r.get("postcheck", {})
        sc = pc.get("stance_coverage", {})
        vcs_scores.append(1.0 if sc.get("response_mentions_both", False) else 0.0)
        cit = pc.get("citations", {})
        total = cit.get("citation_count", 0)
        valid = len(cit.get("valid_citations", []))
        eaa_scores.append(valid / total if total > 0 else 0.0)

    vcs_ci = bootstrap_ci(vcs_scores, n_resamples=bootstrap)
    eaa_ci = bootstrap_ci(eaa_scores, n_resamples=bootstrap)

    metrics_with_ci = {
        "cbr": {**metrics["cbr"], "bootstrap_ci": cbr_ci},
        "vcs": {**metrics["vcs"], "bootstrap_ci": vcs_ci},
        "eaa": {**metrics["eaa"], "bootstrap_ci": eaa_ci},
        "equ": metrics.get("equ", {}),
        "n_examples": len(valid_results),
        "n_errors": error_count,
    }

    # ── LLM Judge: CAS + Semantic EAA ─────────────────────────────
    should_run_judge = run_judge and not no_judge
    if should_run_judge:
        print(f"\n[Eval] Running LLM judge (CAS + Semantic EAA)...")

        budget_arg = max_budget if max_budget > 0 else None
        client = CachedAnthropicClient(cache_dir="./cache", max_budget_usd=budget_arg)
        judge = LLMJudge(client=client)

        judged_path = resolved_dir / "per_example_judged.jsonl"

        # Load already-judged records (resume support)
        already_judged: dict[str, dict] = {}
        if judged_path.exists():
            with open(judged_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        rec = json.loads(line)
                        already_judged[rec.get("example_id", "")] = rec
            if already_judged:
                print(f"[Eval] Resuming — {len(already_judged)} already judged, "
                      f"{len(valid_results) - len(already_judged)} remaining")

        cas_scores: list[float] = []
        semantic_eaa_scores: list[float] = []

        # Collect scores from already-judged records
        for ex_id, rec in already_judged.items():
            cas = rec.get("cas_score", {})
            if isinstance(cas, dict):
                cas_scores.append(cas.get("normalized_score", 0.0))
            seaa = rec.get("semantic_eaa", {})
            if isinstance(seaa, dict):
                semantic_eaa_scores.append(seaa.get("semantic_eaa", 0.0))

        # Open judged file in append mode for incremental writing
        judge_file = open(judged_path, "a")

        try:
            pending = [r for r in valid_results if r.get("example_id") not in already_judged]
            bar = tqdm(pending, desc="judging", unit="ex", dynamic_ncols=True)

            for record in bar:
                ex_id = record.get("example_id", "")
                claim = record.get("claim", "")
                synthesis_text = record.get("synthesis", {}).get("text", "")
                docs = record.get("selected_documents", record.get("retrieved_documents", []))

                try:
                    cas_result = judge.evaluate_cas(
                        claim=claim,
                        documents=docs,
                        response_text=synthesis_text,
                    )
                    seaa_result = judge.evaluate_eaa_semantic(
                        claim=claim,
                        synthesis=synthesis_text,
                        documents=docs,
                    )
                except BudgetExhaustedError as e:
                    bar.write(f"\n{e}")
                    bar.write(
                        f"[Eval] Saved {len(already_judged) + len(cas_scores)} judged examples. "
                        "Rerun with --run-judge after adding credits."
                    )
                    break

                cas_scores.append(cas_result.get("normalized_score", 0.0))
                semantic_eaa_scores.append(seaa_result.get("semantic_eaa", 0.0))

                judged_record = {
                    **record,
                    "cas_score": cas_result,
                    "semantic_eaa": seaa_result,
                }
                judge_file.write(json.dumps(judged_record, default=str) + "\n")
                judge_file.flush()

                _s = client.cost
                bar.set_postfix(cost=f"${_s.cost_usd:.3f}", cache=f"{_s.cache_hits}/{_s.total_calls}")

        finally:
            judge_file.close()

        if cas_scores:
            cas_ci = bootstrap_ci(cas_scores, n_resamples=bootstrap)
            metrics_with_ci["cas"] = {
                "mean": statistics.mean(cas_scores),
                "median": statistics.median(cas_scores),
                "std": statistics.stdev(cas_scores) if len(cas_scores) > 1 else 0.0,
                "bootstrap_ci": cas_ci,
                "n": len(cas_scores),
            }
            print(f"\n  CAS: {cas_ci['point_estimate']:.4f} [{cas_ci['ci_lower']:.4f}, {cas_ci['ci_upper']:.4f}]")

        if semantic_eaa_scores:
            seaa_ci = bootstrap_ci(semantic_eaa_scores, n_resamples=bootstrap)
            metrics_with_ci["semantic_eaa"] = {
                "mean": statistics.mean(semantic_eaa_scores),
                "median": statistics.median(semantic_eaa_scores),
                "std": statistics.stdev(semantic_eaa_scores) if len(semantic_eaa_scores) > 1 else 0.0,
                "bootstrap_ci": seaa_ci,
                "n": len(semantic_eaa_scores),
            }
            print(f"  Semantic EAA: {seaa_ci['point_estimate']:.4f} [{seaa_ci['ci_lower']:.4f}, {seaa_ci['ci_upper']:.4f}]")

        client.print_cost_summary()

    # ── Write metrics.json ─────────────────────────────────────────
    output_path = resolved_dir / "metrics.json"
    with open(output_path, "w") as f:
        json.dump(metrics_with_ci, f, indent=2, default=str)

    print(f"\n[Eval] Results written to {output_path}")
    print(f"\n{'='*50}")
    print(f"  CBR: {cbr_ci['point_estimate']:.4f} [{cbr_ci['ci_lower']:.4f}, {cbr_ci['ci_upper']:.4f}]")
    print(f"  VCS: {vcs_ci['point_estimate']:.4f} [{vcs_ci['ci_lower']:.4f}, {vcs_ci['ci_upper']:.4f}]")
    print(f"  EAA: {eaa_ci['point_estimate']:.4f} [{eaa_ci['ci_lower']:.4f}, {eaa_ci['ci_upper']:.4f}]")
    if "cas" in metrics_with_ci:
        cas_pt = metrics_with_ci["cas"]["bootstrap_ci"]["point_estimate"]
        print(f"  CAS: {cas_pt:.4f}")
    if "semantic_eaa" in metrics_with_ci:
        seaa_pt = metrics_with_ci["semantic_eaa"]["bootstrap_ci"]["point_estimate"]
        print(f"  Semantic EAA: {seaa_pt:.4f}")
    print(f"{'='*50}")


if __name__ == "__main__":
    app()

