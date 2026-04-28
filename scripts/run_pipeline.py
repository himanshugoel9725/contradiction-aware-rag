#!/usr/bin/env python3
"""
Main pipeline orchestrator — runs the full contradiction-aware RAG pipeline.

USAGE:
    python scripts/run_pipeline.py --config configs/smoke.yaml
    python scripts/run_pipeline.py --config configs/smoke.yaml --resume
    python scripts/run_pipeline.py --config configs/smoke.yaml --run-id my_run

    # Strategy comparison (Gap 1)
    python scripts/run_pipeline.py --config configs/experiment_gap1_stratA.yaml \\
        --run-id gap1_strategy_A \\
        --dataset-path data_processed/healthcontradict_clean184.jsonl \\
        --strategy A

    # Ablation (Gap 4)
    python scripts/run_pipeline.py --config configs/experiment_gap4_ablation.yaml \\
        --run-id gap4_minus_stance --strategy C --skip-stance

    # Backbone comparison (Gap 5)
    python scripts/run_pipeline.py --config configs/experiment_gap5_backbone.yaml \\
        --run-id gap5_claude_haiku --model-override claude-3-5-haiku-20241022

    # Retrieval comparison (Gap 7)
    python scripts/run_pipeline.py --config configs/experiment_gap7_retrieval.yaml \\
        --run-id gap7_retrieval_hybrid --use-retrieval --retrieval-strategy hybrid

PIPELINE STAGES (per example):
    1. PICO extraction          (--skip-pico to disable)
    2. Document loading         (gold or retrieved based on --use-retrieval)
    3. Stance classification    (--skip-stance to disable)
    4. Contradiction-aware selection  (--skip-selection to disable)
    5. Evidence quality tagging (--skip-quality to disable)
    6. Structured synthesis generation (--strategy A/B/C/D)
    7. Post-check verification

RESUME: Uses CheckpointManager — interrupt with Ctrl+C anytime, rerun with
--resume to continue where you left off. Zero repeated API calls.

BUDGET: Use --max-budget to cap spending. The run saves progress and exits
cleanly when the limit is reached. Rerun with --resume after adding credits.
"""

from __future__ import annotations

import json
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# Must be imported first — sets offline env vars
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import src.paper4.utils.env_setup  # noqa: F401, E402

import yaml  # noqa: E402
import typer  # noqa: E402
from tqdm import tqdm  # noqa: E402

from src.paper4.utils.checkpoint import CheckpointManager  # noqa: E402
from src.paper4.llm.anthropic_client import CachedAnthropicClient, BudgetExhaustedError  # noqa: E402
from src.paper4.data.data_loaders import load_dataset_jsonl, Example  # noqa: E402
from src.paper4.pipeline.pico import PICOExtractor  # noqa: E402
from src.paper4.pipeline.stance import StanceClassifier  # noqa: E402
from src.paper4.pipeline.selection import select_evidence, compute_stance_coverage  # noqa: E402
from src.paper4.pipeline.synthesis import SynthesisGenerator  # noqa: E402
from src.paper4.pipeline.postcheck import run_postchecks  # noqa: E402
from src.paper4.utils.grade import tag_evidence_quality  # noqa: E402

app = typer.Typer(help="Contradiction-Aware Biomedical RAG Pipeline")


def _load_config(config_path: str) -> dict:
    """Load and validate a YAML experiment config."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def _setup_run_dir(run_id: str) -> Path:
    """Create the output directory for this run."""
    run_dir = Path("runs") / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _derive_ablation_condition(
    skip_pico: bool,
    skip_stance: bool,
    skip_quality: bool,
    skip_selection: bool,
) -> str:
    """Return a human-readable ablation condition name from the skip flags."""
    if not any([skip_pico, skip_stance, skip_quality, skip_selection]):
        return "full"
    parts = []
    if skip_pico:
        parts.append("minus_pico")
    if skip_stance:
        parts.append("minus_stance")
    if skip_quality:
        parts.append("minus_quality")
    if skip_selection:
        parts.append("minus_selection")
    return "_".join(parts)


def _build_corpus_dict(jsonl_path: str) -> dict[str, dict]:
    """Load all documents from a dataset JSONL into a doc_id → document dict.
    Handles two formats:
      - Paper4 format: each line has a "documents" list (claims+evidence)
      - Flat corpus format: each line is a single document with a "doc_id" key
    """
    corpus: dict[str, dict] = {}
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if "documents" in rec:
                for doc in rec["documents"]:
                    corpus[doc["doc_id"]] = doc
            elif "doc_id" in rec:
                corpus[rec["doc_id"]] = rec
    return corpus


def _process_one_example(
    example: Example,
    config: dict,
    client: CachedAnthropicClient,
    pico_extractor: PICOExtractor | None,
    stance_classifier: StanceClassifier | None,
    synthesis_generator: SynthesisGenerator,
    *,
    strategy: str = "C",
    skip_pico: bool = False,
    skip_stance: bool = False,
    skip_quality: bool = False,
    skip_selection: bool = False,
    use_retrieval: bool = False,
    retrieval_strategy: str = "hybrid",
    model_override: str | None = None,
    ablation_condition: str = "full",
    bm25_index: object | None = None,
    embedding_index: object | None = None,
    corpus: dict[str, dict] | None = None,
    all_doc_ids: list[str] | None = None,
) -> dict:
    """
    Run the full pipeline on one example.

    Returns a result dict with all intermediate outputs for analysis.
    """
    result: dict = {
        "example_id": example.example_id,
        "dataset": example.dataset,
        "claim": example.claim,
        "gold_label": example.gold_label,
        "synthesis_strategy": strategy,
        "ablation_condition": ablation_condition,
        "synthesis_model": model_override or "default",
    }

    pipeline_cfg = config.get("pipeline", {})
    gen_cfg = config.get("generation", {})
    retrieval_cfg = config.get("retrieval", {})

    # ── 1. PICO Extraction ──────────────────────────────────────────
    pico = None
    if not skip_pico and pipeline_cfg.get("pico_extraction") and pico_extractor:
        pico = pico_extractor.extract(example.claim)
        result["pico"] = pico

    # ── 2. Get documents ────────────────────────────────────────────
    if use_retrieval and corpus is not None:
        top_k = retrieval_cfg.get("top_k", 10)
        retrieved_doc_ids: list[str] = []

        if retrieval_strategy == "bm25" and bm25_index is not None:
            retrieved = bm25_index.search(example.claim, top_k=top_k)
            retrieved_doc_ids = [d["doc_id"] for d in retrieved]
        elif retrieval_strategy == "embedding" and embedding_index is not None:
            retrieved = embedding_index.search(example.claim, client, top_k=top_k)
            retrieved_doc_ids = [d["doc_id"] for d in retrieved]
        elif retrieval_strategy == "hybrid" and bm25_index is not None and embedding_index is not None:
            from src.paper4.retrieval.embedding_index import hybrid_search
            retrieved = hybrid_search(
                example.claim, bm25_index, embedding_index, client, top_k=top_k,
                bm25_weight=retrieval_cfg.get("bm25_weight", 0.5),
                embedding_weight=retrieval_cfg.get("embedding_weight", 0.5),
            )
            retrieved_doc_ids = [d["doc_id"] for d in retrieved]
        elif retrieval_strategy == "mmr" and embedding_index is not None:
            retrieved = embedding_index.search_mmr(example.claim, client, top_k=top_k)
            retrieved_doc_ids = [d["doc_id"] for d in retrieved]
        elif retrieval_strategy == "random" and all_doc_ids:
            retrieved_doc_ids = random.sample(all_doc_ids, min(top_k, len(all_doc_ids)))
        else:
            retrieved_doc_ids = [d["doc_id"] for d in example.documents]

        documents = [corpus[did] for did in retrieved_doc_ids if did in corpus]
        result["retrieval_strategy"] = retrieval_strategy
        result["retrieved_doc_ids"] = retrieved_doc_ids
    else:
        documents = [d.model_dump() for d in example.documents]
        result["retrieved_documents"] = documents

    # ── 3. Stance Classification ────────────────────────────────────
    stances = []
    if not skip_stance and pipeline_cfg.get("stance_classification") and stance_classifier:
        stances = stance_classifier.classify_batch(example.claim, documents)
        result["stances"] = stances

        stance_map = {s["doc_id"]: s for s in stances}
        for doc in documents:
            s = stance_map.get(doc["doc_id"], {})
            doc["stance_label"] = s.get("label", "NOT_ENOUGH_INFO")
            doc["stance_confidence"] = s.get("confidence", 0.0)

    # ── 4. Contradiction-Aware Selection ────────────────────────────
    if not skip_selection and pipeline_cfg.get("contradiction_aware_selection") and stances:
        selected = select_evidence(
            documents,
            stances,
            top_k=retrieval_cfg.get("top_k", 10),
        )
        coverage = compute_stance_coverage(selected)
        result["selected_documents"] = selected
        result["stance_coverage"] = coverage
    else:
        result["selected_documents"] = documents

    # ── 5. Evidence Quality Tagging ─────────────────────────────────
    if not skip_quality and pipeline_cfg.get("evidence_quality_tagging", True):
        result["selected_documents"] = tag_evidence_quality(result["selected_documents"])

    # ── 6. Synthesis Generation ─────────────────────────────────────
    synthesis = synthesis_generator.generate(
        claim=example.claim,
        documents=result["selected_documents"],
        strategy=strategy,
        pico=pico,
        model_task_type=gen_cfg.get("model_task_type", "generation"),
        model_override=model_override,
    )
    result["synthesis"] = synthesis

    # ── 7. Post-check Verification ──────────────────────────────────
    if pipeline_cfg.get("postcheck_verification", True):
        postchecks = run_postchecks(
            response_text=synthesis["text"],
            documents=result["selected_documents"],
            strategy=strategy,
        )
        result["postcheck"] = postchecks

    return result


@app.command()
def run(
    config: str = typer.Option(..., "--config", "-c", help="Path to YAML config file"),
    run_id: Optional[str] = typer.Option(None, "--run-id", "-r", help="Run identifier (default: config name + timestamp)"),
    resume: bool = typer.Option(False, "--resume", help="Resume from checkpoint"),
    max_minutes: int = typer.Option(0, "--max-minutes", help="Time limit in minutes (0 = unlimited)"),
    max_examples: int = typer.Option(0, "--max-examples", help="Override config max_examples (0 = use config)"),
    # ── New flags ────────────────────────────────────────────────────
    strategy: str = typer.Option("C", "--strategy", "-s", help="Synthesis strategy: A, B, C, or D"),
    dataset_path: Optional[str] = typer.Option(None, "--dataset-path", help="Override config dataset path"),
    skip_pico: bool = typer.Option(False, "--skip-pico", help="Skip PICO extraction"),
    skip_stance: bool = typer.Option(False, "--skip-stance", help="Skip stance classification"),
    skip_quality: bool = typer.Option(False, "--skip-quality", help="Skip evidence quality tagging"),
    skip_selection: bool = typer.Option(False, "--skip-selection", help="Skip contradiction-aware selection"),
    use_retrieval: bool = typer.Option(False, "--use-retrieval", help="Use index-based retrieval instead of gold docs"),
    retrieval_strategy: str = typer.Option("hybrid", "--retrieval-strategy", help="bm25 / embedding / hybrid / mmr / random"),
    model_override: Optional[str] = typer.Option(None, "--model-override", help="Override synthesis model (e.g. claude-3-5-haiku-20241022)"),
    max_budget: float = typer.Option(0.0, "--max-budget", help="Max API spend in USD (0 = unlimited)"),
) -> None:
    """Run the contradiction-aware RAG pipeline."""
    cfg = _load_config(config)
    experiment_name = cfg.get("experiment_name", Path(config).stem)

    if run_id is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        run_id = f"{experiment_name}_{timestamp}"

    run_dir = _setup_run_dir(run_id)
    start_time = time.perf_counter()

    # Save config copy (with CLI overrides noted)
    cfg_copy = dict(cfg)
    cfg_copy["_cli_overrides"] = {
        "strategy": strategy,
        "dataset_path": dataset_path,
        "skip_pico": skip_pico,
        "skip_stance": skip_stance,
        "skip_quality": skip_quality,
        "skip_selection": skip_selection,
        "use_retrieval": use_retrieval,
        "retrieval_strategy": retrieval_strategy,
        "model_override": model_override,
        "max_budget": max_budget,
    }
    with open(run_dir / "config.yaml", "w") as f:
        yaml.dump(cfg_copy, f)

    ablation_condition = _derive_ablation_condition(
        skip_pico, skip_stance, skip_quality, skip_selection
    )

    print(f"\n{'='*60}")
    print(f"  Experiment: {experiment_name}")
    print(f"  Run ID:     {run_id}")
    print(f"  Strategy:   {strategy}  |  Ablation: {ablation_condition}")
    if model_override:
        print(f"  Model:      {model_override}")
    if use_retrieval:
        print(f"  Retrieval:  {retrieval_strategy}")
    if max_budget > 0:
        print(f"  Budget:     ${max_budget:.2f}")
    print(f"{'='*60}\n")

    # ── Load dataset ──────────────────────────────────────────────
    dataset_cfg = cfg.get("dataset", {})
    effective_path = dataset_path or dataset_cfg.get("path", "")
    if not Path(effective_path).exists():
        print(f"[ERROR] Dataset not found: {effective_path}")
        raise typer.Exit(1)

    examples = load_dataset_jsonl(effective_path)
    print(f"[Pipeline] Loaded {len(examples)} examples from {effective_path}")

    limit = max_examples or dataset_cfg.get("max_examples")
    if limit and limit > 0:
        examples = examples[:limit]
        print(f"[Pipeline] Limited to {len(examples)} examples")

    # ── Initialize components ─────────────────────────────────────
    budget_arg = max_budget if max_budget > 0 else None
    client = CachedAnthropicClient(cache_dir="./cache", max_budget_usd=budget_arg)
    pipeline_cfg = cfg.get("pipeline", {})

    pico_extractor = (
        PICOExtractor(client) if not skip_pico and pipeline_cfg.get("pico_extraction") else None
    )
    stance_classifier = (
        StanceClassifier(client) if not skip_stance and pipeline_cfg.get("stance_classification") else None
    )
    synthesis_generator = SynthesisGenerator(client)

    # ── Retrieval indices (loaded once at startup) ─────────────────
    bm25_index = None
    embedding_index = None
    corpus: dict[str, dict] | None = None
    all_doc_ids: list[str] | None = None

    if use_retrieval:
        from src.paper4.retrieval.bm25_index import BM25Index
        from src.paper4.retrieval.embedding_index import EmbeddingIndex

        full_corpus_path = dataset_cfg.get("full_corpus_path", effective_path)
        print(f"[Pipeline] Loading retrieval corpus from {full_corpus_path}...")
        corpus = _build_corpus_dict(full_corpus_path)
        all_doc_ids = list(corpus.keys())
        print(f"[Pipeline] Corpus: {len(corpus)} documents")

        _model_name = cfg.get("retrieval", {}).get("model_name", "healthcontradict")
        if retrieval_strategy in ("bm25", "hybrid"):
            bm25_path = f"models/bm25_{_model_name}.pkl"
            if Path(bm25_path).exists():
                bm25_index = BM25Index()
                bm25_index.load(bm25_path)
                print(f"[Pipeline] Loaded BM25 index: {len(bm25_index.doc_ids)} docs")
            else:
                print(f"[WARN] BM25 index not found at {bm25_path}")

        if retrieval_strategy in ("embedding", "hybrid", "mmr"):
            emb_path = f"models/embeddings_{_model_name}"
            if Path(emb_path + ".faiss").exists():
                embedding_index = EmbeddingIndex()
                embedding_index.load(emb_path)
                print(f"[Pipeline] Loaded embedding index: {embedding_index.size} vectors")
            else:
                print(f"[WARN] Embedding index not found at {emb_path}.faiss")

    # ── Checkpoint setup ──────────────────────────────────────────
    ckpt = CheckpointManager(run_id, checkpoint_dir=str(run_dir / "checkpoints"))

    if ckpt.is_complete() and not resume:
        print(f"[Pipeline] Run already complete. Use --resume to rerun or choose a new --run-id.")
        raise typer.Exit(0)

    items = [{"example_id": ex.example_id, "example": ex} for ex in examples]
    remaining, completed = ckpt.start_or_resume(items, id_key="example_id")

    # ── Process examples ──────────────────────────────────────────
    budget_cfg = cfg.get("budget", {})
    alert_every = budget_cfg.get("cost_alert_every_n_calls", 50)
    max_time = max_minutes * 60 if max_minutes else (budget_cfg.get("max_minutes", 0) * 60)

    n_initial_done = len(completed)
    bar = tqdm(
        enumerate(remaining),
        total=len(examples),
        initial=n_initial_done,
        desc=experiment_name,
        unit="ex",
        dynamic_ncols=True,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}",
    )
    for i, item in bar:
        if max_time > 0:
            elapsed = time.perf_counter() - start_time
            if elapsed > max_time:
                bar.write(f"[Pipeline] Time limit reached. Rerun with --resume.")
                break

        example = item["example"]
        example_id = item["example_id"]
        bar.set_postfix(id=example_id, refresh=False)

        try:
            result = _process_one_example(
                example=example,
                config=cfg,
                client=client,
                pico_extractor=pico_extractor,
                stance_classifier=stance_classifier,
                synthesis_generator=synthesis_generator,
                strategy=strategy,
                skip_pico=skip_pico,
                skip_stance=skip_stance,
                skip_quality=skip_quality,
                skip_selection=skip_selection,
                use_retrieval=use_retrieval,
                retrieval_strategy=retrieval_strategy,
                model_override=model_override or None,
                ablation_condition=ablation_condition,
                bm25_index=bm25_index,
                embedding_index=embedding_index,
                corpus=corpus,
                all_doc_ids=all_doc_ids,
            )
            ckpt.save(example_id, result)
            completed[example_id] = result
            _s = client.cost
            _rate = _s.cache_hits / _s.total_calls if _s.total_calls else 0.0
            bar.set_postfix(
                id=example_id,
                cost=f"${_s.cost_usd:.3f}",
                cache=f"{_rate:.0%}",
                refresh=True,
            )

        except BudgetExhaustedError as e:
            bar.write(f"\n{e}")
            bar.write(
                f"[Pipeline] Saved {len(completed)} examples. "
                f"Rerun with --resume after adding credits."
            )
            bar.close()
            # Write partial per_example.jsonl so progress isn't lost
            _write_per_example(run_dir, examples, completed)
            client.print_cost_summary()
            raise typer.Exit(2)

        except KeyboardInterrupt:
            bar.write(f"[Pipeline] Interrupted after {len(completed)} examples. Rerun with --resume.")
            bar.close()
            break

        except Exception as e:
            bar.write(f"[ERROR] {example_id}: {e}")
            ckpt.save(example_id, {"error": str(e), "example_id": example_id})
            completed[example_id] = {"error": str(e)}

        if (i + 1) % alert_every == 0:
            _sr = client.cost
            _rr = _sr.cache_hits / _sr.total_calls if _sr.total_calls else 0.0
            bar.write(f"[Cost] calls={_sr.total_calls} cost=${_sr.cost_usd:.3f} cache={_rr:.0%}")

    bar.close()

    # ── Finalize ──────────────────────────────────────────────────
    if len(completed) == len(examples):
        ckpt.finalize(completed)

    _write_per_example(run_dir, examples, completed)
    print(f"\n[Pipeline] {len(completed)}/{len(examples)} examples → {run_dir / 'per_example.jsonl'}")

    elapsed = time.perf_counter() - start_time
    print(f"[Pipeline] Elapsed: {elapsed:.1f}s ({elapsed/60:.1f}m)")
    client.print_cost_summary()


def _write_per_example(run_dir: Path, examples: list, completed: dict) -> None:
    """Write per_example.jsonl from completed results (overwrites if exists)."""
    per_example_path = run_dir / "per_example.jsonl"
    with open(per_example_path, "w") as f:
        for ex in examples:
            result = completed.get(ex.example_id, {})
            f.write(json.dumps(result, default=str) + "\n")


if __name__ == "__main__":
    app()

