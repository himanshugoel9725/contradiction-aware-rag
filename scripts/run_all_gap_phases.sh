#!/usr/bin/env bash
# Run all gap-filling experiment phases sequentially.
# Each phase: pipeline → evaluate (with judge)
# Run from workspace root with venv active:
#   source .venv/bin/activate && bash scripts/run_all_gap_phases.sh
set -euo pipefail

PYTHON=".venv/bin/python"

phase() { echo; echo "========================================"; echo "  $*"; echo "========================================"; }
run_pipeline() { $PYTHON scripts/run_pipeline.py "$@"; }
run_eval()     { $PYTHON scripts/evaluate.py "$@" --run-judge; }

# ── Phase 2b: Evaluate strategy A (already ran) ──────────────────────────────
phase "Phase 2b: Evaluate gap1_strategy_A"
run_eval --run-dir runs/gap1_strategy_A

# ── Phase 2: Strategy B, C, D ─────────────────────────────────────────────────
phase "Phase 2: gap1_strategy_B"
run_pipeline --config configs/experiment_gap1_stratB.yaml --run-id gap1_strategy_B --strategy B
run_eval --run-dir runs/gap1_strategy_B

phase "Phase 2: gap1_strategy_C"
run_pipeline --config configs/experiment_gap1_stratC.yaml --run-id gap1_strategy_C --strategy C
run_eval --run-dir runs/gap1_strategy_C

phase "Phase 2: gap1_strategy_D"
run_pipeline --config configs/experiment_gap1_stratD.yaml --run-id gap1_strategy_D --strategy D
run_eval --run-dir runs/gap1_strategy_D

# ── Phase 3: Ablation (uses gap1_strategy_C as "full" — already evaluated) ───
phase "Phase 3: gap4_minus_pico"
run_pipeline --config configs/experiment_gap4_ablation.yaml --run-id gap4_minus_pico --strategy C --skip-pico
run_eval --run-dir runs/gap4_minus_pico

phase "Phase 3: gap4_minus_stance"
run_pipeline --config configs/experiment_gap4_ablation.yaml --run-id gap4_minus_stance --strategy C --skip-stance
run_eval --run-dir runs/gap4_minus_stance

phase "Phase 3: gap4_minus_quality"
run_pipeline --config configs/experiment_gap4_ablation.yaml --run-id gap4_minus_quality --strategy C --skip-quality
run_eval --run-dir runs/gap4_minus_quality

phase "Phase 3: gap4_minus_selection"
run_pipeline --config configs/experiment_gap4_ablation.yaml --run-id gap4_minus_selection --strategy C --skip-selection
run_eval --run-dir runs/gap4_minus_selection

# ── Phase 4: Backbone comparison ─────────────────────────────────────────────
# gap1_strategy_C (sonnet) is the sonnet baseline — already done above
phase "Phase 4: gap5_haiku"
run_pipeline --config configs/experiment_gap5_backbone.yaml --run-id gap5_haiku --model-override claude-haiku-4-5-20251001
run_eval --run-dir runs/gap5_haiku

phase "Phase 4: gap5_gpt4o_mini"
run_pipeline --config configs/experiment_gap5_backbone.yaml --run-id gap5_gpt4o_mini --model-override gpt-4o-mini
run_eval --run-dir runs/gap5_gpt4o_mini

# ── Phase 5: Retrieval strategies ─────────────────────────────────────────────
for strat in bm25 embedding hybrid mmr random; do
  phase "Phase 5: gap7_retrieval_$strat"
  run_pipeline --config configs/experiment_gap7_retrieval.yaml \
    --run-id "gap7_retrieval_${strat}" \
    --use-retrieval --retrieval-strategy "$strat"
  run_eval --run-dir "runs/gap7_retrieval_${strat}"
done

# ── Phase 6: SciFact cross-dataset ────────────────────────────────────────────
phase "Phase 6: prepare SciFact data"
$PYTHON scripts/prepare_scifact.py

phase "Phase 6: gap8_scifact_gold_stratC"
run_pipeline --config configs/experiment_gap8_scifact.yaml --run-id gap8_scifact_gold_stratC
run_eval --run-dir runs/gap8_scifact_gold_stratC

phase "Phase 6: gap8_scifact_retrieval_hybrid"
run_pipeline --config configs/experiment_gap8_scifact_retrieval.yaml \
  --run-id gap8_scifact_retrieval_hybrid \
  --use-retrieval --retrieval-strategy hybrid
run_eval --run-dir runs/gap8_scifact_retrieval_hybrid

# ── Phase 8: Generate all figures ────────────────────────────────────────────
phase "Phase 8: Generate all figures"
$PYTHON scripts/generate_figures.py --figure-set all --output-dir figures/

echo
echo "All phases complete!"
ls runs/ | grep -E "^gap" | sort
