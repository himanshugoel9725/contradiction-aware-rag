#!/usr/bin/env bash
# Run only the remaining phases (gap7, gap8, figures).
# Gaps 1/4/5 are already complete.
set -euo pipefail

PYTHON=".venv/bin/python"
phase() { echo; echo "========================================"; echo "  $*"; echo "========================================"; }
run_pipeline() { $PYTHON scripts/run_pipeline.py "$@"; }
run_eval()     { $PYTHON scripts/evaluate.py "$@" --run-judge; }

# ── Phase 5: Retrieval strategies (gap7) ─────────────────────────────────────
for strat in bm25 embedding hybrid mmr random; do
  phase "Phase 5: gap7_retrieval_${strat}"
  run_pipeline --config configs/experiment_gap7_retrieval.yaml \
    --run-id "gap7_retrieval_${strat}" \
    --use-retrieval --retrieval-strategy "$strat"
  run_eval --run-dir "runs/gap7_retrieval_${strat}"
done

# ── Phase 6: SciFact cross-dataset ────────────────────────────────────────────
phase "Phase 6: prepare SciFact data"
$PYTHON scripts/prepare_scifact.py

phase "Phase 6: build SciFact BM25 index from full corpus"
$PYTHON - << 'PYEOF'
import sys, json
sys.path.insert(0, '.')
from src.paper4.retrieval.bm25_index import BM25Index
docs = []
with open('data_processed/scifact_corpus.jsonl') as f:
    for line in f:
        line = line.strip()
        if line:
            docs.append(json.loads(line))
print(f"[SciFact BM25] Building index from {len(docs)} documents...")
idx = BM25Index()
idx.build(docs)
idx.save('models/bm25_scifact.pkl')
PYEOF

phase "Phase 6: build SciFact embedding index from full corpus"
$PYTHON - << 'PYEOF'
import sys, json
sys.path.insert(0, '.')
import src.paper4.utils.env_setup  # noqa
from src.paper4.retrieval.embedding_index import EmbeddingIndex
from src.paper4.llm.openai_client import CachedOpenAIClient
docs = []
with open('data_processed/scifact_corpus.jsonl') as f:
    for line in f:
        line = line.strip()
        if line:
            docs.append(json.loads(line))
print(f"[SciFact Embeddings] Building index from {len(docs)} documents...")
client = CachedOpenAIClient(cache_dir='./cache')
idx = EmbeddingIndex()
idx.build(docs, client, batch_size=100)
idx.save('models/embeddings_scifact')
client.print_cost_summary()
PYEOF

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
echo "============================================"
echo "  All remaining phases complete!"
echo "============================================"
ls runs/ | grep "^gap" | sort
