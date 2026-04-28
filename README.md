# Paper 4: Contradiction-Aware Biomedical RAG

Contradiction-Aware Retrieval-Augmented Generation for Biomedical Literature.

When retrieved biomedical evidence contains both supporting and refuting findings,
vanilla RAG systems produce one-sided or overconfident syntheses — a failure mode
we call **"contradiction blindness."** This pipeline explicitly detects, represents,
and synthesizes contradictory evidence, measured by novel metrics (CBR, CAS, VCS, EAA, EQU).

## Stoppage and Resume

Every operation supports interruption and resumption:
- **API calls**: Cached in SQLite — rerun costs $0
- **Per-example**: CheckpointManager saves after each example — resume skips completed
- **Per-experiment**: `_DONE` marker — entire experiment skipped on rerun
- **Time budget**: `--max-minutes` flag for graceful early exit
- **Ctrl+C**: Catches interrupt, saves checkpoint before exit
