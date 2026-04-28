"""
Paper 4: Contradiction-Aware Retrieval-Augmented Generation for Biomedical Literature.

Package structure:
    data/       — Dataset download, parsing, and schema normalization
    retrieval/  — BM25 sparse and OpenAI dense embedding retrieval indices
    llm/        — OpenAI client wrapper with SQLite caching, retries, and cost tracking
    pipeline/   — Orchestration stages: PICO → stance → selection → synthesis → postcheck
    eval/       — Novel metrics (CBR, CAS, VCS, EAA, EQU), LLM judge rubrics, bootstrap CIs
    utils/      — Environment setup, checkpointing, GRADE hierarchy
"""

__version__ = "0.1.0"
