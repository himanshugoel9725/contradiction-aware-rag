"""
Structured Synthesis — generate contradiction-aware responses.

WHY: Standard RAG generation ignores contradictions among retrieved documents.
A good answer should explicitly acknowledge disagreements, explain why studies
might differ, and provide a balanced conclusion.

HOW: Four generation strategies (A–D) with increasing contradiction awareness:
    A — Vanilla:        Standard prompt with retrieved context, no special formatting
    B — Stance-labeled: Documents prefixed with [SUPPORT]/[CONTRADICT] labels
    C — Structured:     Four-section output: Agreement / Disagreement / Quality / Conclusion
    D — PICO-decomposed: Evidence grouped by PICO element, contradictions per element

The strategy is selected via config. Strategy C is the primary/recommended approach.
"""

from __future__ import annotations

from typing import Any


def _clip_text(text: str, max_chars: int = 3500) -> str:
    """Trim long document text to keep generation prompts within budget."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n\n[TRUNCATED]"


# ── System prompts per strategy ───────────────────────────────────────────────

SYSTEM_PROMPTS = {
    "A": """You are a biomedical evidence summarizer. Given a claim and retrieved documents, 
provide a comprehensive answer based on the evidence. Cite specific documents when making claims.""",

    "B": """You are a biomedical evidence summarizer. The retrieved documents are labeled with their 
stance toward the claim: [SUPPORT], [CONTRADICT], or [NEUTRAL]. 
Provide a comprehensive answer that considers the evidence from both sides. 
Cite specific documents when making claims.""",

    "C": """You are a biomedical evidence synthesizer specializing in contradiction detection.
Given a claim and retrieved documents (with stance labels), produce a structured synthesis with exactly four sections:

## Agreement
Summarize evidence that supports the claim. Cite specific documents.

## Disagreement  
Summarize evidence that contradicts the claim. Cite specific documents.

## Evidence Quality
Note differences in study design, population sizes, or methodology that may explain contradictions.

## Conclusion
Provide a balanced conclusion that explicitly acknowledges the contradiction and states which position has stronger evidence and why.

If there are no contradictions, still use all four sections but note the absence of disagreement.""",

    "D": """You are a biomedical evidence synthesizer with PICO expertise.
The claim has been decomposed into PICO elements (Population, Intervention, Comparator, Outcome).
For each PICO element, assess whether the evidence agrees or disagrees.

Structure your response as:
## Population
[Whether studies agree on the relevant population, any differences]

## Intervention  
[Whether studies agree on the intervention details]

## Outcome
[Key section: whether studies agree on the outcome, noting contradictions]

## Synthesis
Provide a balanced conclusion. If studies disagree, explain WHY they might disagree 
(different populations, doses, endpoints, study designs) and which evidence is stronger.""",
}


def _format_context_vanilla(documents: list[dict[str, Any]]) -> str:
    """Strategy A: plain text context."""
    parts = []
    for i, doc in enumerate(documents, 1):
        title = doc.get("title", "")
        text = _clip_text(doc.get("text", ""))
        header = f"[Document {i}]"
        if title:
            header += f" {title}"
        parts.append(f"{header}\n{text}")
    return "\n\n".join(parts)


def _format_context_stance_labeled(documents: list[dict[str, Any]]) -> str:
    """Strategy B: docs prefixed with stance labels."""
    parts = []
    for i, doc in enumerate(documents, 1):
        label = doc.get("stance_label", "NEUTRAL")
        title = doc.get("title", "")
        text = _clip_text(doc.get("text", ""))
        header = f"[Document {i}] [{label}]"
        if title:
            header += f" {title}"
        parts.append(f"{header}\n{text}")
    return "\n\n".join(parts)


def _format_context_structured(documents: list[dict[str, Any]]) -> str:
    """Strategy C: same as stance-labeled (the structure is in the prompt)."""
    return _format_context_stance_labeled(documents)


def _format_context_pico(
    documents: list[dict[str, Any]],
    pico: dict[str, Any] | None = None,
) -> str:
    """Strategy D: documents + PICO decomposition context."""
    parts = []

    if pico:
        parts.append("PICO Decomposition:")
        parts.append(f"  Population: {pico.get('population', 'N/A')}")
        parts.append(f"  Intervention: {pico.get('intervention', 'N/A')}")
        parts.append(f"  Comparator: {pico.get('comparator', 'N/A')}")
        parts.append(f"  Outcome: {pico.get('outcome', 'N/A')}")
        parts.append("")

    parts.append(_format_context_stance_labeled(documents))
    return "\n".join(parts)


_CONTEXT_FORMATTERS = {
    "A": _format_context_vanilla,
    "B": _format_context_stance_labeled,
    "C": _format_context_structured,
    "D": _format_context_pico,
}


class SynthesisGenerator:
    """
    Generates contradiction-aware synthesis using one of 4 strategies (A–D).

    The strategy is specified at generation time, allowing easy comparison
    in ablation experiments.
    """

    def __init__(self, client: Any) -> None:
        """
        Args:
            client: CachedOpenAIClient instance.
        """
        self.client = client

    def generate(
        self,
        claim: str,
        documents: list[dict[str, Any]],
        strategy: str = "C",
        pico: dict[str, Any] | None = None,
        max_tokens: int = 2048,
        model_task_type: str = "generation",
        model_override: str | None = None,
    ) -> dict[str, Any]:
        """
        Generate a synthesis response.

        Args:
            claim: Biomedical claim text.
            documents: Retrieved docs with stance labels.
            strategy: "A", "B", "C", or "D".
            pico: PICO extraction dict (required for strategy D).
            max_tokens: Max output tokens.
            model_task_type: Key into model routing table.

        Returns:
            Dict with 'text' (generated response), 'strategy', 'model', 'usage'.
        """
        system_prompt = SYSTEM_PROMPTS.get(strategy, SYSTEM_PROMPTS["C"])

        # Format context based on strategy
        if strategy == "D":
            context = _format_context_pico(documents, pico)
        else:
            formatter = _CONTEXT_FORMATTERS.get(strategy, _format_context_structured)
            context = formatter(documents)

        user_message = f"Claim: {claim}\n\nEvidence:\n{context}"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]

        response = self.client.chat(
            model_task_type,
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.0,
            model_override=model_override,
        )

        content = self.client.extract_content(response)
        usage = response.get("usage", {})

        return {
            "text": content,
            "strategy": strategy,
            "model": response.get("model", ""),
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
        }
