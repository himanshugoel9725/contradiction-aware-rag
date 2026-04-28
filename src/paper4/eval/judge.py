"""
LLM Judge — rubric-based evaluation of synthesis quality using GPT-4o.

WHY: CAS (Contradiction Acknowledgment Score) and some aspects of VCS/EAA
require semantic judgment that can't be computed with string matching alone.
An LLM judge applies a standardized rubric to produce scores on a 1-5 scale.

HOW: The judge receives the claim, evidence documents, and generated response.
It applies a rubric and returns a structured JSON score. Two complexity levels:
    - judge_simple: For clear-cut cases (gpt-4o-mini, cheaper)
    - judge_complex: For nuanced cases (gpt-4o, more accurate)
"""

from __future__ import annotations

from typing import Any


CAS_RUBRIC = """Rate the response's handling of contradictions on a 1-5 scale:

5 — Excellent: Explicitly identifies the contradiction, discusses both sides with citations, 
    explains possible reasons for disagreement (population, methodology, etc.), and provides a 
    balanced conclusion.

4 — Good: Explicitly mentions the contradiction and discusses both sides, but may lack 
    depth in explaining reasons for disagreement.

3 — Adequate: Mentions that evidence is mixed or conflicting, but doesn't deeply 
    engage with both sides.

2 — Poor: Briefly mentions one side more than the other, or only implicitly 
    acknowledges the contradiction.

1 — Very poor: Makes no attempt to surface the contradiction. Presents only one side 
    as if there were consensus.

Consider:
- Does the response explicitly state that studies disagree?
- Are both supporting and contradicting studies cited?
- Is there an explanation of WHY studies might disagree?
- Is the conclusion appropriately hedged/balanced?

Return your score and reasoning in structured JSON."""

VCS_RUBRIC = """Rate how well the response covers all viewpoints present in the evidence (1-5 scale):

5 — Covers all major viewpoints from the evidence, giving proportional attention to each.
4 — Covers most viewpoints but may underemphasize a minor one.
3 — Covers supporting and contradicting views but with clear imbalance.
2 — Primarily covers one side with brief mention of the other.
1 — Only covers one viewpoint despite multiple being present.

Return your score and reasoning in structured JSON."""

EAA_RUBRIC = """Rate the accuracy of evidence attribution in the response (1-5 scale):

5 — Every claim in the response is properly attributed to a specific document, 
    and all attributions are accurate.
4 — Most claims are attributed, with only minor issues.
3 — Some claims are attributed but others are not, or some attributions are vague.
2 — Few attributions, or some are inaccurate.
1 — No attributions, or attributions are mostly incorrect.

Return your score and reasoning in structured JSON."""


JUDGE_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "judge_evaluation",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "score": {"type": "integer"},
                "reasoning": {"type": "string"},
                "key_observations": {"type": "string"},
            },
            "required": ["score", "reasoning", "key_observations"],
            "additionalProperties": False,
        },
    },
}


class LLMJudge:
    """
    Rubric-based LLM evaluation for CAS, VCS, and EAA.

    Uses CachedOpenAIClient so repeated evaluations are cache-free.
    """

    def __init__(self, client: Any) -> None:
        self.client = client

    def evaluate_cas(
        self,
        claim: str,
        documents: list[dict[str, Any]],
        response_text: str,
        use_complex: bool = False,
    ) -> dict[str, Any]:
        """
        Evaluate Contradiction Acknowledgment Score (CAS).

        Args:
            claim: Biomedical claim.
            documents: Retrieved evidence with stance labels.
            response_text: Generated synthesis text.
            use_complex: If True, use gpt-4o (judge_complex); else gpt-4o-mini.

        Returns:
            Dict with score (1-5), normalized_score (0-1), reasoning.
        """
        return self._judge(
            rubric=CAS_RUBRIC,
            metric_name="CAS",
            claim=claim,
            documents=documents,
            response_text=response_text,
            task_type="judge_complex" if use_complex else "judge_simple",
        )

    def evaluate_vcs(
        self,
        claim: str,
        documents: list[dict[str, Any]],
        response_text: str,
        use_complex: bool = False,
    ) -> dict[str, Any]:
        """Evaluate Viewpoint Coverage Score via LLM judge."""
        return self._judge(
            rubric=VCS_RUBRIC,
            metric_name="VCS",
            claim=claim,
            documents=documents,
            response_text=response_text,
            task_type="judge_complex" if use_complex else "judge_simple",
        )

    def evaluate_eaa(
        self,
        claim: str,
        documents: list[dict[str, Any]],
        response_text: str,
        use_complex: bool = False,
    ) -> dict[str, Any]:
        """Evaluate Evidence Attribution Accuracy via LLM judge."""
        return self._judge(
            rubric=EAA_RUBRIC,
            metric_name="EAA",
            claim=claim,
            documents=documents,
            response_text=response_text,
            task_type="judge_complex" if use_complex else "judge_simple",
        )

    def _judge(
        self,
        rubric: str,
        metric_name: str,
        claim: str,
        documents: list[dict[str, Any]],
        response_text: str,
        task_type: str,
    ) -> dict[str, Any]:
        """Internal: run one judge evaluation."""
        doc_context = "\n\n".join(
            f"[Document {i+1}] [{d.get('stance_label', 'UNKNOWN')}] {d.get('text', '')[:500]}"
            for i, d in enumerate(documents)
        )

        messages = [
            {
                "role": "system",
                "content": f"You are an expert evaluator for biomedical RAG systems.\n\n{rubric}",
            },
            {
                "role": "user",
                "content": (
                    f"Claim: {claim}\n\n"
                    f"Evidence documents:\n{doc_context}\n\n"
                    f"Generated response:\n{response_text}"
                ),
            },
        ]

        try:
            response = self.client.chat(
                task_type,
                messages=messages,
                response_format=JUDGE_RESPONSE_FORMAT,
                max_tokens=512,
            )
            parsed = self.client.extract_json(response)

            score = max(1, min(5, int(parsed.get("score", 1))))
            normalized = (score - 1) / 4.0  # Map 1-5 to 0-1

            return {
                "metric": metric_name,
                "score": score,
                "normalized_score": normalized,
                "reasoning": parsed.get("reasoning", ""),
                "key_observations": parsed.get("key_observations", ""),
            }

        except Exception as e:
            return {
                "metric": metric_name,
                "score": 1,
                "normalized_score": 0.0,
                "reasoning": f"Judge evaluation failed: {e}",
                "key_observations": "",
                "_error": str(e),
            }

    def evaluate_all(
        self,
        claim: str,
        documents: list[dict[str, Any]],
        response_text: str,
        use_complex: bool = False,
    ) -> dict[str, dict[str, Any]]:
        """Run all three judge evaluations."""
        return {
            "cas": self.evaluate_cas(claim, documents, response_text, use_complex),
            "vcs": self.evaluate_vcs(claim, documents, response_text, use_complex),
            "eaa": self.evaluate_eaa(claim, documents, response_text, use_complex),
        }

    def evaluate_eaa_semantic(
        self,
        claim: str,
        synthesis: str,
        documents: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """
        Verify that cited claims actually match their source documents (semantic EAA).

        Unlike the postcheck citation-format check (which only verifies that
        [Document N] refers to a real document), this method checks whether the
        sentence containing the citation is actually supported by the cited document.

        For each citation found, sends the (sentence, document_text) pair to
        claude-3-5-haiku-20241022 with a structured attribution_check schema.

        Args:
            claim: The biomedical claim being addressed.
            synthesis: The full generated synthesis text.
            documents: Retrieved/selected documents (with 'text' and 'doc_id').

        Returns:
            Dict with semantic_eaa (0–1), total_citations, accurate_citations, details.
        """
        import re

        citation_pattern = re.compile(
            r"\[(?:Document |Doc )?(\d+)\]|(?:Document|document) (\d+)\b"
        )

        # Split synthesis into sentences (simple sentence splitter)
        # We keep them joined to extract the sentence containing each citation.
        sentences: list[str] = []
        for sent in re.split(r"(?<=[.!?])\s+", synthesis):
            sentences.append(sent.strip())

        # Find all (sentence, doc_number) pairs
        citations_to_check: list[dict[str, Any]] = []
        for sent in sentences:
            for match in citation_pattern.finditer(sent):
                num_str = match.group(1) or match.group(2)
                doc_num = int(num_str)
                if 1 <= doc_num <= len(documents):
                    citations_to_check.append({
                        "sentence": sent,
                        "doc_num": doc_num,
                        "doc_id": documents[doc_num - 1].get("doc_id", f"doc{doc_num}"),
                    })

        if not citations_to_check:
            return {
                "semantic_eaa": 0.0,
                "total_citations": 0,
                "accurate_citations": 0,
                "details": [],
                "note": "No citations found in synthesis",
            }

        attribution_schema = {
            "type": "json_schema",
            "json_schema": {
                "name": "attribution_check",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "is_accurate": {"type": "boolean"},
                        "reason": {"type": "string"},
                    },
                    "required": ["is_accurate", "reason"],
                    "additionalProperties": False,
                },
            },
        }

        details: list[dict[str, Any]] = []
        accurate_count = 0

        for item in citations_to_check:
            doc = documents[item["doc_num"] - 1]
            doc_text = doc.get("text", "")[:1500]  # Cap to keep costs low

            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are an evidence attribution verifier. Given a claim sentence "
                        "and the source document it cites, determine if the attribution is accurate."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Claim sentence: {item['sentence']}\n\n"
                        f"Source document: {doc_text}\n\n"
                        "Is the claim in this sentence actually supported by the source document?"
                    ),
                },
            ]

            try:
                response = self.client.chat(
                    "judge_simple",
                    messages=messages,
                    response_format=attribution_schema,
                    max_tokens=256,
                )
                parsed = self.client.extract_json(response)
                is_accurate = bool(parsed.get("is_accurate", False))
                reason = parsed.get("reason", "")
            except Exception as e:
                is_accurate = False
                reason = f"Evaluation failed: {e}"

            if is_accurate:
                accurate_count += 1

            details.append({
                "sentence": item["sentence"],
                "doc_id": item["doc_id"],
                "doc_num": item["doc_num"],
                "is_accurate": is_accurate,
                "reason": reason,
            })

        total = len(citations_to_check)
        return {
            "semantic_eaa": accurate_count / total,
            "total_citations": total,
            "accurate_citations": accurate_count,
            "details": details,
        }
