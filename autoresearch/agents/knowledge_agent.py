"""
Knowledge Agent — knowledge grounding and novelty checking.

Inspired by:
  - AutoResearcher (2025): Knowledge Grounding agent + Novelty agent
  - AIRS-Bench (2026): reliability and reproducibility evaluation

Uses soft acceptance scoring (weighted combination) instead of hard per-dimension
thresholds to reduce false rejections on otherwise strong hypotheses.
"""

from __future__ import annotations

import hashlib
import logging
import re
from typing import Any, Dict, List, Optional, Set

from autoresearch.agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)


class KnowledgeAgent(BaseAgent):
    """
    Validates that generated hypotheses and research ideas are:
      1. Scientifically grounded (backed by prior literature)
      2. Novel (not duplicating existing work)
      3. Consistent (no internal logical contradictions)

    Acceptance uses a *soft* weighted score so a hypothesis that is
    exceptionally novel but moderately grounded is not rejected outright.
    """

    def __init__(self, config: Optional[Any] = None, **kwargs: Any) -> None:
        super().__init__(config=config, name="KnowledgeAgent", **kwargs)
        self._seen_fingerprints: Set[str] = set()

    def _default_parameters(self) -> Dict[str, Any]:
        return {
            "novelty_weight": 0.40,
            "grounding_weight": 0.35,
            "consistency_weight": 0.25,
            "acceptance_threshold": 0.55,   # soft overall score to accept
            "novelty_threshold": 0.40,       # hard floor on novelty (dedup)
        }

    def _execute(self, task: str, context: Dict[str, Any]) -> Dict[str, Any]:
        # Accept both "hypotheses" (orchestrator) and "best_hypotheses" (legacy)
        hypotheses: List[Any] = (
            context.get("hypotheses") or context.get("best_hypotheses") or []
        )
        # Accept both "learnings" (orchestrator) and "papers" (legacy)
        papers_raw: List[Any] = context.get("papers") or []
        learnings:  List[str] = context.get("learnings") or []

        # Convert learnings to paper-like dicts for _check_grounding
        papers: List[Dict[str, Any]] = list(papers_raw)
        for l in learnings:
            papers.append({"title": l, "abstract": l, "key_findings": []})

        results: List[Dict[str, Any]] = []
        for hyp in hypotheses:
            text = hyp.get("hypothesis", "") if isinstance(hyp, dict) else str(hyp)
            novelty = self._check_novelty(text, papers)
            grounding = self._check_grounding(text, papers)
            consistency = self._check_consistency(text, task)
            overall = (
                self._parameters["novelty_weight"] * novelty
                + self._parameters["grounding_weight"] * grounding
                + self._parameters["consistency_weight"] * consistency
            )
            # Soft acceptance: overall score + hard novelty floor (dedup guard)
            accepted = (
                overall >= self._parameters["acceptance_threshold"]
                and novelty >= self._parameters["novelty_threshold"]
            )
            rejection_reason = ""
            if not accepted:
                if novelty < self._parameters["novelty_threshold"]:
                    rejection_reason = f"duplicate or too similar to prior work (novelty={novelty:.2f})"
                else:
                    rejection_reason = (
                        f"overall score {overall:.2f} below threshold "
                        f"{self._parameters['acceptance_threshold']:.2f}"
                    )

            results.append(
                {
                    "hypothesis": text,
                    "novelty_score": round(novelty, 3),
                    "grounding_score": round(grounding, 3),
                    "consistency_score": round(consistency, 3),
                    "overall_score": round(overall, 3),
                    "accepted": accepted,
                    "rejection_reason": rejection_reason,
                }
            )

        accepted = [r for r in results if r["accepted"]]
        return {
            "task": task,
            "evaluated": results,
            "validated_hypotheses": accepted,    # orchestrator reads this key
            "accepted_hypotheses": accepted,     # legacy key
            "acceptance_rate": len(accepted) / max(len(results), 1),
        }

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------

    def _check_novelty(self, hypothesis: str, papers: List[Dict[str, Any]]) -> float:
        """
        Score novelty by checking how different the hypothesis is from prior work.
        Uses fingerprint deduplication + Jaccard overlap scoring.
        """
        fp = self._fingerprint(hypothesis)
        if fp in self._seen_fingerprints:
            return 0.0
        self._seen_fingerprints.add(fp)

        hyp_tokens = set(re.split(r"\W+", hypothesis.lower())) - {""}
        max_overlap = 0.0
        for paper in papers:
            paper_text = " ".join([
                paper.get("title", ""),
                paper.get("abstract", ""),
                " ".join(paper.get("key_findings", [])),
            ]).lower()
            paper_tokens = set(re.split(r"\W+", paper_text)) - {""}
            union = hyp_tokens | paper_tokens
            if union:
                overlap = len(hyp_tokens & paper_tokens) / len(union)
                max_overlap = max(max_overlap, overlap)

        novelty = 1.0 - max_overlap
        return max(0.0, min(novelty, 1.0))

    @staticmethod
    def _check_grounding(hypothesis: str, papers: List[Dict[str, Any]]) -> float:
        """
        Score grounding — how well the hypothesis is supported by literature.
        Normalises by the *hypothesis* token set (not paper tokens) for consistency.
        """
        if not papers:
            return 0.5  # neutral when no papers available
        hyp_tokens = set(re.split(r"\W+", hypothesis.lower())) - {""}
        if not hyp_tokens:
            return 0.0
        total_support = 0.0
        for paper in papers[:5]:
            support_text = " ".join([
                paper.get("methodology", ""),
                " ".join(paper.get("key_findings", [])),
            ]).lower()
            support_tokens = set(re.split(r"\W+", support_text)) - {""}
            if support_tokens:
                overlap = len(hyp_tokens & support_tokens) / len(hyp_tokens)
                total_support += overlap
        return min(total_support / max(len(papers[:5]), 1), 1.0)

    @staticmethod
    def _check_consistency(hypothesis: str, task: str) -> float:
        """
        Consistency check: does the hypothesis address the task?
        Normalises correctly (no arbitrary multiplier).
        """
        task_tokens = set(re.split(r"\W+", task.lower())) - {""}
        hyp_tokens = set(re.split(r"\W+", hypothesis.lower())) - {""}
        if not task_tokens or not hyp_tokens:
            return 1.0
        # Recall: fraction of task tokens mentioned in hypothesis
        overlap = len(task_tokens & hyp_tokens) / len(task_tokens)
        return min(overlap, 1.0)

    @staticmethod
    def _fingerprint(text: str) -> str:
        normalised = re.sub(r"\W+", " ", text.lower()).strip()
        return hashlib.md5(normalised.encode()).hexdigest()  # noqa: S324 (not for security)

    def reset_fingerprints(self) -> None:
        """Clear deduplication cache (call between independent research sessions)."""
        self._seen_fingerprints.clear()

    def _score_result(self, result: Any) -> float:
        if not isinstance(result, dict):
            return 0.0
        rate = result.get("acceptance_rate", 0.0)
        n = len(result.get("evaluated", []))
        if n == 0:
            return 0.5   # no hypotheses to score — neutral
        return rate
