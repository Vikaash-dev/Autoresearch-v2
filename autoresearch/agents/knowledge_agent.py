"""
Knowledge Agent — knowledge grounding and novelty checking.

Inspired by:
  - AutoResearcher (2025): Knowledge Grounding agent + Novelty agent
  - AIRS-Bench (2026): reliability and reproducibility evaluation
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

    In production the knowledge base is connected to an embedding store
    (e.g. Chroma / Pinecone) for semantic similarity checks.
    """

    def __init__(self, config: Optional[Any] = None, **kwargs: Any) -> None:
        super().__init__(config=config, name="KnowledgeAgent", **kwargs)
        self._seen_fingerprints: Set[str] = set()

    def _default_parameters(self) -> Dict[str, Any]:
        return {
            "novelty_threshold": 0.7,       # minimum novelty score to accept
            "grounding_threshold": 0.5,      # minimum grounding score
            "consistency_threshold": 0.6,
            "max_similar_papers": 3,
        }

    def _execute(self, task: str, context: Dict[str, Any]) -> Dict[str, Any]:
        hypotheses: List[Dict[str, Any]] = context.get("best_hypotheses", [])
        papers: List[Dict[str, Any]] = context.get("papers", [])

        results: List[Dict[str, Any]] = []
        for hyp in hypotheses:
            text = hyp.get("hypothesis", "")
            novelty = self._check_novelty(text, papers)
            grounding = self._check_grounding(text, papers)
            consistency = self._check_consistency(text, task)
            overall = (novelty * 0.4 + grounding * 0.35 + consistency * 0.25)
            accepted = (
                novelty >= self._parameters["novelty_threshold"]
                and grounding >= self._parameters["grounding_threshold"]
                and consistency >= self._parameters["consistency_threshold"]
            )
            results.append(
                {
                    "hypothesis": text,
                    "novelty_score": round(novelty, 3),
                    "grounding_score": round(grounding, 3),
                    "consistency_score": round(consistency, 3),
                    "overall_score": round(overall, 3),
                    "accepted": accepted,
                }
            )

        accepted = [r for r in results if r["accepted"]]
        return {
            "task": task,
            "evaluated": results,
            "accepted_hypotheses": accepted,
            "acceptance_rate": len(accepted) / max(len(results), 1),
        }

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------

    def _check_novelty(self, hypothesis: str, papers: List[Dict[str, Any]]) -> float:
        """
        Score novelty by checking how different the hypothesis is from prior work.
        Uses fingerprint deduplication + overlap scoring.
        """
        fp = self._fingerprint(hypothesis)
        if fp in self._seen_fingerprints:
            return 0.0
        self._seen_fingerprints.add(fp)

        hyp_tokens = set(re.split(r"\W+", hypothesis.lower()))
        max_overlap = 0.0
        for paper in papers:
            paper_text = " ".join([
                paper.get("title", ""),
                paper.get("abstract", ""),
                " ".join(paper.get("key_findings", [])),
            ]).lower()
            paper_tokens = set(re.split(r"\W+", paper_text))
            if paper_tokens:
                overlap = len(hyp_tokens & paper_tokens) / len(hyp_tokens | paper_tokens)
                max_overlap = max(max_overlap, overlap)

        # Novelty = inverse of overlap
        novelty = 1.0 - max_overlap
        return max(0.0, min(novelty, 1.0))

    @staticmethod
    def _check_grounding(hypothesis: str, papers: List[Dict[str, Any]]) -> float:
        """
        Score grounding — how well the hypothesis is supported by literature.
        High overlap with methodology/findings → higher grounding.
        """
        if not papers:
            return 0.5  # neutral when no papers available
        hyp_tokens = set(re.split(r"\W+", hypothesis.lower()))
        total_support = 0.0
        for paper in papers[:5]:
            support_text = " ".join([
                paper.get("methodology", ""),
                " ".join(paper.get("key_findings", [])),
            ]).lower()
            support_tokens = set(re.split(r"\W+", support_text))
            if support_tokens:
                overlap = len(hyp_tokens & support_tokens) / len(support_tokens)
                total_support += overlap
        return min(total_support / len(papers[:5]), 1.0)

    @staticmethod
    def _check_consistency(hypothesis: str, task: str) -> float:
        """
        Simple consistency check: does the hypothesis address the task?
        Production: use LLM-based logical consistency checker.
        """
        task_tokens = set(re.split(r"\W+", task.lower()))
        hyp_tokens = set(re.split(r"\W+", hypothesis.lower()))
        if not task_tokens:
            return 1.0
        overlap = len(task_tokens & hyp_tokens) / len(task_tokens)
        return min(overlap * 2, 1.0)

    @staticmethod
    def _fingerprint(text: str) -> str:
        normalised = re.sub(r"\W+", " ", text.lower()).strip()
        return hashlib.md5(normalised.encode()).hexdigest()  # noqa: S324 (not for security)

    def _score_result(self, result: Any) -> float:
        return result.get("acceptance_rate", 0.0)
