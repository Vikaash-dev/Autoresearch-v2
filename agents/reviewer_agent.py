"""
Reviewer Agent — adversarial review simulation using Reviewer-ToM.

Implements the pre-submit prediction loop:
  1. Query ReviewerToM for the expected review
  2. Score the draft (predicted_score, major_concerns, required_changes)
  3. Return feedback to the WriterAgent for targeted revision

Inspired by:
  - NousResearch/autonovel Opus Review Loop (dual-persona literary critic + professor)
  - AI-Scientist v2 LLM review phase
  - The key insight: a different model from the writer gives more honest review
"""

from __future__ import annotations

import logging
from typing import Any

from agents.base_agent import BaseAgent
from core.blackboard import Blackboard
from tom.engine import TheoryOfMindEngine

logger = logging.getLogger(__name__)

_REVIEWER_SYSTEM = """You are a rigorous, adversarial scientific reviewer.
You review papers for top ML conferences. You are fair but demanding.
You will find genuine weaknesses — not just superficial ones.
Do NOT be sycophantic. Point out missing baselines, missing ablations,
weak experimental design, overclaiming, and novelty issues.
"""


class ReviewerAgent(BaseAgent):
    """Adversarial paper reviewer using Reviewer-ToM predictions."""

    def __init__(
        self,
        blackboard: Blackboard,
        tom_engine: TheoryOfMindEngine,
        llm_fn: Any = None,
    ) -> None:
        super().__init__("reviewer", blackboard, tom_engine, llm_fn)

    def act(self, context: dict[str, Any]) -> dict[str, Any]:
        """
        Review a paper draft and return structured feedback.

        Context keys:
            paper_draft: str — the full paper text
            reviewer_model: ReviewerModel (optional, uses ToM engine model if not given)
        """
        paper_draft: str = context.get("paper_draft", "")
        reviewer_model = context.get("reviewer_model") or self._tom.reviewer_model

        if not paper_draft:
            return {"status": "failed", "error": "No paper draft provided"}

        # Use Reviewer-ToM to predict the review
        review = self._tom.predict_review(paper_draft)

        # If we have a real LLM, do a dual-persona review (autonovel Opus loop pattern)
        if self._llm is not None:
            detailed = self._dual_persona_review(paper_draft, reviewer_model)
            review.update(detailed)

        self.update_own_belief(
            last_action=f"reviewed_draft_score={review.get('predicted_score','?')}",
            current_uncertainty=0.2,
        )
        self.record_success("review")

        return {**review, "status": "ok"}

    def _dual_persona_review(self, paper_draft: str, reviewer_model: Any) -> dict[str, Any]:
        """
        Dual-persona review: first as technical expert, then as domain critic.
        Directly inspired by NousResearch/autonovel's Opus Review Loop.
        """
        prompt = (
            "Review the following paper twice:\n"
            "First as a TECHNICAL EXPERT: evaluate methodology, experiments, statistics.\n"
            "Second as a DOMAIN CRITIC: evaluate novelty, framing, related work coverage.\n\n"
            f"Venue: {reviewer_model.venue}\n"
            f"Domain: {reviewer_model.domain}\n"
            f"Known pet peeves at this venue: {reviewer_model.pet_peeves}\n\n"
            f"Paper (first 4000 chars):\n{paper_draft[:4000]}\n\n"
            "Return your review as JSON:\n"
            "{\n"
            '  "technical_score": <1-10>,\n'
            '  "novelty_score": <1-10>,\n'
            '  "predicted_score": <average, 1-10>,\n'
            '  "accept_probability": <0-1>,\n'
            '  "would_accept": <bool>,\n'
            '  "major_concerns": [<list of specific, actionable concerns>],\n'
            '  "required_changes": [<list of required additions/fixes>],\n'
            '  "strengths": [<list>],\n'
            '  "reasoning": "<one paragraph summary>"\n'
            "}"
        )
        import json
        try:
            raw = self.call_llm(prompt, system=_REVIEWER_SYSTEM)
            raw = raw.strip().lstrip("```json").rstrip("```").strip()
            return json.loads(raw)
        except Exception as exc:
            logger.warning("Dual-persona review failed: %s", exc)
            return {
                "major_concerns": ["Review failed"],
                "required_changes": [],
                "reasoning": str(exc),
            }
