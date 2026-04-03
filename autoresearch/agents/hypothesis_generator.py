"""Hypothesis Generator Agent."""
from __future__ import annotations
from typing import Any, TYPE_CHECKING
from .base import BaseAgent

if TYPE_CHECKING:
    from ..core.objective_graph import Objective
    from ..core.state import RunState
    from ..config.schema import AutoResearchConfig


class HypothesisGeneratorAgent(BaseAgent):
    name = "hypothesis_generator"
    role = "Hypothesis Generator"

    def run(self, objective: "Objective", state: "RunState", config: "AutoResearchConfig") -> Any:
        self.logger.info("Generating hypotheses for: %s", state.topic)
        evidence_summary = f"{len(state.evidence)} evidence items collected"
        raw = self._llm_prompt(
            f"Based on the topic '{state.topic}' and {evidence_summary}, "
            "generate 3 novel, testable hypotheses. For each hypothesis provide: "
            "title, description, rationale, testability_score (0-1), novelty_score (0-1)."
        )
        hypotheses = [
            {
                "id": f"hyp_{i}",
                "title": f"Hypothesis {i+1} for {state.topic}",
                "description": raw,
                "testability_score": 0.8,
                "novelty_score": 0.75,
                "status": "proposed",
            }
            for i in range(3)
        ]
        state.hypotheses.extend(hypotheses)
        return {"hypotheses_generated": len(hypotheses), "ids": [h["id"] for h in hypotheses]}
