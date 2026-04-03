"""Reviewer/Adversary Agent."""
from __future__ import annotations
from typing import Any, TYPE_CHECKING
from .base import BaseAgent

if TYPE_CHECKING:
    from ..core.objective_graph import Objective
    from ..core.state import RunState
    from ..config.schema import AutoResearchConfig


class ReviewerAgent(BaseAgent):
    name = "reviewer"
    role = "Reviewer/Adversary"

    def run(self, objective: "Objective", state: "RunState", config: "AutoResearchConfig") -> Any:
        from ..tom.adversarial import AdversarialChallengeGenerator
        from ..tom.profiles import ReviewerPersonaRegistry

        self.logger.info("Running adversarial review")
        registry = ReviewerPersonaRegistry()
        challenger = AdversarialChallengeGenerator(registry)

        reviews = []
        for persona_name in config.tom.reviewer_personas:
            persona = registry.get(persona_name)
            for hyp in state.hypotheses:
                challenge = challenger.generate(hyp, persona)
                reviews.append({
                    "persona": persona_name,
                    "hypothesis_id": hyp.get("id"),
                    "challenge": challenge,
                    "score": 0.65,
                })
        state.reviews.extend(reviews)
        self.logger.info("Generated %d reviews", len(reviews))
        return {"reviews_generated": len(reviews)}
