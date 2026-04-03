"""Adversarial challenge generation for ToM review."""
from __future__ import annotations
import logging
from typing import Any, Dict, TYPE_CHECKING
from .profiles import ReviewerPersonaRegistry, ReviewerPersona

logger = logging.getLogger(__name__)


class AdversarialChallengeGenerator:
    """Generates adversarial challenges from reviewer personas."""

    def __init__(self, registry: ReviewerPersonaRegistry) -> None:
        self.registry = registry

    def generate(self, hypothesis: Dict[str, Any], persona: ReviewerPersona) -> Dict[str, Any]:
        """Generate adversarial challenge for a hypothesis from a persona's perspective."""
        title = hypothesis.get("title", "")
        triggers = persona.rejection_triggers

        challenges = []
        for trigger in triggers:
            challenges.append({
                "type": trigger,
                "challenge": f"From perspective of '{persona.label}': Does '{title}' address '{trigger}'?",
                "severity": "high" if trigger in persona.priorities else "medium",
            })

        return {
            "persona": persona.name,
            "hypothesis_title": title,
            "challenges": challenges,
            # Skepticism = complement of mean bias score across all dimensions.
            # Higher average bias (closer to 1) means the persona is more accepting,
            # so skepticism approaches 0; a neutral persona yields ~0.33 skepticism.
            "overall_skepticism": 1.0 - sum(persona.bias_profile.values()) / max(len(persona.bias_profile), 1),
            "recommended_actions": persona.acceptance_criteria,
        }

    def generate_minority_report(self, hypothesis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a minority report preserving dissenting views."""
        dissents = []
        for persona in self.registry.all():
            challenge = self.generate(hypothesis, persona)
            if challenge["overall_skepticism"] > 0.3:
                dissents.append({
                    "persona": persona.name,
                    "dissent": challenge["challenges"][0] if challenge["challenges"] else {},
                })
        return {"hypothesis_id": hypothesis.get("id"), "minority_reports": dissents}
