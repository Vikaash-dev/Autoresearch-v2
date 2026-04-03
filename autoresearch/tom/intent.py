"""Collaborator intent profile for ToM layer."""
from __future__ import annotations
from typing import Dict, Any, TYPE_CHECKING
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from ..config.schema import AutoResearchConfig


class CollaboratorIntentProfile(BaseModel):
    """Models the user/collaborator's research intent and preferences."""

    risk_appetite: str = "moderate"  # conservative | moderate | aggressive
    primary_goal: str = "publishability"  # publishability | truth_seeking | speed | rigor
    novelty_weight: float = 0.7
    rigor_weight: float = 0.8
    speed_weight: float = 0.5
    interpretability_required: bool = True
    target_venues: list[str] = Field(default_factory=lambda: ["NeurIPS", "ICLR", "ICML"])
    constraints: Dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_config(cls, config: "AutoResearchConfig") -> "CollaboratorIntentProfile":
        """Infer intent profile from run configuration."""
        stop = config.stop_criteria
        return cls(
            primary_goal="publishability",
            novelty_weight=0.7,
            rigor_weight=stop.get("min_confidence", 0.7),
        )

    def alignment_score(self, result: Dict[str, Any]) -> float:
        """Compute how well a result aligns with stated intent."""
        novelty = result.get("novelty_score", 0.5)
        rigor = result.get("rigor_score", 0.5)
        return self.novelty_weight * novelty + self.rigor_weight * rigor
