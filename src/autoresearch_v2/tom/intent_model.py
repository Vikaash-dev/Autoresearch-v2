from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class CollaboratorIntentModel:
    risk_appetite: float = 0.5
    rigor_vs_speed: float = 0.5
    venue_bias: float = 0.5
    collaboration_style: float = 0.5

    def adjust_budgets(self, budgets: dict[str, float]) -> dict[str, float]:
        adjusted = dict(budgets)
        if "verification" in adjusted:
            adjusted["verification"] *= 1.0 + 0.5 * self.rigor_vs_speed
        if "exploration" in adjusted:
            adjusted["exploration"] *= 1.0 + 0.4 * self.risk_appetite
        return adjusted

