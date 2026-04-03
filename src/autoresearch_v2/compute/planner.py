from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class ComputePlanner:
    total_budget: float = 100.0

    def plan(self, weights: dict[str, float]) -> dict[str, float]:
        if not weights:
            return {}
        weight_sum = sum(max(0.0, w) for w in weights.values()) or 1.0
        return {k: self.total_budget * max(0.0, w) / weight_sum for k, w in weights.items()}

