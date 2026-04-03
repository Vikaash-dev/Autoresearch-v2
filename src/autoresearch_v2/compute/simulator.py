from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class ComputeSimulator:
    def estimate_duration(self, objective_count: int, avg_seconds_per_objective: float = 2.0) -> float:
        return max(0.0, objective_count * avg_seconds_per_objective)

