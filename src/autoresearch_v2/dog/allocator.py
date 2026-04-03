from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class ComputeBudgetAllocator:
    total_budget: float

    def allocate(self, objective_ids: list[str]) -> dict[str, float]:
        if not objective_ids:
            return {}
        per_objective = self.total_budget / len(objective_ids)
        return {objective_id: per_objective for objective_id in objective_ids}

