from __future__ import annotations

from dataclasses import dataclass, field

from .objective import Objective, ObjectiveStatus


@dataclass(slots=True)
class DynamicObjectiveGraph:
    objectives: dict[str, Objective] = field(default_factory=dict)

    def add_objective(self, objective: Objective) -> None:
        if objective.id in self.objectives:
            raise ValueError(f"Objective already exists: {objective.id}")
        self.objectives[objective.id] = objective

    def get(self, objective_id: str) -> Objective:
        return self.objectives[objective_id]

    def ready_objectives(self) -> list[Objective]:
        ready: list[Objective] = []
        for obj in self.objectives.values():
            if obj.status != ObjectiveStatus.PENDING:
                continue
            if all(self.objectives[d].status == ObjectiveStatus.COMPLETE for d in obj.dependencies):
                ready.append(obj)
        return sorted(ready, key=lambda o: o.priority, reverse=True)

    def mark_complete(self, objective_id: str, outputs: dict | None = None) -> None:
        obj = self.get(objective_id)
        obj.status = ObjectiveStatus.COMPLETE
        if outputs:
            obj.outputs.update(outputs)

