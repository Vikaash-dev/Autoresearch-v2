from __future__ import annotations

from dataclasses import dataclass

from autoresearch_v2.dog.graph import DynamicObjectiveGraph
from autoresearch_v2.dog.objective import Objective, ObjectiveStatus


@dataclass(slots=True)
class ScheduledObjective:
    objective_id: str
    agent_id: str


class ObjectiveScheduler:
    def __init__(self, graph: DynamicObjectiveGraph) -> None:
        self.graph = graph

    def schedule(self, available_agents: list[str], limit: int | None = None) -> list[ScheduledObjective]:
        if not available_agents:
            return []
        ready = self.graph.ready_objectives()
        if limit is not None:
            ready = ready[:limit]
        assignments: list[ScheduledObjective] = []
        for index, objective in enumerate(ready):
            agent_id = objective.assigned_agent or available_agents[index % len(available_agents)]
            objective.status = ObjectiveStatus.IN_PROGRESS
            objective.assigned_agent = agent_id
            assignments.append(ScheduledObjective(objective_id=objective.id, agent_id=agent_id))
        return assignments

    def mark_failed(self, objective_id: str) -> None:
        obj: Objective = self.graph.get(objective_id)
        obj.status = ObjectiveStatus.FAILED

