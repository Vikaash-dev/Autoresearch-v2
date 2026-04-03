from __future__ import annotations

from dataclasses import dataclass, field
from time import time
from typing import Any

from autoresearch_v2.dog.objective import Objective


@dataclass(slots=True)
class AgentTelemetry:
    agent_id: str
    executions: int = 0
    total_latency_sec: float = 0.0
    failures: int = 0
    notes: list[str] = field(default_factory=list)

    @property
    def avg_latency_sec(self) -> float:
        if self.executions == 0:
            return 0.0
        return self.total_latency_sec / self.executions


@dataclass(slots=True)
class ObjectiveResult:
    objective_id: str
    success: bool
    outputs: dict[str, Any]
    notes: list[str] = field(default_factory=list)


class BaseAgent:
    def __init__(self, agent_id: str) -> None:
        self.agent_id = agent_id
        self._telemetry = AgentTelemetry(agent_id=agent_id)

    def execute(self, objective: Objective, context: dict[str, Any]) -> ObjectiveResult:
        start = time()
        try:
            outputs = self._run(objective, context)
            result = ObjectiveResult(objective_id=objective.id, success=True, outputs=outputs)
            return result
        except Exception as exc:  # pragma: no cover - defensive path
            self._telemetry.failures += 1
            self._telemetry.notes.append(str(exc))
            return ObjectiveResult(objective_id=objective.id, success=False, outputs={}, notes=[str(exc)])
        finally:
            self._telemetry.executions += 1
            self._telemetry.total_latency_sec += max(0.0, time() - start)

    def _run(self, objective: Objective, context: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    def get_telemetry(self) -> AgentTelemetry:
        return self._telemetry

