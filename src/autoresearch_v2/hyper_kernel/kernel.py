from __future__ import annotations

from dataclasses import dataclass

from .modifier import AgentModification, BoundedModifier
from .telemetry import RunTelemetry


@dataclass(slots=True)
class HyperKernel:
    modifier: BoundedModifier

    def optimize_agent(self, agent_id: str, telemetry: RunTelemetry) -> AgentModification | None:
        return self.modifier.suggest(agent_id=agent_id, telemetry=telemetry)

    def apply_modification(self, modification: AgentModification | None) -> bool:
        return modification is not None

