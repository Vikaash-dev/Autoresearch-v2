from __future__ import annotations

from dataclasses import dataclass

from .telemetry import RunTelemetry


@dataclass(slots=True)
class AgentModification:
    agent_id: str
    change: str


class BoundedModifier:
    def suggest(self, agent_id: str, telemetry: RunTelemetry) -> AgentModification | None:
        failure = telemetry.failure_rates.get(agent_id, 0.0)
        if failure > 0.2:
            return AgentModification(agent_id=agent_id, change="tighten prompts and add retries")
        latency = telemetry.agent_latency.get(agent_id, 0.0)
        if latency > 10.0:
            return AgentModification(agent_id=agent_id, change="reduce context and tool calls")
        return None

