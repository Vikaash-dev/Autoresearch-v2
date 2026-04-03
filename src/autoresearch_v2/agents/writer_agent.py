from __future__ import annotations

from typing import Any

from autoresearch_v2.agents.base_agent import BaseAgent
from autoresearch_v2.dog.objective import Objective


class WriterAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__(agent_id="writer_agent")

    def _run(self, objective: Objective, context: dict[str, Any]) -> dict[str, Any]:
        topic = context.get("topic", "unknown-topic")
        hypotheses = context.get("hypotheses", [])
        citations = context.get("citations", [])
        manuscript = (
            f"# Draft Manuscript\n\nTopic: {topic}\n\n"
            f"Hypotheses explored: {len(hypotheses)}\n"
            f"Citations used: {len(citations)}\n"
        )
        return {"manuscript": manuscript}

