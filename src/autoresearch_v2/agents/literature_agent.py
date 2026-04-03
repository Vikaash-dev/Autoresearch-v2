from __future__ import annotations

from typing import Any

from autoresearch_v2.agents.base_agent import BaseAgent
from autoresearch_v2.dog.objective import Objective


class LiteratureAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__(agent_id="literature_agent")

    def _run(self, objective: Objective, context: dict[str, Any]) -> dict[str, Any]:
        topic = context.get("topic", objective.description)
        candidates = context.get("literature_candidates", [])
        return {
            "topic": topic,
            "citations": candidates[:10],
            "summary": f"Collected {len(candidates[:10])} references for {topic}",
        }

