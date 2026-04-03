from __future__ import annotations

from typing import Any

from autoresearch_v2.agents.base_agent import BaseAgent
from autoresearch_v2.dog.objective import Objective


class CriticAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__(agent_id="critic_agent")

    def _run(self, objective: Objective, context: dict[str, Any]) -> dict[str, Any]:
        manuscript = context.get("manuscript", "")
        citations = context.get("citations", [])
        score = min(1.0, 0.4 + 0.1 * min(5, len(citations)) + (0.2 if manuscript else 0.0))
        return {"critic_score": round(score, 3)}

