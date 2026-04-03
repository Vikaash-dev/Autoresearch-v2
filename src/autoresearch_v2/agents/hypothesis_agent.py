from __future__ import annotations

from typing import Any

from autoresearch_v2.agents.base_agent import BaseAgent
from autoresearch_v2.dog.objective import Objective


class HypothesisAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__(agent_id="hypothesis_agent")

    def _run(self, objective: Objective, context: dict[str, Any]) -> dict[str, Any]:
        topic = context.get("topic", "unknown-topic")
        max_hypotheses = max(1, int(context.get("max_hypotheses", 3)))
        hypotheses = [f"{topic}: hypothesis {i+1}" for i in range(max_hypotheses)]
        return {"hypotheses": hypotheses}

