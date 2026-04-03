from __future__ import annotations

from typing import Any

from autoresearch_v2.agents.base_agent import BaseAgent
from autoresearch_v2.dog.objective import Objective


class CoderAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__(agent_id="coder_agent")

    def _run(self, objective: Objective, context: dict[str, Any]) -> dict[str, Any]:
        hypotheses = context.get("hypotheses", [])
        execution_log = [f"validated::{h}" for h in hypotheses]
        return {"execution_log": execution_log, "validated_count": len(execution_log)}

