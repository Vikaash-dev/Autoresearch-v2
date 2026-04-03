"""Executor/Debugger Agent."""
from __future__ import annotations
from typing import Any, TYPE_CHECKING
from .base import BaseAgent

if TYPE_CHECKING:
    from ..core.objective_graph import Objective
    from ..core.state import RunState
    from ..config.schema import AutoResearchConfig


class ExecutorAgent(BaseAgent):
    name = "executor"
    role = "Execution/Debugger"

    def run(self, objective: "Objective", state: "RunState", config: "AutoResearchConfig") -> Any:
        from ..experiments.sandbox import SandboxExecutor

        self.logger.info("Executing experiments")
        executor = SandboxExecutor(config)
        results = []
        for exp in state.experiments:
            result = executor.execute(exp)
            results.append(result)
            # Update experiment status in state
            for e in state.experiments:
                if e.get("id") == exp.get("id"):
                    e["result"] = result
                    e["status"] = result.get("status", "unknown")
        return {"experiments_run": len(results), "results": results}
