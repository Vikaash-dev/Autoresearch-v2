"""Meta-Agent: self-improvement logic."""
from __future__ import annotations
from typing import Any, TYPE_CHECKING
from .base import BaseAgent

if TYPE_CHECKING:
    from ..core.objective_graph import Objective
    from ..core.state import RunState
    from ..config.schema import AutoResearchConfig


class MetaAgent(BaseAgent):
    name = "meta_agent"
    role = "Meta-Agent (Self-Improvement)"

    def run(self, objective: "Objective", state: "RunState", config: "AutoResearchConfig") -> Any:
        from ..reflection.self_review import SelfReviewModule

        self.logger.info("Running meta-level self-improvement analysis")
        reviewer = SelfReviewModule(config)
        reflection = reviewer.analyze(state)
        state.reflections.append(reflection)

        if config.reflection.safe_modification_only:
            policy_updates = reflection.get("proposed_policy_updates", [])
            applied = []
            for update in policy_updates[: config.reflection.max_policy_updates_per_run]:
                if update.get("safe", True):
                    state.policy_updates.append(update)
                    applied.append(update)
            self.logger.info("Applied %d safe policy updates", len(applied))
            return {"reflection": reflection, "policy_updates_applied": len(applied)}
        return {"reflection": reflection}
