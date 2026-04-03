"""Research Planner Agent."""
from __future__ import annotations
from typing import Any, TYPE_CHECKING
from .base import BaseAgent

if TYPE_CHECKING:
    from ..core.objective_graph import Objective
    from ..core.state import RunState
    from ..config.schema import AutoResearchConfig


class ResearchPlannerAgent(BaseAgent):
    name = "planner"
    role = "Research Planner"

    def run(self, objective: "Objective", state: "RunState", config: "AutoResearchConfig") -> Any:
        self.logger.info("Planning research for topic: %s", state.topic)
        plan = {
            "topic": state.topic,
            "objectives": [
                "mine_literature",
                "generate_hypotheses",
                "design_experiments",
                "run_experiments",
                "adversarial_review",
                "self_reflect",
            ],
            "rationale": self._llm_prompt(
                f"Create a detailed research plan for the topic: {state.topic}. "
                "Include key research questions, methodology, and expected outcomes."
            ),
            "constraints": {
                "max_iterations": config.max_iterations,
                "stop_criteria": config.stop_criteria,
            },
        }
        state.metadata["plan"] = plan
        self.logger.info("Research plan created with %d objectives", len(plan["objectives"]))
        return plan
