"""Experiment Coder Agent."""
from __future__ import annotations
from typing import Any, TYPE_CHECKING
from .base import BaseAgent

if TYPE_CHECKING:
    from ..core.objective_graph import Objective
    from ..core.state import RunState
    from ..config.schema import AutoResearchConfig


class ExperimentCoderAgent(BaseAgent):
    name = "experiment_coder"
    role = "Experiment Coder"

    def run(self, objective: "Objective", state: "RunState", config: "AutoResearchConfig") -> Any:
        from ..experiments.spec import ExperimentSpec
        from ..experiments.generator import ExperimentGenerator

        self.logger.info("Generating experiment code for hypotheses")
        generator = ExperimentGenerator(config)
        specs = []
        for hyp in state.hypotheses[:2]:  # top 2 hypotheses
            spec = generator.generate(hyp, state.topic)
            specs.append(spec.model_dump())
            state.experiments.append(spec.model_dump())
        return {"specs_generated": len(specs)}
