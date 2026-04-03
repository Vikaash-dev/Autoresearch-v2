"""Theory-of-Mind (ToM) Modeling Agent."""
from __future__ import annotations
from typing import Any, TYPE_CHECKING
from .base import BaseAgent

if TYPE_CHECKING:
    from ..core.objective_graph import Objective
    from ..core.state import RunState
    from ..config.schema import AutoResearchConfig


class TomAgent(BaseAgent):
    name = "tom_agent"
    role = "Theory-of-Mind Modeling"

    def run(self, objective: "Objective", state: "RunState", config: "AutoResearchConfig") -> Any:
        from ..tom.negotiation import EpistemicNegotiationLoop
        from ..tom.intent import CollaboratorIntentProfile

        self.logger.info("Running ToM epistemic negotiation")
        intent_profile = CollaboratorIntentProfile.from_config(config)
        loop = EpistemicNegotiationLoop(intent_profile, config)
        negotiation_result = loop.run(state)
        state.metadata["tom_negotiation"] = negotiation_result
        return negotiation_result
