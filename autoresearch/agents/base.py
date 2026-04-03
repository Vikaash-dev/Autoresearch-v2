"""Base agent class for AutoResearch v2."""
from __future__ import annotations
import logging
from abc import ABC, abstractmethod
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from ..core.objective_graph import Objective
    from ..core.state import RunState
    from ..config.schema import AutoResearchConfig

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """Abstract base class for all AutoResearch agents."""

    name: str = "base"
    role: str = "generic"

    def __init__(self, config: "AutoResearchConfig") -> None:
        self.config = config
        self.logger = logging.getLogger(f"autoresearch.agents.{self.name}")

    @abstractmethod
    def run(self, objective: "Objective", state: "RunState", config: "AutoResearchConfig") -> Any:
        """Execute the agent's task for the given objective."""

    def _llm_prompt(self, prompt: str) -> str:
        """Placeholder LLM call — replace with real provider integration."""
        self.logger.debug("LLM prompt (%d chars): %s...", len(prompt), prompt[:100])
        # In production, call self.config.llm.provider API here
        return f"[LLM response to: {prompt[:80]}...]"
