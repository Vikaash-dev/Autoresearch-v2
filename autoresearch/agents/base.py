"""Base agent class for AutoResearch v2."""
from __future__ import annotations
import logging
from abc import ABC, abstractmethod
from typing import Any, TYPE_CHECKING
from ..llm import LLMClient

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
        self.llm_client = LLMClient(config.llm)

    @abstractmethod
    def run(self, objective: "Objective", state: "RunState", config: "AutoResearchConfig") -> Any:
        """Execute the agent's task for the given objective."""

    def _llm_prompt(self, prompt: str) -> str:
        """LLM call via configured provider with safe fallback."""
        self.logger.debug("LLM prompt (%d chars): %s...", len(prompt), prompt[:100])
        return self.llm_client.prompt(prompt)
