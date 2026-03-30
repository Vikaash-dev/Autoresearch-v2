"""
Base self-evolving agent.

Implements the DGM / Bilevel concept of an agent that:
  1. Maintains an internal parameter vector (soft config).
  2. Tracks its own performance history.
  3. Updates itself via gradient-free parameter evolution between rounds.
  4. Exposes a uniform ``run(task)`` interface used by the orchestrator.
"""

from __future__ import annotations

import abc
import copy
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class AgentStatus(Enum):
    IDLE = auto()
    RUNNING = auto()
    SUCCEEDED = auto()
    FAILED = auto()
    EVOLVED = auto()


@dataclass
class AgentState:
    """Snapshot of an agent's internal state — stored in the experience memory."""

    agent_id: str
    agent_class: str
    round_number: int
    task: str
    status: AgentStatus
    result: Optional[Any]
    score: float                        # 0.0 – 1.0 quality score
    elapsed_seconds: float
    parameters: Dict[str, Any]          # current soft-config snapshot
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


class BaseAgent(abc.ABC):
    """
    Self-evolving base agent (DGM-Hyperagents / Bilevel Autoresearch style).

    Subclasses implement ``_execute(task, context)`` and optionally
    ``_score_result(result)`` for domain-specific quality measurement.

    The base class handles:
      - Parameter tracking & soft self-evolution
      - Round/history bookkeeping
      - Structured state snapshots for the experience store
      - Failure recycling (Bilevel: failed runs become training signal)
    """

    def __init__(
        self,
        agent_id: Optional[str] = None,
        name: Optional[str] = None,
        config: Optional[Any] = None,
    ) -> None:
        self.agent_id: str = agent_id or str(uuid.uuid4())
        self.name: str = name or self.__class__.__name__
        self.config = config
        self.status: AgentStatus = AgentStatus.IDLE
        self._round: int = 0
        self._history: List[AgentState] = []
        # Soft parameter vector — agents mutate this during self-evolution
        self._parameters: Dict[str, Any] = self._default_parameters()
        logger.debug("Agent %s (%s) initialised.", self.agent_id, self.name)

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def _execute(self, task: str, context: Dict[str, Any]) -> Any:
        """Core logic — implemented by each concrete agent."""

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(self, task: str, context: Optional[Dict[str, Any]] = None) -> AgentState:
        """Execute the agent for one round and return a state snapshot."""
        context = context or {}
        self._round += 1
        self.status = AgentStatus.RUNNING
        start = time.time()
        result: Any = None
        score: float = 0.0
        try:
            result = self._execute(task, context)
            score = self._score_result(result)
            self.status = AgentStatus.SUCCEEDED
            logger.info("[%s] round %d succeeded (score=%.3f).", self.name, self._round, score)
        except Exception as exc:  # noqa: BLE001
            score = 0.0
            self.status = AgentStatus.FAILED
            result = {"error": str(exc)}
            logger.warning("[%s] round %d FAILED: %s", self.name, self._round, exc)

        elapsed = time.time() - start
        state = AgentState(
            agent_id=self.agent_id,
            agent_class=self.__class__.__name__,
            round_number=self._round,
            task=task,
            status=self.status,
            result=result,
            score=score,
            elapsed_seconds=elapsed,
            parameters=copy.deepcopy(self._parameters),
        )
        self._history.append(state)
        return state

    def evolve(self, feedback: Optional[Dict[str, Any]] = None) -> None:
        """
        Self-evolution step (DGM / Bilevel Autoresearch).

        Uses the history of previous rounds to nudge internal parameters
        toward higher-scoring configurations without gradient information.
        Concrete agents may override this for domain-specific evolution.
        """
        if not self._history:
            return

        feedback = feedback or {}
        best_state = max(self._history, key=lambda s: s.score)
        lr = (
            self.config.agent.evolution_lr
            if self.config is not None
            else 0.1
        )

        # Simple parameter perturbation toward best known configuration
        for key in self._parameters:
            best_val = best_state.parameters.get(key)
            cur_val = self._parameters[key]
            if isinstance(cur_val, float) and isinstance(best_val, float):
                self._parameters[key] = cur_val + lr * (best_val - cur_val)
            elif best_val is not None and best_val != cur_val:
                # Non-numeric: copy best value with probability proportional to lr
                import random  # local import to avoid module-level side-effects
                if random.random() < lr:
                    self._parameters[key] = best_val

        # Incorporate external feedback
        for key, val in feedback.items():
            if key in self._parameters:
                self._parameters[key] = val

        self.status = AgentStatus.EVOLVED
        logger.info("[%s] evolved parameters after %d rounds.", self.name, self._round)

    # ------------------------------------------------------------------
    # Optional hooks
    # ------------------------------------------------------------------

    def _score_result(self, result: Any) -> float:  # noqa: ARG002
        """Return a quality score in [0, 1].  Override for domain scoring."""
        return 1.0 if result is not None else 0.0

    def _default_parameters(self) -> Dict[str, Any]:
        """Initial soft parameter vector.  Override to add agent-specific params."""
        return {}

    # ------------------------------------------------------------------
    # Introspection helpers
    # ------------------------------------------------------------------

    @property
    def round(self) -> int:
        return self._round

    @property
    def history(self) -> List[AgentState]:
        return list(self._history)

    @property
    def best_score(self) -> float:
        if not self._history:
            return 0.0
        return max(s.score for s in self._history)

    @property
    def average_score(self) -> float:
        if not self._history:
            return 0.0
        return sum(s.score for s in self._history) / len(self._history)

    @property
    def failure_rate(self) -> float:
        if not self._history:
            return 0.0
        failed = sum(1 for s in self._history if s.status == AgentStatus.FAILED)
        return failed / len(self._history)

    def summary(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "rounds": self._round,
            "best_score": self.best_score,
            "average_score": self.average_score,
            "failure_rate": self.failure_rate,
            "status": self.status.name,
            "parameters": copy.deepcopy(self._parameters),
        }
