"""
Base self-evolving agent — SERA Architecture.

Implements the DGM / Bilevel concept of an agent that:
  1. Maintains an internal parameter vector (soft config).
  2. Tracks its own performance history (bounded ring buffer).
  3. Updates itself via gradient-free parameter evolution between rounds.
  4. Recycles failures as training signal (Bilevel Autoresearch).
  5. Exposes a uniform ``run(task)`` interface used by the orchestrator.
"""

from __future__ import annotations

import abc
import copy
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Maximum history entries kept in memory per agent (ring-buffer limit)
_DEFAULT_MAX_HISTORY = 200


class AgentStatus(Enum):
    IDLE = auto()
    RUNNING = auto()
    SUCCEEDED = auto()
    FAILED = auto()
    EVOLVED = auto()
    TIMEOUT = auto()


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
      - Bounded round/history bookkeeping (ring buffer)
      - Structured state snapshots for the experience store
      - Failure recycling (Bilevel: failed runs become training signal)
      - Execution timeout (prevents hanging experiments)
    """

    def __init__(
        self,
        agent_id: Optional[str] = None,
        name: Optional[str] = None,
        config: Optional[Any] = None,
        max_history: int = _DEFAULT_MAX_HISTORY,
        timeout_seconds: Optional[float] = None,
    ) -> None:
        self.agent_id: str = agent_id or str(uuid.uuid4())
        self.name: str = name or self.__class__.__name__
        self.config = config
        self.max_history = max_history
        self.timeout_seconds = timeout_seconds
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
        final_status = AgentStatus.SUCCEEDED

        try:
            if self.timeout_seconds is not None:
                result = self._run_with_timeout(task, context, self.timeout_seconds)
                if result is _TIMEOUT_SENTINEL:
                    final_status = AgentStatus.TIMEOUT
                    result = {"error": "execution timed out"}
                    score = 0.0
                else:
                    score = self._score_result(result)
                    final_status = AgentStatus.SUCCEEDED
            else:
                result = self._execute(task, context)
                score = self._score_result(result)
                final_status = AgentStatus.SUCCEEDED

            logger.info("[%s] round %d %s (score=%.3f).", self.name, self._round, final_status.name, score)
        except Exception as exc:  # noqa: BLE001
            score = 0.0
            final_status = AgentStatus.FAILED
            result = {"error": str(exc), "type": type(exc).__name__}
            logger.warning("[%s] round %d FAILED (%s): %s", self.name, self._round, type(exc).__name__, exc)

        self.status = final_status
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
        self._append_history(state)
        return state

    def _run_with_timeout(self, task: str, context: Dict[str, Any], timeout: float) -> Any:
        """Run _execute in a thread with a hard timeout."""
        result_container: List[Any] = [_TIMEOUT_SENTINEL]
        exc_container: List[Optional[Exception]] = [None]

        def target() -> None:
            try:
                result_container[0] = self._execute(task, context)
            except Exception as e:  # noqa: BLE001
                exc_container[0] = e

        thread = threading.Thread(target=target, daemon=True)
        thread.start()
        thread.join(timeout=timeout)
        if thread.is_alive():
            return _TIMEOUT_SENTINEL
        if exc_container[0] is not None:
            raise exc_container[0]
        return result_container[0]

    def evolve(self, feedback: Optional[Dict[str, Any]] = None) -> None:
        """
        Self-evolution step (DGM / Bilevel Autoresearch).

        Uses the history of previous rounds to nudge internal parameters
        toward higher-scoring configurations without gradient information.
        Failed runs are recycled as signal (Bilevel: identify bottlenecks).
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

        # Collect failed states as negative signal (Bilevel failure recycling)
        failed_states = [s for s in self._history if s.status == AgentStatus.FAILED]
        failed_params: Dict[str, Any] = {}
        if failed_states:
            # Average failed parameter values for avoidance
            for key in self._parameters:
                vals = [
                    s.parameters[key]
                    for s in failed_states
                    if key in s.parameters and isinstance(s.parameters[key], float)
                ]
                if vals:
                    failed_params[key] = sum(vals) / len(vals)

        # Move toward best, away from failure centroid
        import random  # local import to keep module import side-effect free
        rng = random.Random()
        for key in self._parameters:
            best_val = best_state.parameters.get(key)
            cur_val = self._parameters[key]
            if isinstance(cur_val, float) and isinstance(best_val, float):
                step = lr * (best_val - cur_val)
                # Repulsion from failure centroid
                if key in failed_params:
                    step -= 0.5 * lr * (failed_params[key] - cur_val)
                self._parameters[key] = cur_val + step
            elif best_val is not None and best_val != cur_val:
                if rng.random() < lr:
                    self._parameters[key] = best_val

        # Incorporate external feedback (e.g. from MetaAgent)
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
    # History helpers
    # ------------------------------------------------------------------

    def _append_history(self, state: AgentState) -> None:
        """Append to bounded ring-buffer history."""
        self._history.append(state)
        if len(self._history) > self.max_history:
            self._history = self._history[-self.max_history :]

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
        failed = sum(
            1 for s in self._history
            if s.status in (AgentStatus.FAILED, AgentStatus.TIMEOUT)
        )
        return failed / len(self._history)

    def recyclable_failures(self) -> List[AgentState]:
        """
        Return failed states whose score is above the recycling threshold.
        Used by the Bilevel outer loop to mine signal from failures.
        """
        threshold = (
            self.config.memory.recycling_threshold
            if self.config is not None
            else 0.3
        )
        return [
            s for s in self._history
            if s.status == AgentStatus.FAILED and s.score >= threshold
        ]

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
            "history_size": len(self._history),
        }


# Sentinel returned when execution times out
class _TimeoutSentinel:
    pass


_TIMEOUT_SENTINEL = _TimeoutSentinel()
