"""
Base agent class for all Autoresearch-v2 agents.

Agents:
  - Read from and write to the shared Blackboard before/after every action
  - Use Theory of Mind to model other agents' states before acting
  - Log self-observations for Self-ToM analysis
  - Emit skill-evolution triggers when they encounter repeated failures

Design influenced by:
  - NousResearch/hermes-agent (skill system, learning loop)
  - AIDE tree search agents
  - SakanaAI/AI-Scientist v2 agent manager
"""

from __future__ import annotations

import logging
import time
import uuid
from abc import ABC, abstractmethod
from typing import Any

from core.blackboard import Blackboard
from tom.engine import TheoryOfMindEngine

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """
    Abstract base for all research agents.

    Every agent has:
      - A unique ID and role label
      - Access to the shared Blackboard
      - A Theory-of-Mind engine for user/agent/self modeling
      - A skill store reference (populated by the SkillEvolver)
      - A failure counter per subtask for Self-ToM reporting
    """

    def __init__(
        self,
        role: str,
        blackboard: Blackboard,
        tom_engine: TheoryOfMindEngine,
        llm_fn: Any = None,             # callable(prompt: str) -> str
        agent_id: str | None = None,
    ) -> None:
        self.role = role
        self.agent_id = agent_id or f"{role}_{str(uuid.uuid4())[:8]}"
        self._bb = blackboard
        self._tom = tom_engine
        self._llm = llm_fn
        self._failure_counts: dict[str, int] = {}
        self._action_log: list[dict[str, Any]] = []

        # Register on the blackboard so other agents can model us
        self._bb.register_agent(self.agent_id, role)
        logger.debug("Agent registered: %s (%s)", self.agent_id, role)

    # ------------------------------------------------------------------ #
    #  Abstract interface                                                   #
    # ------------------------------------------------------------------ #

    @abstractmethod
    def act(self, context: dict[str, Any]) -> dict[str, Any]:
        """
        Primary action method. Called by the Orchestrator.

        Args:
            context: task-specific context dict (e.g. current node, idea, etc.)

        Returns:
            result dict with at minimum {"status": "ok"|"failed", "output": ...}
        """

    # ------------------------------------------------------------------ #
    #  Shared utilities                                                     #
    # ------------------------------------------------------------------ #

    def call_llm(self, prompt: str, system: str | None = None) -> str:
        """Wrapper around the LLM function with logging."""
        if self._llm is None:
            raise RuntimeError(f"Agent {self.agent_id} has no LLM configured")
        full_prompt = prompt if system is None else f"{system}\n\n{prompt}"
        start = time.time()
        try:
            response = self._llm(full_prompt)
            elapsed = time.time() - start
            logger.debug(
                "LLM call by %s: %.1fs, %d input chars, %d output chars",
                self.agent_id, elapsed, len(full_prompt), len(response)
            )
            return response
        except Exception as exc:
            logger.error("LLM call failed for %s: %s", self.agent_id, exc)
            raise

    def observe_other_agents(self) -> dict[str, Any]:
        """
        Agent-to-Agent ToM: read all other agents' belief states from Blackboard.
        Use this before acting to avoid duplicate work.
        """
        all_beliefs = self._bb.query_all_beliefs()
        summary = {}
        for aid, belief in all_beliefs.items():
            if aid != self.agent_id:
                summary[aid] = {
                    "role": belief.role,
                    "last_action": belief.last_action,
                    "known_failures": belief.known_failures[-3:],
                    "current_uncertainty": belief.current_uncertainty,
                }
        return summary

    def update_own_belief(self, **kwargs: Any) -> None:
        """Write self's current state to the Blackboard for others to read."""
        self._bb.update_agent_belief(self.agent_id, **kwargs)

    def record_failure(self, subtask: str, error: str) -> None:
        """Track failures for Self-ToM analysis and GEPA evolution trigger."""
        self._failure_counts[subtask] = self._failure_counts.get(subtask, 0) + 1
        self._tom.observe_self(
            action=f"{self.role}.{subtask}",
            outcome="failure",
        )
        self._bb.update_agent_belief(
            self.agent_id,
            last_action=f"FAILED:{subtask}",
            known_failures=list(self._bb.query_agent_belief(self.agent_id).known_failures) + [error[:80]],
        )
        # If same subtask fails repeatedly → trigger evolution signal
        if self._failure_counts.get(subtask, 0) >= 3:
            logger.warning(
                "Agent %s: subtask %r failed %d times — evolution trigger",
                self.agent_id, subtask, self._failure_counts[subtask]
            )
            self._bb.log_tom_observation(
                observer=self.agent_id,
                target="evolution_engine",
                observation=f"repeated_failure:{subtask}",
                inferred_state={"subtask": subtask, "failure_count": self._failure_counts[subtask]},
            )

    def record_success(self, subtask: str, metric: float | None = None) -> None:
        """Track successes for Self-ToM analysis."""
        self._tom.observe_self(
            action=f"{self.role}.{subtask}",
            outcome="success",
            metric=metric,
        )
        self._bb.update_agent_belief(
            self.agent_id,
            last_action=f"OK:{subtask}",
        )

    def get_recent_failures_from_board(self) -> list[str]:
        """Get the last few failures recorded by any agent — avoids repeating known mistakes."""
        all_nodes = self._bb.get_all_nodes()
        failed = [n.error_trace for n in all_nodes if n.status == "failed" and n.error_trace]
        return failed[-5:]
