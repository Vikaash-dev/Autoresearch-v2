"""Multi-agent runtime orchestrator for AutoResearch v2."""
from __future__ import annotations
import logging
from typing import Optional, TYPE_CHECKING
from .objective_graph import ObjectiveGraph, Objective, ObjectiveStatus
from .state import RunState
from .checkpoint import CheckpointManager

if TYPE_CHECKING:
    from ..config.schema import AutoResearchConfig
    from ..agents.base import BaseAgent

logger = logging.getLogger(__name__)


class AgentRuntime:
    """Orchestrates objective graph execution across multiple agents."""

    def __init__(self, config: "AutoResearchConfig", agents: dict[str, "BaseAgent"]) -> None:
        self.config = config
        self.agents = agents
        self.graph = ObjectiveGraph()
        self.state = RunState(run_id=config.run_id or "run", topic=config.topic)
        self.checkpoint_mgr = CheckpointManager(
            run_id=self.state.run_id,
            save_every=config.checkpoint.save_every_n_steps,
        )

    def add_objective(self, objective: Objective, agent_name: str) -> None:
        agent = self.agents[agent_name]
        self.graph.add(objective, agent.run)

    def run(self) -> RunState:
        """Execute all objectives respecting dependency order."""
        logger.info("Starting run: %s | topic: %s", self.state.run_id, self.state.topic)
        self.state.status = "running"

        while not self.graph.is_done():
            ready = self.graph.get_ready()
            if not ready:
                logger.warning("No ready objectives — possible cycle or all blocked")
                break
            for obj in ready:
                self.graph.mark_running(obj.id)
                logger.info("Running objective: %s", obj.name)
                handler = self.graph.get_handler(obj.id)
                try:
                    result = handler(obj, self.state, self.config)
                    self.graph.mark_completed(obj.id, result)
                    self.state.objectives_completed.append(obj.id)
                except Exception as exc:
                    logger.error("Objective %s failed: %s", obj.id, exc)
                    self.graph.mark_failed(obj.id, str(exc))
                    self.state.status = "failed"
                self.checkpoint_mgr.save(self.state)

        if self.state.status == "running":
            self.state.status = "completed"
        logger.info("Run finished: %s | status: %s", self.state.run_id, self.state.status)
        return self.state

    def resume(self) -> Optional[RunState]:
        """Resume from checkpoint if available."""
        saved = self.checkpoint_mgr.load_latest()
        if saved:
            self.state = saved
            self.state.status = "running"
            logger.info("Resumed run: %s (iteration %d)", self.state.run_id, self.state.iteration)
        return saved
