"""
Shared research blackboard — the single source of truth for all agents.

Every agent reads from and writes to the blackboard before/after acting.
This prevents duplicate work, circular dependencies, and enables Theory of Mind
between agents (each agent can query what other agents know/have tried/believe).

Design influenced by:
  - NousResearch/hermes-agent  (skill + memory store)
  - plastic-labs/honcho        (dialectic user/agent modeling)
  - SakanaAI/AI-Scientist v2   (unified_tree_viz state)
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any


@dataclass
class AgentBelief:
    """A single agent's current mental state — what it knows, tried, and believes."""

    agent_id: str
    role: str  # hypothesis | experiment | literature | writer | reviewer | orchestrator
    last_action: str = ""
    known_failures: list[str] = field(default_factory=list)
    known_successes: list[str] = field(default_factory=list)
    current_uncertainty: float = 1.0  # 0=certain, 1=fully uncertain
    confidence_domain: str = ""  # area where this agent has highest confidence
    last_updated: float = field(default_factory=time.time)


@dataclass
class ExperimentNode:
    """A single node in the Best-First Tree Search experiment graph."""

    node_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    parent_id: str | None = None
    depth: int = 0
    hypothesis: str = ""
    code_patch: str = ""
    metric: float | None = None  # lower or higher depending on objective
    status: str = "pending"  # pending | running | success | failed | pruned
    error_trace: str = ""
    stage: int = 1  # 1=baseline 2=ablation 3=full 4=writeup
    created_at: float = field(default_factory=time.time)
    children: list[str] = field(default_factory=list)


@dataclass
class ResearchState:
    """Top-level shared state of the entire research run."""

    run_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    idea: dict[str, Any] = field(default_factory=dict)
    best_metric: float | None = None
    best_node_id: str | None = None
    total_nodes_explored: int = 0
    total_tokens_used: int = 0
    stage: int = 1
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)


class Blackboard:
    """
    Thread-safe shared state store for all agents in a research run.

    Inspired by:
      - Classic blackboard AI architecture
      - karpathy/autoresearch program.md as the shared context
      - NousResearch/hermes-agent's FTS5 session memory
    """

    def __init__(self, persist_path: Path | None = None) -> None:
        self._lock = threading.RLock()
        self._state = ResearchState()
        self._nodes: dict[str, ExperimentNode] = {}
        self._agent_beliefs: dict[str, AgentBelief] = {}
        self._tom_observations: list[dict[str, Any]] = []  # ToM trace log
        self._persist_path = persist_path

    # ------------------------------------------------------------------ #
    #  Research State                                                       #
    # ------------------------------------------------------------------ #

    def init_run(self, idea: dict[str, Any]) -> str:
        with self._lock:
            self._state = ResearchState(idea=idea)
            self._nodes.clear()
            self._agent_beliefs.clear()
            self._tom_observations.clear()
            self._flush()
            return self._state.run_id

    @property
    def run_id(self) -> str:
        return self._state.run_id

    @property
    def state(self) -> ResearchState:
        return self._state

    def update_best(self, node_id: str, metric: float) -> None:
        with self._lock:
            self._state.best_metric = metric
            self._state.best_node_id = node_id
            self._state.updated_at = time.time()
            self._flush()

    # ------------------------------------------------------------------ #
    #  Experiment Nodes (BFTS tree)                                        #
    # ------------------------------------------------------------------ #

    def add_node(self, node: ExperimentNode) -> None:
        with self._lock:
            self._nodes[node.node_id] = node
            if node.parent_id and node.parent_id in self._nodes:
                self._nodes[node.parent_id].children.append(node.node_id)
            self._state.total_nodes_explored += 1
            self._state.updated_at = time.time()
            self._flush()

    def update_node(self, node_id: str, **kwargs: Any) -> None:
        with self._lock:
            if node_id not in self._nodes:
                raise KeyError(f"Node {node_id!r} not found on blackboard")
            for k, v in kwargs.items():
                setattr(self._nodes[node_id], k, v)
            self._flush()

    def get_node(self, node_id: str) -> ExperimentNode | None:
        with self._lock:
            return self._nodes.get(node_id)

    def get_all_nodes(self) -> list[ExperimentNode]:
        with self._lock:
            return list(self._nodes.values())

    def get_best_nodes(self, n: int = 5) -> list[ExperimentNode]:
        """Return top-n successful nodes by metric (higher = better by default)."""
        with self._lock:
            successful = [
                nd for nd in self._nodes.values()
                if nd.status == "success" and nd.metric is not None
            ]
            return sorted(successful, key=lambda nd: nd.metric, reverse=True)[:n]

    def get_frontier_nodes(self) -> list[ExperimentNode]:
        """Return nodes eligible for expansion (pending, no children yet)."""
        with self._lock:
            return [
                nd for nd in self._nodes.values()
                if nd.status == "success" and not nd.children
            ]

    # ------------------------------------------------------------------ #
    #  Agent Beliefs (Agent-to-Agent ToM)                                  #
    # ------------------------------------------------------------------ #

    def register_agent(self, agent_id: str, role: str) -> None:
        with self._lock:
            self._agent_beliefs[agent_id] = AgentBelief(
                agent_id=agent_id, role=role
            )

    def update_agent_belief(self, agent_id: str, **kwargs: Any) -> None:
        with self._lock:
            if agent_id not in self._agent_beliefs:
                raise KeyError(f"Agent {agent_id!r} not registered on blackboard")
            b = self._agent_beliefs[agent_id]
            for k, v in kwargs.items():
                setattr(b, k, v)
            b.last_updated = time.time()
            self._flush()

    def query_agent_belief(self, agent_id: str) -> AgentBelief | None:
        with self._lock:
            return self._agent_beliefs.get(agent_id)

    def query_all_beliefs(self) -> dict[str, AgentBelief]:
        with self._lock:
            return dict(self._agent_beliefs)

    # ------------------------------------------------------------------ #
    #  ToM Observation Log                                                 #
    # ------------------------------------------------------------------ #

    def log_tom_observation(
        self,
        observer: str,
        target: str,
        observation: str,
        inferred_state: dict[str, Any],
    ) -> None:
        """Record a Theory-of-Mind inference made by one agent about another."""
        with self._lock:
            self._tom_observations.append(
                {
                    "observer": observer,
                    "target": target,
                    "observation": observation,
                    "inferred_state": inferred_state,
                    "timestamp": time.time(),
                }
            )
            self._flush()

    def get_tom_observations(
        self, observer: str | None = None, target: str | None = None
    ) -> list[dict[str, Any]]:
        with self._lock:
            obs = self._tom_observations
            if observer:
                obs = [o for o in obs if o["observer"] == observer]
            if target:
                obs = [o for o in obs if o["target"] == target]
            return obs

    # ------------------------------------------------------------------ #
    #  Serialisation                                                        #
    # ------------------------------------------------------------------ #

    def to_dict(self) -> dict[str, Any]:
        with self._lock:
            return {
                "state": asdict(self._state),
                "nodes": {k: asdict(v) for k, v in self._nodes.items()},
                "agent_beliefs": {k: asdict(v) for k, v in self._agent_beliefs.items()},
                "tom_observations": self._tom_observations,
            }

    def _flush(self) -> None:
        """Persist to disk if a path was provided (called under lock)."""
        if self._persist_path:
            self._persist_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._persist_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self.to_dict(), indent=2))
            tmp.replace(self._persist_path)

    @classmethod
    def load(cls, path: Path) -> "Blackboard":
        bb = cls(persist_path=path)
        if path.exists():
            data = json.loads(path.read_text())
            bb._state = ResearchState(**data["state"])
            bb._nodes = {
                k: ExperimentNode(**v) for k, v in data["nodes"].items()
            }
            bb._agent_beliefs = {
                k: AgentBelief(**v) for k, v in data["agent_beliefs"].items()
            }
            bb._tom_observations = data.get("tom_observations", [])
        return bb
