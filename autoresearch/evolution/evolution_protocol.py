"""
Evolution Protocol — Autogenesis RSPL + SEPL implemented for SERA-X.

From SkyworkAI/Autogenesis:
  RSPL (Resource Substrate Protocol Layer):
    - Prompts, agents, tools, environments, and memory are versioned resources
    - Every resource has explicit state, lifecycle, and versioned interfaces

  SEPL (Self Evolution Protocol Layer):
    - Closed-loop: Propose → Assess → Commit (with rollback)
    - Every evolution step has auditable lineage

From HyperAgents (Meta):
    - MetaAgent proposes diffs to the codebase
    - Archive stores every version with scores
    - Parent selection picks the best ancestor to evolve from

From Karpathy/autoresearch:
    - program.md is the human-editable org charter
    - strategy.md is the live evolving strategy
    - The agent edits strategy.md each round based on results
"""

from __future__ import annotations

import copy
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# RSPL — Resource definitions
# ---------------------------------------------------------------------------

class ResourceType(Enum):
    AGENT = "agent"
    PROMPT = "prompt"
    TOOL = "tool"
    MEMORY = "memory"
    STRATEGY = "strategy"


class ResourceLifecycle(Enum):
    DRAFT = auto()
    ACTIVE = auto()
    DEPRECATED = auto()
    ROLLED_BACK = auto()


@dataclass
class VersionedResource:
    """
    RSPL: a versioned, lifecycle-tracked resource.
    Every agent, prompt, and strategy document is a VersionedResource.
    """
    resource_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    resource_type: ResourceType = ResourceType.AGENT
    name: str = ""
    version: int = 1
    lifecycle: ResourceLifecycle = ResourceLifecycle.ACTIVE
    state: Dict[str, Any] = field(default_factory=dict)
    parent_version: Optional[int] = None
    created_at: float = field(default_factory=time.time)
    score: float = 0.0
    lineage: List[str] = field(default_factory=list)  # list of resource_ids

    def bump_version(self) -> "VersionedResource":
        """Return a new resource with incremented version, same lineage."""
        child = copy.deepcopy(self)
        child.resource_id = str(uuid.uuid4())[:8]
        child.parent_version = self.version
        child.version = self.version + 1
        child.lifecycle = ResourceLifecycle.DRAFT
        child.created_at = time.time()
        child.score = 0.0
        child.lineage = self.lineage + [self.resource_id]
        return child

    def to_dict(self) -> Dict[str, Any]:
        return {
            "resource_id": self.resource_id,
            "resource_type": self.resource_type.value,
            "name": self.name,
            "version": self.version,
            "lifecycle": self.lifecycle.name,
            "state": self.state,
            "parent_version": self.parent_version,
            "score": self.score,
            "lineage_depth": len(self.lineage),
        }


# ---------------------------------------------------------------------------
# SEPL — Evolution operations
# ---------------------------------------------------------------------------

class EvolutionStatus(Enum):
    PROPOSED = auto()
    ASSESSED = auto()
    COMMITTED = auto()
    ROLLED_BACK = auto()


@dataclass
class EvolutionProposal:
    """
    SEPL: a single evolution step with full lineage.
    Mirrors HyperAgents' diff-based MetaAgent output.
    """
    proposal_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    resource_id: str = ""
    resource_name: str = ""
    proposed_changes: Dict[str, Any] = field(default_factory=dict)
    rationale: str = ""
    status: EvolutionStatus = EvolutionStatus.PROPOSED
    assessed_score: float = 0.0
    baseline_score: float = 0.0
    committed_at: Optional[float] = None
    rolled_back_at: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def improvement(self) -> float:
        return self.assessed_score - self.baseline_score

    def to_dict(self) -> Dict[str, Any]:
        return {
            "proposal_id": self.proposal_id,
            "resource_name": self.resource_name,
            "rationale": self.rationale,
            "status": self.status.name,
            "baseline_score": round(self.baseline_score, 3),
            "assessed_score": round(self.assessed_score, 3),
            "improvement": round(self.improvement, 3),
            "proposed_changes": self.proposed_changes,
        }


class EvolutionArchive:
    """
    HyperAgents-style archive of all versioned resources and evolution history.
    Enables parent selection: always pick the best ancestor to evolve from.
    """

    def __init__(self) -> None:
        self._resources: Dict[str, List[VersionedResource]] = {}   # name → versions
        self._proposals: List[EvolutionProposal] = []

    # Resource management
    def register(self, resource: VersionedResource) -> None:
        self._resources.setdefault(resource.name, []).append(resource)

    def best_version(self, name: str) -> Optional[VersionedResource]:
        """Return the highest-scoring ACTIVE version — HyperAgents parent selection."""
        versions = [
            r for r in self._resources.get(name, [])
            if r.lifecycle == ResourceLifecycle.ACTIVE
        ]
        if not versions:
            return None
        return max(versions, key=lambda r: r.score)

    def all_versions(self, name: str) -> List[VersionedResource]:
        return list(self._resources.get(name, []))

    # Proposal management
    def record_proposal(self, proposal: EvolutionProposal) -> None:
        self._proposals.append(proposal)

    def commit(self, proposal: EvolutionProposal, resource: VersionedResource) -> None:
        proposal.status = EvolutionStatus.COMMITTED
        proposal.committed_at = time.time()
        resource.lifecycle = ResourceLifecycle.ACTIVE
        self.register(resource)

    def rollback(self, proposal: EvolutionProposal, resource: VersionedResource) -> None:
        proposal.status = EvolutionStatus.ROLLED_BACK
        proposal.rolled_back_at = time.time()
        resource.lifecycle = ResourceLifecycle.ROLLED_BACK
        logger.info("Rolled back proposal %s for resource %s", proposal.proposal_id, resource.name)

    def summary(self) -> Dict[str, Any]:
        total_committed = sum(1 for p in self._proposals if p.status == EvolutionStatus.COMMITTED)
        total_rolled_back = sum(1 for p in self._proposals if p.status == EvolutionStatus.ROLLED_BACK)
        avg_improvement = (
            sum(p.improvement for p in self._proposals if p.status == EvolutionStatus.COMMITTED)
            / max(total_committed, 1)
        )
        return {
            "total_resources": sum(len(v) for v in self._resources.values()),
            "total_proposals": len(self._proposals),
            "committed": total_committed,
            "rolled_back": total_rolled_back,
            "avg_improvement_per_commit": round(avg_improvement, 3),
        }


class EvolutionEngine:
    """
    SEPL closed-loop: Propose → Assess → Commit/Rollback.

    Inspired by:
    - Autogenesis SEPL (SkyworkAI): formal propose/assess/commit with rollback
    - HyperAgents (Meta): MetaAgent proposes code diffs
    - Karpathy: agent updates strategy.md based on experiment outcome
    - Bilevel Autoresearch: outer loop optimises the inner loop's hyperparams
    """

    def __init__(
        self,
        archive: Optional[EvolutionArchive] = None,
        strategy_path: str = "strategy.md",
        min_improvement_to_commit: float = 0.01,
    ) -> None:
        self.archive = archive or EvolutionArchive()
        self.strategy_path = Path(strategy_path)
        self.min_improvement = min_improvement_to_commit
        self._generation: int = 0

    @property
    def generation(self) -> int:
        return self._generation

    # ------------------------------------------------------------------
    # Main evolution step
    # ------------------------------------------------------------------

    def evolve(
        self,
        resource: VersionedResource,
        round_score: float,
        agent_scores: Dict[str, float],
        learnings: List[str],
        assessor: Optional[Callable[[VersionedResource], float]] = None,
    ) -> Tuple[VersionedResource, EvolutionProposal]:
        """
        Run one SEPL evolution cycle for a resource.

        1. Propose changes based on scores and learnings
        2. Assess the proposed version (run assessor or estimate)
        3. Commit if improvement > threshold, else rollback

        Returns the (possibly updated) resource and the proposal record.
        """
        baseline_score = resource.score if resource.score > 0 else round_score
        proposal = self._propose(resource, round_score, agent_scores, learnings)
        candidate = resource.bump_version()
        candidate.state = {**resource.state, **proposal.proposed_changes}

        # Assess
        if assessor is not None:
            assessed_score = assessor(candidate)
        else:
            assessed_score = self._estimate_score(candidate, round_score, proposal)

        proposal.baseline_score = baseline_score
        proposal.assessed_score = assessed_score
        proposal.status = EvolutionStatus.ASSESSED
        self.archive.record_proposal(proposal)

        # Commit or rollback
        if proposal.improvement >= self.min_improvement:
            candidate.score = assessed_score
            self.archive.commit(proposal, candidate)
            self._generation += 1
            logger.info(
                "Evolution committed: gen=%d, improvement=+%.3f",
                self._generation, proposal.improvement,
            )
            self._update_strategy(proposal, agent_scores, learnings)
            return candidate, proposal
        else:
            self.archive.rollback(proposal, candidate)
            resource.score = max(resource.score, round_score)
            logger.info(
                "Evolution rolled back: improvement=%.3f < threshold=%.3f",
                proposal.improvement, self.min_improvement,
            )
            return resource, proposal

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _propose(
        self,
        resource: VersionedResource,
        round_score: float,
        agent_scores: Dict[str, float],
        learnings: List[str],
    ) -> EvolutionProposal:
        """
        Generate parameter change proposals based on agent performance.
        In production: backed by MetaAgent LLM call.
        Here: heuristic rules inspired by Bilevel Autoresearch.
        """
        changes: Dict[str, Any] = {}
        notes: List[str] = []
        state = resource.state

        # Heuristic: if overall score is low, increase search depth/breadth
        if round_score < 0.4:
            changes["search_depth"] = min(state.get("search_depth", 2) + 1, 4)
            notes.append("low round score → increase search depth")

        # Heuristic: if hypothesis agent scored low, increase MCTS iterations
        if agent_scores.get("HypothesisAgent", 1.0) < 0.5:
            changes["mcts_iterations"] = min(state.get("mcts_iterations", 20) + 10, 60)
            notes.append("hypothesis agent underperformed → more MCTS iterations")

        # Heuristic: if knowledge agent acceptance rate is low, soften threshold
        if agent_scores.get("KnowledgeAgent", 1.0) < 0.3:
            changes["acceptance_threshold"] = max(
                state.get("acceptance_threshold", 0.55) - 0.05, 0.30
            )
            notes.append("knowledge agent rejecting too many → lower acceptance threshold")

        # Incorporate learnings into open_directions
        if learnings:
            changes["open_directions"] = learnings[-3:]
            notes.append(f"added {min(3, len(learnings))} new research directions")

        # If nothing changed, try small random perturbation of search_breadth
        if not changes:
            current_breadth = state.get("search_breadth", 4)
            changes["search_breadth"] = current_breadth + (1 if round_score > 0.6 else -1)
            changes["search_breadth"] = max(2, min(changes["search_breadth"], 8))
            notes.append("minor breadth adjustment")

        return EvolutionProposal(
            resource_id=resource.resource_id,
            resource_name=resource.name,
            proposed_changes=changes,
            rationale=" | ".join(notes),
            metadata={"round_score": round_score, "agent_scores": agent_scores},
        )

    @staticmethod
    def _estimate_score(
        candidate: VersionedResource,
        current_score: float,
        proposal: EvolutionProposal,
    ) -> float:
        """
        Simple score estimator when no real assessor is available.
        Production: run a fast evaluation pass.
        """
        # Proposals with clear rationale get a small boost
        rationale_bonus = min(len(proposal.rationale.split("|")) * 0.02, 0.06)
        return min(current_score + rationale_bonus, 1.0)

    def _update_strategy(
        self,
        proposal: EvolutionProposal,
        agent_scores: Dict[str, float],
        learnings: List[str],
    ) -> None:
        """
        Karpathy pattern: update strategy.md after a successful evolution.
        The strategy file is the live memory of what works.
        """
        if not self.strategy_path.exists():
            return
        try:
            existing = self.strategy_path.read_text(encoding="utf-8")
            # Find and update the parameters block
            import re
            params_block = json.dumps(proposal.proposed_changes, indent=2)
            new_section = (
                f"\n## Round {self._generation} Update\n"
                f"- Score improvement: +{proposal.improvement:.3f}\n"
                f"- Changes: {proposal.rationale}\n"
                f"- Agent scores: {json.dumps(agent_scores)}\n"
            )
            updated = existing + new_section
            self.strategy_path.write_text(updated, encoding="utf-8")
            logger.info("strategy.md updated for generation %d", self._generation)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not update strategy.md: %s", exc)
