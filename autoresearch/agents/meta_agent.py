"""
MetaAgent — self-evolving meta-agent that improves the research strategy.

Synthesises ideas from:
- HyperAgents (Meta): MetaAgent reads eval results, proposes codebase diffs
- Karpathy/autoresearch: agent reads program.md + strategy.md, iterates train.py
- Autogenesis (SkyworkAI): SEPL propose→assess→commit with rollback + lineage
- DGM-Hyperagents (Meta): meta-level improvement of the improvement procedure itself
- Hermes Agent (NousResearch): skills auto-created from experience, self-improve with use
- Bilevel Autoresearch (2026): outer loop optimises inner loop hyperparameters

The MetaAgent:
1. Reads program.md (human intent) and strategy.md (current live strategy)
2. Reads agent performance history from the ExperienceStore
3. Proposes parameter and strategy changes (EvolutionProposal)
4. Commits changes to strategy.md if they improve over baseline
5. Auto-generates/updates Skills from successful patterns
"""

from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from autoresearch.agents.base_agent import BaseAgent
from autoresearch.evolution.evolution_protocol import (
    EvolutionArchive,
    EvolutionEngine,
    EvolutionStatus,
    ResourceLifecycle,
    ResourceType,
    VersionedResource,
)
from autoresearch.memory.experience_store import ExperienceStore, SkillRecord

logger = logging.getLogger(__name__)


class MetaAgent(BaseAgent):
    """
    Self-evolving MetaAgent — reads results, proposes improvements, commits if better.

    This is the agent that makes the system self-improving over time.
    It does NOT run research itself — it improves the agents that do.
    """

    def __init__(
        self,
        config: Optional[Any] = None,
        store: Optional[ExperienceStore] = None,
        evolution_engine: Optional[EvolutionEngine] = None,
        program_path: str = "program.md",
        strategy_path: str = "strategy.md",
        **kwargs: Any,
    ) -> None:
        super().__init__(config=config, name="MetaAgent", **kwargs)
        self.store = store or ExperienceStore()
        self.engine = evolution_engine or EvolutionEngine(
            strategy_path=strategy_path,
            min_improvement_to_commit=self._parameters.get("min_improvement", 0.01),
        )
        self.program_path = Path(program_path)
        self.strategy_path = Path(strategy_path)

        # Register the strategy document as the primary evolving resource
        self._strategy_resource = VersionedResource(
            name="research_strategy",
            resource_type=ResourceType.STRATEGY,
            state=self._load_strategy_params(),
        )
        self.engine.archive.register(self._strategy_resource)

    def _default_parameters(self) -> Dict[str, Any]:
        return {
            "min_improvement": 0.01,
            "max_skills_per_round": 3,
            "skill_score_threshold": 0.6,
            "history_window": 10,
        }

    def _execute(self, task: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        One MetaAgent evolution cycle.

        context must contain:
          - round_score: float — overall score of the just-completed research round
          - agent_scores: dict[str, float] — per-agent scores
          - learnings: list[str] — key learnings from the round
          - round_id: str — ID of the completed round
        """
        round_score: float = context.get("round_score", 0.0)
        agent_scores: Dict[str, float] = context.get("agent_scores", {})
        learnings: List[str] = context.get("learnings", [])
        round_id: str = context.get("round_id", "")

        # 1. Read program.md intent and current strategy
        program_intent = self._read_program()
        current_strategy = self._load_strategy_params()

        # 2. Read agent performance history
        history = self._build_history(agent_scores)

        # 3. Diagnose weak agents
        weak_agents = [
            name for name, score in agent_scores.items() if score < 0.5
        ]

        # 4. Run evolution cycle (SEPL: propose → assess → commit/rollback)
        updated_resource, proposal = self.engine.evolve(
            resource=self._strategy_resource,
            round_score=round_score,
            agent_scores=agent_scores,
            learnings=learnings,
        )
        self._strategy_resource = updated_resource

        # 5. Auto-generate skills from successful patterns (Hermes pattern)
        new_skills = []
        if round_score >= self._parameters["skill_score_threshold"]:
            new_skills = self._extract_skills(task, learnings, agent_scores, round_id)

        # 6. Build feedback for the orchestrator
        return {
            "task": task,
            "generation": self.engine.generation,
            "round_score": round_score,
            "evolution_committed": proposal.status == EvolutionStatus.COMMITTED,
            "improvement": proposal.improvement,
            "proposed_changes": proposal.proposed_changes,
            "rationale": proposal.rationale,
            "weak_agents": weak_agents,
            "new_skills": [s.name for s in new_skills],
            "strategy_version": self._strategy_resource.version,
            "archive_summary": self.engine.archive.summary(),
            "recommendations": self._build_recommendations(
                round_score, agent_scores, proposal, history
            ),
        }

    # ------------------------------------------------------------------
    # Program + strategy reading
    # ------------------------------------------------------------------

    def _read_program(self) -> str:
        if self.program_path.exists():
            return self.program_path.read_text(encoding="utf-8")[:2000]
        return "No program.md found — using defaults."

    def _load_strategy_params(self) -> Dict[str, Any]:
        """Parse the JSON params block from strategy.md."""
        if not self.strategy_path.exists():
            return {}
        text = self.strategy_path.read_text(encoding="utf-8")
        match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass
        return {}

    # ------------------------------------------------------------------
    # History analysis
    # ------------------------------------------------------------------

    def _build_history(self, current_scores: Dict[str, float]) -> Dict[str, Any]:
        """Pull per-agent history from the store and compute trends."""
        trends: Dict[str, Any] = {}
        n = self._parameters.get("history_window", 10)
        for agent_name, current_score in current_scores.items():
            history = self.store.agent_history(agent_name, n=n)
            if history:
                past_scores = [h.get("score", 0.0) for h in history]
                avg = sum(past_scores) / len(past_scores)
                trend = current_score - avg
                trends[agent_name] = {
                    "current": round(current_score, 3),
                    "avg": round(avg, 3),
                    "trend": round(trend, 3),
                    "runs": len(past_scores),
                }
        return trends

    # ------------------------------------------------------------------
    # Skill extraction (Hermes pattern)
    # ------------------------------------------------------------------

    def _extract_skills(
        self,
        task: str,
        learnings: List[str],
        agent_scores: Dict[str, float],
        round_id: str,
    ) -> List[SkillRecord]:
        """
        Auto-generate skills from a successful research round.
        Skill = a reusable pattern that produced a high score.
        """
        skills: List[SkillRecord] = []
        max_skills = self._parameters.get("max_skills_per_round", 3)

        # Build skill from top learnings
        top_learnings = learnings[:max_skills]
        for i, learning in enumerate(top_learnings):
            # Derive skill name from first 4 words of learning
            words = re.findall(r"\w+", learning)[:4]
            skill_name = "_".join(w.lower() for w in words) or f"skill_{round_id}_{i}"

            existing = self.store.get_skill(skill_name)
            if existing:
                # Skill already exists — update its score (Hermes: self-improving)
                self.store.update_skill_score(skill_name, agent_scores.get("WriterAgent", 0.7))
                skills.append(existing)
            else:
                skill = SkillRecord(
                    name=skill_name,
                    description=learning[:200],
                    code=f"# Auto-extracted from round {round_id}\n# Task: {task[:80]}\n# {learning[:150]}",
                    avg_score=agent_scores.get("WriterAgent", 0.7),
                    metadata={"source_round": round_id, "task": task[:80]},
                )
                self.store.save_skill(skill)
                skills.append(skill)

        return skills

    # ------------------------------------------------------------------
    # Recommendations
    # ------------------------------------------------------------------

    def _build_recommendations(
        self,
        round_score: float,
        agent_scores: Dict[str, float],
        proposal: Any,
        history: Dict[str, Any],
    ) -> List[str]:
        recs: List[str] = []

        if round_score < 0.3:
            recs.append("Score critically low — consider simplifying the task or increasing search depth")
        elif round_score < 0.5:
            recs.append("Score below target — check weak agents and increase MCTS iterations")

        for agent_name, data in history.items():
            if data.get("trend", 0) < -0.1:
                recs.append(f"{agent_name} is declining — review its parameters")
            elif data.get("trend", 0) > 0.1:
                recs.append(f"{agent_name} is improving — keep current parameters")

        if proposal.status == EvolutionStatus.COMMITTED:
            recs.append(f"Evolution committed (gen {self.engine.generation}): {proposal.rationale}")
        else:
            recs.append("Evolution rolled back — maintaining current strategy")

        # Check for skills to leverage
        top_skills = self.store.list_skills(min_score=0.7)
        if top_skills:
            names = [s.name for s in top_skills[:3]]
            recs.append(f"High-value skills available: {', '.join(names)}")

        return recs

    def _score_result(self, result: Any) -> float:
        return result.get("improvement", 0.0) + (0.1 if result.get("evolution_committed") else 0.0)
