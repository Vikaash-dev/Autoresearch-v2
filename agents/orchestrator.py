"""
Orchestrator: manages the full research pipeline end-to-end.

Synthesises:
  - AI-Scientist v2 Experiment Manager Agent
  - karpathy/autoresearch "research org code" (program.md as the coordination layer)
  - NousResearch/hermes-agent's task delegation to subagents
  - NousResearch/autonovel's pipeline phases (foundation → draft → revise → export)

Pipeline stages:
  Stage 0: Task decomposition (Task-ToM) + idea loading
  Stage 1: Baseline establishment (BFTS seeds)
  Stage 2: Ablation experiments (tree expansion)
  Stage 3: Full experiments + analysis
  Stage 4: Paper writeup + adversarial review (Reviewer-ToM)
  Stage 5: Self-evolution (GEPA optimizes agent skills for next run)
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from core.blackboard import Blackboard
from core.loop import ResearchLoop, LoopConfig
from core.tree_search import BestFirstTreeSearch, BFTSConfig
from agents.base_agent import BaseAgent
from agents.hypothesis_agent import HypothesisAgent
from agents.experiment_agent import ExperimentAgent
from agents.literature_agent import LiteratureAgent
from agents.writer_agent import WriterAgent
from agents.reviewer_agent import ReviewerAgent
from tom.engine import TheoryOfMindEngine
from evolution.gepa_optimizer import GEPAOptimizer, GEPAConfig, Candidate

logger = logging.getLogger(__name__)


class Orchestrator:
    """
    Top-level coordinator for the full Autoresearch-v2 pipeline.

    The Orchestrator is the "research org" — it programs the agents
    (like karpathy programs program.md) and delegates execution to them.
    It uses Theory of Mind to understand the user, model other agents,
    and proactively shape the research direction.
    """

    def __init__(
        self,
        blackboard: Blackboard,
        tom_engine: TheoryOfMindEngine,
        llm_fn: Any,
        bfts_config: BFTSConfig | None = None,
        loop_config: LoopConfig | None = None,
        output_dir: Path | None = None,
        tavily_keys: list[str] | None = None,
    ) -> None:
        self._bb = blackboard
        self._tom = tom_engine
        self._llm = llm_fn
        self._bfts_cfg = bfts_config or BFTSConfig()
        self._loop_cfg = loop_config or LoopConfig()
        self._output_dir = output_dir or Path("experiments/default_run")
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._tavily_keys = tavily_keys

        # Instantiate all agents
        self._hypothesis_agent = HypothesisAgent(
            blackboard=blackboard, tom_engine=tom_engine, llm_fn=llm_fn
        )
        self._experiment_agent = ExperimentAgent(
            blackboard=blackboard, tom_engine=tom_engine, llm_fn=llm_fn
        )
        self._literature_agent = LiteratureAgent(
            blackboard=blackboard,
            tom_engine=tom_engine,
            llm_fn=llm_fn,
            tavily_keys=tavily_keys,
        )
        self._writer_agent = WriterAgent(
            blackboard=blackboard, tom_engine=tom_engine, llm_fn=llm_fn
        )
        self._reviewer_agent = ReviewerAgent(
            blackboard=blackboard, tom_engine=tom_engine, llm_fn=llm_fn
        )

    # ------------------------------------------------------------------ #
    #  Main pipeline                                                       #
    # ------------------------------------------------------------------ #

    def run(
        self,
        idea: dict[str, Any],
        user_description: str = "",
        venue: str = "NeurIPS",
    ) -> Path | None:
        """
        Execute the full research pipeline for a given idea.

        Returns path to the generated PDF (or None if writeup failed).
        """
        run_id = self._bb.init_run(idea)
        logger.info("=== Autoresearch-v2 run started: %s ===", run_id)
        start = time.time()

        # ── Stage 0: Understand intent ──────────────────────────────── #
        logger.info("Stage 0: Task decomposition + user modeling")
        if user_description:
            self._tom.update_user_model(user_description)
        task = self._tom.infer_user_intent(
            idea.get("Title", "") + " — " + idea.get("Abstract", "")
        )
        logger.info("Task decomposed: %s", task.stated_goal)

        # ── Stage 0b: Literature survey (Community-ToM) ──────────────── #
        logger.info("Stage 0b: Literature survey")
        papers = self._literature_agent.act({
            "query": idea.get("Keywords", ""),
            "max_results": 20,
        })
        if papers.get("abstracts"):
            self._tom.update_community_model(
                domain=idea.get("Keywords", ""),
                paper_abstracts=papers["abstracts"],
            )

        # ── Stage 0c: Build Reviewer-ToM before writing starts ────────── #
        logger.info("Stage 0c: Building reviewer model for %s", venue)
        self._tom.build_reviewer_model(
            venue=venue,
            domain=idea.get("Keywords", ""),
            recent_papers=papers.get("titles", []),
        )

        # ── Stage 1–3: BFTS experiments ──────────────────────────────── #
        logger.info("Stages 1-3: Best-First Tree Search experiments")
        bfts = BestFirstTreeSearch(
            blackboard=self._bb,
            config=self._bfts_cfg,
            expand_fn=self._hypothesis_agent.expand,
            evaluate_fn=self._experiment_agent.evaluate,
        )
        best_node = bfts.run()

        if best_node is None:
            logger.error("No successful experiment nodes — aborting")
            return None

        logger.info(
            "Best node found: %s metric=%.4f",
            best_node.node_id, best_node.metric or 0.0
        )

        # ── Stage 4: Write paper ──────────────────────────────────────── #
        logger.info("Stage 4: Writing paper")
        paper_draft = self._writer_agent.act({
            "best_node": best_node,
            "idea": idea,
            "task": task,
            "reviewer_model": self._tom.reviewer_model,
            "community_model": self._tom.community_model,
            "output_dir": str(self._output_dir),
        })

        if paper_draft.get("status") != "ok":
            logger.error("Writer agent failed: %s", paper_draft.get("error"))
            return None

        # ── Stage 4b: Adversarial review (Reviewer-ToM) ──────────────── #
        logger.info("Stage 4b: Adversarial review")
        review = self._reviewer_agent.act({
            "paper_draft": paper_draft.get("draft_text", ""),
            "reviewer_model": self._tom.reviewer_model,
        })

        if review.get("predicted_score", 0) < 5:
            logger.info("Review score low (%s) — triggering revision", review.get("predicted_score"))
            paper_draft = self._writer_agent.act({
                "best_node": best_node,
                "idea": idea,
                "task": task,
                "reviewer_model": self._tom.reviewer_model,
                "community_model": self._tom.community_model,
                "output_dir": str(self._output_dir),
                "review_feedback": review,
                "is_revision": True,
            })

        # ── Stage 5: Self-evolution ────────────────────────────────────── #
        logger.info("Stage 5: Self-evolution analysis")
        self_analysis = self._tom.analyze_self()
        logger.info(
            "Self-analysis: success_rate=%.2f, recommendations=%s",
            self_analysis.get("success_rate", 0),
            self_analysis.get("recommendations", [])[:3],
        )

        # Save the tree visualization
        tree_viz_path = self._output_dir / "tree_viz.txt"
        tree_viz_path.write_text(bfts.render_ascii())

        elapsed = time.time() - start
        logger.info("=== Run complete in %.0fs (%.1f min) ===", elapsed, elapsed / 60)

        # Return path to output directory
        return self._output_dir

    # ------------------------------------------------------------------ #
    #  GEPA-based agent skill optimization                                 #
    # ------------------------------------------------------------------ #

    def evolve_agent_skill(
        self,
        agent: BaseAgent,
        skill_name: str,
        seed_skill_text: str,
        eval_dataset: list[dict[str, Any]],
    ) -> str:
        """
        Use GEPA to evolve an agent's skill file.
        Implements hermes-agent-self-evolution Phase 1 (skill evolution).
        """
        logger.info("Evolving skill %r for agent %s", skill_name, agent.agent_id)

        def evaluator(candidate: Candidate) -> tuple[float, str]:
            scores = []
            traces = []
            for example in eval_dataset[:10]:
                try:
                    result = agent.act({**example, "_skill_override": candidate.artifact})
                    score = float(result.get("metric", 0.0))
                    scores.append(score)
                    traces.append(str(result))
                except Exception as exc:
                    scores.append(0.0)
                    traces.append(str(exc))
            return sum(scores) / max(len(scores), 1), "\n".join(traces)

        optimizer = GEPAOptimizer(
            config=GEPAConfig(max_metric_calls=50),
            llm_fn=self._llm,
            evaluator_fn=evaluator,
            persist_path=self._output_dir / f"gepa_{skill_name}.json",
        )
        best = optimizer.optimize(seed_skill_text, artifact_type="skill")
        logger.info(
            "Skill %r evolved: %.4f -> %.4f",
            skill_name, 0.0, best.metric
        )
        return best.artifact
