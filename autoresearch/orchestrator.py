"""
SERA-X Orchestrator — Self-Evolving Research Architecture eXtended.

The main entry point. Ties together all 6 layers:

  Layer 0 · INTAKE           — parse task, load program.md + strategy.md
  Layer 1 · DEEP SEARCH      — dzhng recursive SERP (breadth/depth)
  Layer 2 · KNOWLEDGE STORE  — SQLite experience, skills, AgentRxiv learnings
  Layer 3 · WORKFORCE        — LiteratureAgent→HypothesisAgent→KnowledgeAgent
                               →ExperimentAgent→WriterAgent→ReviewerAgent
  Layer 4 · SELF-EVOLUTION   — MetaAgent + Bilevel 3-level loop + Centaur HPO
  Layer 5 · SANDBOX          — SWE-agent ACI + code execution
  Layer 6 · OUTPUT + PUBLISH — AIRS-Bench evaluation, report export, strategy update

Inspired by:
  - karpathy/autoresearch: fixed budget, program.md org charter, strategy.md evolution
  - HyperAgents (Meta): MetaAgent proposes diffs, archive+parent selection, Docker sandbox
  - Bilevel Autoresearch: 3-level loop (task/config/mechanism), 5× improvement
  - AI Scientist-v2: full research lifecycle, template-free, tree search
  - CORAL: multi-agent git worktrees, shared state, adaptive strategy transitions
  - n-autoresearch: structured experiment API, explore→exploit→combine→ablation
  - Sibyl: dual-loop self-evolution (inner=research quality, outer=system)
  - AutoResearchClaw: 23-stage pipeline, MetaClaw cross-run skill injection
  - AIRS-Bench: benchmark evaluation, performance ceiling tracking
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from autoresearch.config import AutoresearchConfig
from autoresearch.agents.research_agent import ResearchAgent
from autoresearch.agents.hypothesis_agent import HypothesisAgent
from autoresearch.agents.knowledge_agent import KnowledgeAgent
from autoresearch.agents.experiment_agent import ExperimentAgent
from autoresearch.agents.writer_agent import WriterAgent
from autoresearch.agents.reviewer_agent import ReviewerAgent
from autoresearch.agents.meta_agent import MetaAgent
from autoresearch.pipeline.deep_search import DeepSearchEngine, SearchProvider, LLMProvider
from autoresearch.memory.experience_store import ExperienceStore
from autoresearch.optimization.bilevel import BilevelEngine, BilevelConfig
from autoresearch.evaluation.benchmark import AIRSBenchEvaluator, BENCHMARK_TASKS

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Round result
# ---------------------------------------------------------------------------

@dataclass
class RoundResult:
    round_id:     str   = field(default_factory=lambda: str(uuid.uuid4())[:8])
    round_num:    int   = 0
    task:         str   = ""
    learnings:    List[str] = field(default_factory=list)
    hypotheses:   List[Any] = field(default_factory=list)
    experiments:  List[Any] = field(default_factory=list)
    report:       Dict[str, Any] = field(default_factory=dict)
    review:       Dict[str, Any] = field(default_factory=dict)
    evolution:    Dict[str, Any] = field(default_factory=dict)
    score:        float = 0.0
    elapsed_s:    float = 0.0
    status:       str   = "completed"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "round_id":    self.round_id,
            "round_num":   self.round_num,
            "score":       round(self.score, 3),
            "status":      self.status,
            "learnings":   len(self.learnings),
            "hypotheses":  len(self.hypotheses),
            "experiments": len(self.experiments),
            "elapsed_s":   round(self.elapsed_s, 1),
            "review_decision": self.review.get("accept_decision", ""),
            "evolution_committed": self.evolution.get("evolution_committed", False),
        }


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class ResearchOrchestrator:
    """
    SERA-X: runs the complete self-evolving research loop.

    Usage (CLI):
        python -m autoresearch "What are the best hyperparameter tuning methods for LLMs?"

    Usage (API):
        from autoresearch import ResearchOrchestrator, AutoresearchConfig
        orch = ResearchOrchestrator(AutoresearchConfig())
        result = orch.run("Research question here", max_rounds=5)
        print(result["best_report"]["markdown"])
    """

    def __init__(
        self,
        config: Optional[AutoresearchConfig] = None,
        search_provider: Optional[SearchProvider] = None,
        llm_provider: Optional[LLMProvider] = None,
        db_path: str = "autoresearch_memory.db",
        eval_path: Optional[str] = None,
        on_round_complete: Optional[Callable[[RoundResult], None]] = None,
        program_path: str = "program.md",
        strategy_path: str = "strategy.md",
    ) -> None:
        self.config = config or AutoresearchConfig()
        self.on_round_complete = on_round_complete

        # --- Layer 1: Deep Search Engine ---
        self.search_engine = DeepSearchEngine(
            search_provider=search_provider,
            llm_provider=llm_provider,
            max_budget_seconds=self.config.search_budget_seconds,
        )

        # --- Layer 2: Knowledge Store ---
        self.store = ExperienceStore(db_path=db_path)

        # --- Layer 3: Research Workforce ---
        self.literature_agent  = ResearchAgent(config=config)
        self.hypothesis_agent  = HypothesisAgent(config=config)
        self.knowledge_agent   = KnowledgeAgent(config=config)
        self.experiment_agent  = ExperimentAgent(config=config)
        self.writer_agent      = WriterAgent(config=config)
        self.reviewer_agent    = ReviewerAgent(config=config)

        # --- Layer 4: Self-Evolution ---
        self.meta_agent = MetaAgent(
            config=config,
            store=self.store,
            program_path=program_path,
            strategy_path=strategy_path,
        )

        # --- Layer 6: Evaluation ---
        self.evaluator = AIRSBenchEvaluator(save_path=eval_path)

        # State
        self._rounds:     List[RoundResult] = []
        self._best_score: float = 0.0
        self._best_round: Optional[RoundResult] = None
        self._start_time: float = 0.0

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(
        self,
        task: str,
        max_rounds: int = 5,
        success_threshold: float = 0.75,
        search_breadth: int = 4,
        search_depth: int = 2,
        benchmark_task_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Run the complete self-evolving research loop.

        Karpathy pattern: give it a task, go to sleep, wake up to a log of
        experiments and (hopefully) better research than when you started.
        """
        self._start_time = time.time()
        logger.info("=" * 60)
        logger.info("SERA-X starting: task='%s', max_rounds=%d", task, max_rounds)
        logger.info("=" * 60)

        bench_task = BENCHMARK_TASKS.get(benchmark_task_id or "literature_synthesis")
        prior_learnings: List[str] = self.store.search_learnings(task, n=20)

        for round_num in range(max_rounds):
            round_start = time.time()
            logger.info("\n--- Round %d/%d ---", round_num + 1, max_rounds)

            round_rec = self.store.start_round(task, metadata={"round_num": round_num})
            rr = RoundResult(
                round_id=round_rec.round_id,
                round_num=round_num,
                task=task,
            )

            try:
                rr = self._execute_round(
                    rr=rr,
                    task=task,
                    prior_learnings=prior_learnings,
                    search_breadth=max(2, search_breadth),
                    search_depth=max(1, search_depth),
                )
            except Exception as exc:  # noqa: BLE001
                logger.error("Round %d failed: %s", round_num + 1, exc, exc_info=True)
                rr.status = "failed"
                rr.score  = 0.0

            rr.elapsed_s = time.time() - round_start

            # AIRS-Bench evaluation
            if bench_task and rr.review:
                eval_result = self.evaluator.evaluate(
                    task=bench_task,
                    review_result=rr.review,
                    agent_id="sera_x",
                    round_num=round_num,
                )
                rr.review["airs_bench"] = eval_result.to_dict()

            # Persist round
            self.store.finish_round(round_rec.round_id, score=rr.score, status=rr.status)
            self.store.save_learnings(round_rec.round_id, rr.learnings)
            prior_learnings = (prior_learnings + rr.learnings)[-50:]  # rolling window

            if rr.score > self._best_score:
                self._best_score = rr.score
                self._best_round = rr

            self._rounds.append(rr)

            if self.on_round_complete:
                self.on_round_complete(rr)

            logger.info(
                "Round %d complete: score=%.3f best=%.3f elapsed=%.1fs",
                round_num + 1, rr.score, self._best_score, rr.elapsed_s,
            )

            # Early exit on success
            if rr.score >= success_threshold:
                logger.info("Success threshold %.2f reached — stopping.", success_threshold)
                break

        return self._build_final_output(task)

    # ------------------------------------------------------------------
    # Single round execution
    # ------------------------------------------------------------------

    def _execute_round(
        self,
        rr: RoundResult,
        task: str,
        prior_learnings: List[str],
        search_breadth: int,
        search_depth: int,
    ) -> RoundResult:

        # ── Layer 0: load prior skills ─────────────────────────────────
        prior_skills = [s.name for s in self.store.list_skills(min_score=0.6)]

        # ── Layer 1: Deep Search ───────────────────────────────────────
        logger.info("Layer 1: Deep Search (breadth=%d, depth=%d)", search_breadth, search_depth)
        search_out = self.search_engine.run(
            query=task,
            breadth=search_breadth,
            depth=search_depth,
            prior_learnings=prior_learnings,
        )
        rr.learnings = search_out.learnings
        logger.info("  Found %d learnings from %d URLs", len(rr.learnings), len(search_out.urls))

        # ── Layer 3a: Literature Agent ─────────────────────────────────
        lit_state = self.literature_agent.run(task, {
            "search_results": rr.learnings,
            "prior_skills": prior_skills,
        })

        # ── Layer 3b: Hypothesis Agent (MCTS — AI Scientist-v2) ────────
        logger.info("Layer 3b: Hypothesis generation (MCTS)")
        hyp_state = self.hypothesis_agent.run(task, {
            "learnings": rr.learnings,
            "prior_context": lit_state.result.get("findings", []) if lit_state.result else [],
        })
        rr.hypotheses = _extract_list(hyp_state.result, "hypotheses")

        # ── Layer 3c: Knowledge Agent (grounding + novelty) ───────────
        logger.info("Layer 3c: Knowledge grounding + novelty filter")
        know_state = self.knowledge_agent.run(task, {
            "hypotheses": rr.hypotheses,
            "learnings": rr.learnings,
        })
        validated_hypotheses = _extract_list(know_state.result, "validated_hypotheses") or rr.hypotheses

        # ── Layer 3d: Experiment Agent ─────────────────────────────────
        logger.info("Layer 3d: Experiment design + execution")
        exp_state = self.experiment_agent.run(task, {
            "accepted_hypotheses": validated_hypotheses,
            "hypotheses": validated_hypotheses,   # legacy key
            "learnings": rr.learnings,
        })
        rr.experiments = (
            _extract_list(exp_state.result, "experiment_results") or
            _extract_list(exp_state.result, "experiments")
        )

        # ── Layer 3e: Writer Agent (AI Scientist-v2 full lifecycle) ────
        logger.info("Layer 3e: Report writing")
        writer_state = self.writer_agent.run(task, {
            "learnings":   rr.learnings,
            "urls":        list(search_out.urls),
            "hypotheses":  validated_hypotheses,
            "experiments": rr.experiments,
            "prior_skills": prior_skills,
        })
        rr.report = writer_state.result or {}

        # ── Layer 3f: Reviewer Agent (AIRS-Bench 7-dim) ───────────────
        logger.info("Layer 3f: Peer review + AIRS-Bench scoring")
        review_state = self.reviewer_agent.run(task, {
            "report":      rr.report,
            "markdown":    rr.report.get("markdown", ""),
            "learnings":   rr.learnings,
            "experiments": rr.experiments,
            "task":        task,
        })
        rr.review = review_state.result or {}
        rr.score  = rr.review.get("overall_score", 0.0)

        # ── Layer 4: MetaAgent — self-evolution ────────────────────────
        logger.info("Layer 4: MetaAgent evolution (gen=%d)", self.meta_agent.engine.generation)
        agent_scores = {
            "LiteratureAgent":  getattr(lit_state, "score", 0.5),
            "HypothesisAgent":  getattr(hyp_state, "score", 0.5),
            "KnowledgeAgent":   getattr(know_state, "score", 0.5),
            "ExperimentAgent":  getattr(exp_state, "score", 0.5),
            "WriterAgent":      getattr(writer_state, "score", 0.5),
            "ReviewerAgent":    getattr(review_state, "score", 0.5),
        }
        meta_state = self.meta_agent.run(task, {
            "round_score":  rr.score,
            "agent_scores": agent_scores,
            "learnings":    rr.learnings,
            "round_id":     rr.round_id,
        })
        rr.evolution = meta_state.result or {}

        # Persist agent states
        for state in [lit_state, hyp_state, know_state, exp_state, writer_state, review_state, meta_state]:
            self.store.save_agent_state(rr.round_id, state)

        return rr

    # ------------------------------------------------------------------
    # Final output
    # ------------------------------------------------------------------

    def _build_final_output(self, task: str) -> Dict[str, Any]:
        elapsed = time.time() - self._start_time
        rounds_summary = [r.to_dict() for r in self._rounds]
        scores = [r.score for r in self._rounds]
        trend = "improving" if len(scores) >= 2 and scores[-1] > scores[0] else "flat"

        return {
            "task":           task,
            "total_rounds":   len(self._rounds),
            "best_score":     round(self._best_score, 3),
            "trend":          trend,
            "elapsed_seconds": round(elapsed, 1),
            "rounds":         rounds_summary,
            "best_report": (
                self._best_round.report if self._best_round else {}
            ),
            "experience_summary": self.store.summary(),
            "evolution_summary":  self.meta_agent.engine.archive.summary(),
            "airs_bench_reliability": self.evaluator.reliability_report(),
            "airs_bench_leaderboard": self.evaluator.leaderboard(top_n=5),
        }

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @property
    def best_score(self) -> float:
        return self._best_score

    @property
    def rounds(self) -> List[RoundResult]:
        return list(self._rounds)

    def print_summary(self) -> None:
        """Print a human-readable summary (Karpathy: wake up to a log of experiments)."""
        print("\n" + "=" * 60)
        print(f"SERA-X Summary — {len(self._rounds)} round(s)")
        print("=" * 60)
        for rr in self._rounds:
            dec = rr.review.get("accept_decision", "?")
            ev  = "✓ evolved" if rr.evolution.get("evolution_committed") else "  no change"
            print(f"  Round {rr.round_num + 1}: score={rr.score:.3f}  decision={dec}  {ev}  {rr.elapsed_s:.0f}s")
        print(f"\n  Best score: {self._best_score:.3f}")
        print(f"  Memory:     {self.store.summary()}")
        print(f"  Evolution:  {self.meta_agent.engine.archive.summary()}")
        print("=" * 60)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _extract_list(result: Any, key: str) -> List[Any]:
    if isinstance(result, dict):
        val = result.get(key, [])
        if isinstance(val, list):
            return val
    return []


def main() -> None:
    import argparse
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    )
    parser = argparse.ArgumentParser(description="SERA-X: Self-Evolving Research Architecture")
    parser.add_argument("task", help="Research task or question")
    parser.add_argument("--rounds",    type=int,   default=3,    help="Max research rounds (default: 3)")
    parser.add_argument("--breadth",   type=int,   default=4,    help="Search breadth (default: 4)")
    parser.add_argument("--depth",     type=int,   default=2,    help="Search depth (default: 2)")
    parser.add_argument("--threshold", type=float, default=0.75, help="Success score threshold (default: 0.75)")
    parser.add_argument("--db",        type=str,   default="autoresearch_memory.db")
    parser.add_argument("--output",    type=str,   default=None, help="Write final report to file")
    args = parser.parse_args()

    orch = ResearchOrchestrator(db_path=args.db)
    result = orch.run(
        task=args.task,
        max_rounds=args.rounds,
        success_threshold=args.threshold,
        search_breadth=args.breadth,
        search_depth=args.depth,
    )

    orch.print_summary()

    if args.output:
        out_path = Path(args.output)
        best_report = result.get("best_report", {})
        md = best_report.get("markdown", json.dumps(result, indent=2, default=str))
        out_path.write_text(md, encoding="utf-8")
        print(f"\nReport written to: {args.output}")
    else:
        best_report = result.get("best_report", {})
        md = best_report.get("markdown", "")
        if md:
            print("\n" + "=" * 60)
            print("BEST REPORT:")
            print("=" * 60)
            print(md[:3000])
            if len(md) > 3000:
                print(f"\n... ({len(md) - 3000} more characters)")


if __name__ == "__main__":
    main()
