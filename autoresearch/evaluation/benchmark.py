"""
AIRS-Bench Evaluation Framework.

Paper: AIRS-Bench (2026) — "A Suite of Tasks for Frontier AI Research Science Agents"
Source: https://ai.meta.com/research/ (Feb 10, 2026)

Key findings from AIRS-Bench:
  - Current agents fail to match human SOTA in complex tasks
  - Performance ceiling = theoretical max achievable by a perfect agent
  - Reliability gap = failure rate on non-trivial experiment execution
  - The benchmark defines the roadmap for next-generation agents

This module implements:
  1. BenchmarkTask — a structured research task with ground-truth criteria
  2. AgentEvaluation — evaluate any autoresearch agent on a task
  3. LeaderboardTracker — track improvements across rounds/versions
  4. PerformanceCeilingAnalyser — compute distance from theoretical SOTA
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Benchmark task definitions
# ---------------------------------------------------------------------------

@dataclass
class EvaluationCriteria:
    """AIRS-Bench style evaluation rubric for a research task."""
    min_novelty:         float = 0.5    # research must be non-trivial
    min_validity:        float = 0.6    # claims must be supported
    min_reproducibility: float = 0.4    # methodology must be documentable
    min_technical_depth: float = 0.5
    max_hallucinations:  int   = 3      # from ACM evaluation finding
    requires_experiments: bool = False
    requires_citations:   bool = True
    human_sota_score:     float = 1.0   # theoretical performance ceiling


@dataclass
class BenchmarkTask:
    """A single AIRS-Bench evaluation task."""
    task_id:     str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    name:        str = ""
    description: str = ""
    domain:      str = "general"       # ml / nlp / vision / biology / general
    difficulty:  str = "medium"        # easy / medium / hard / frontier
    criteria:    EvaluationCriteria = field(default_factory=EvaluationCriteria)
    tags:        List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id":    self.task_id,
            "name":       self.name,
            "domain":     self.domain,
            "difficulty": self.difficulty,
            "tags":       self.tags,
        }


# ---------------------------------------------------------------------------
# Evaluation result
# ---------------------------------------------------------------------------

@dataclass
class EvaluationResult:
    task_id:          str
    agent_id:         str
    round_num:        int
    timestamp:        float = field(default_factory=time.time)
    overall_score:    float = 0.0
    dimension_scores: Dict[str, float] = field(default_factory=dict)
    passed:           bool  = False
    hallucination_count: int = 0
    reliability_flags: List[str] = field(default_factory=list)
    performance_ceiling_gap: float = 1.0
    elapsed_seconds:  float = 0.0
    metadata:         Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id":           self.task_id,
            "agent_id":          self.agent_id,
            "round_num":         self.round_num,
            "overall_score":     round(self.overall_score, 3),
            "passed":            self.passed,
            "hallucination_count": self.hallucination_count,
            "performance_ceiling_gap": round(self.performance_ceiling_gap, 3),
            "dimension_scores":  {k: round(v, 3) for k, v in self.dimension_scores.items()},
            "reliability_flags": self.reliability_flags[:5],
        }


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class AIRSBenchEvaluator:
    """
    AIRS-Bench style evaluator.

    Evaluates an autoresearch agent's output against a BenchmarkTask,
    tracking the performance ceiling gap and reliability metrics.
    """

    def __init__(self, save_path: Optional[str] = None) -> None:
        self._results: List[EvaluationResult] = []
        self._save_path = Path(save_path) if save_path else None

    def evaluate(
        self,
        task: BenchmarkTask,
        review_result: Dict[str, Any],
        agent_id:   str = "sera_x",
        round_num:  int = 0,
    ) -> EvaluationResult:
        """
        Score a research output against a BenchmarkTask using AIRS-Bench criteria.

        review_result should come from ReviewerAgent._execute() output.
        """
        t0 = time.time()
        criteria = task.criteria

        overall = review_result.get("overall_score", 0.0)
        dims    = review_result.get("dimension_scores", {})
        flags   = review_result.get("reliability_flags", [])
        h_count = len(review_result.get("reliability_flags", []))
        gap     = review_result.get("performance_ceiling_gap",
                                    max(0.0, criteria.human_sota_score - overall))

        # Pass/fail against criteria
        passed = (
            dims.get("novelty", 0)         >= criteria.min_novelty
            and dims.get("validity", 0)    >= criteria.min_validity
            and dims.get("reproducibility", 0) >= criteria.min_reproducibility
            and dims.get("technical_depth", 0) >= criteria.min_technical_depth
            and h_count                    <= criteria.max_hallucinations
        )

        result = EvaluationResult(
            task_id=task.task_id,
            agent_id=agent_id,
            round_num=round_num,
            overall_score=overall,
            dimension_scores=dims,
            passed=passed,
            hallucination_count=h_count,
            reliability_flags=flags,
            performance_ceiling_gap=gap,
            elapsed_seconds=time.time() - t0,
        )

        self._results.append(result)
        if self._save_path:
            self._persist(result)

        logger.info(
            "AIRS-Bench eval: task=%s agent=%s score=%.3f passed=%s gap=%.3f",
            task.name, agent_id, overall, passed, gap,
        )
        return result

    # ------------------------------------------------------------------
    # Leaderboard & trend analysis
    # ------------------------------------------------------------------

    def leaderboard(self, top_n: int = 10) -> List[Dict[str, Any]]:
        """Return top-N results by overall score."""
        ranked = sorted(self._results, key=lambda r: r.overall_score, reverse=True)
        return [r.to_dict() for r in ranked[:top_n]]

    def trend(self, agent_id: str) -> Dict[str, Any]:
        """Score trend for a specific agent across rounds."""
        agent_results = sorted(
            [r for r in self._results if r.agent_id == agent_id],
            key=lambda r: r.round_num,
        )
        if not agent_results:
            return {"agent_id": agent_id, "rounds": 0, "trend": "no data"}

        scores = [r.overall_score for r in agent_results]
        improving = len(scores) >= 2 and scores[-1] > scores[0]

        return {
            "agent_id":    agent_id,
            "rounds":      len(agent_results),
            "first_score": round(scores[0], 3),
            "best_score":  round(max(scores), 3),
            "last_score":  round(scores[-1], 3),
            "improvement": round(scores[-1] - scores[0], 3),
            "pass_rate":   round(sum(r.passed for r in agent_results) / len(agent_results), 3),
            "avg_hallucinations": round(
                sum(r.hallucination_count for r in agent_results) / len(agent_results), 2
            ),
            "trend":       "improving" if improving else ("flat" if len(scores) < 2 else "declining"),
        }

    def reliability_report(self) -> Dict[str, Any]:
        """ACM-style reliability analysis (42% failure rate finding)."""
        if not self._results:
            return {"error": "no results"}
        total = len(self._results)
        passed = sum(1 for r in self._results if r.passed)
        failures = total - passed
        avg_hallucinations = sum(r.hallucination_count for r in self._results) / total
        avg_ceiling_gap = sum(r.performance_ceiling_gap for r in self._results) / total

        all_flags: List[str] = []
        for r in self._results:
            all_flags.extend(r.reliability_flags)

        return {
            "total_evaluations": total,
            "pass_rate":          round(passed / total, 3),
            "failure_rate":       round(failures / total, 3),
            "avg_hallucinations": round(avg_hallucinations, 2),
            "avg_ceiling_gap":    round(avg_ceiling_gap, 3),
            "total_flags_raised": len(all_flags),
            "most_common_flags":  _top_flags(all_flags),
            "airs_bench_note": (
                "Failure rate benchmarked against AIRS-Bench (2026) finding: "
                "current agents achieve 42% failure rate on complex tasks."
            ),
        }

    # ------------------------------------------------------------------

    def _persist(self, result: EvaluationResult) -> None:
        try:
            self._save_path.parent.mkdir(parents=True, exist_ok=True)
            data: List[Dict] = []
            if self._save_path.exists():
                try:
                    data = json.loads(self._save_path.read_text())
                except (json.JSONDecodeError, OSError):
                    data = []
            data.append(result.to_dict())
            self._save_path.write_text(json.dumps(data, indent=2))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not persist evaluation result: %s", exc)


# ---------------------------------------------------------------------------
# Built-in benchmark tasks (representative AIRS-Bench suite)
# ---------------------------------------------------------------------------

BENCHMARK_TASKS: Dict[str, BenchmarkTask] = {
    "ml_optimisation": BenchmarkTask(
        name="ML Hyperparameter Optimisation",
        description="Optimise val_bpb on a language model with a 5-minute training budget",
        domain="ml",
        difficulty="medium",
        criteria=EvaluationCriteria(
            min_novelty=0.3,
            min_validity=0.7,
            min_reproducibility=0.6,
            requires_experiments=True,
            human_sota_score=0.95,
        ),
        tags=["karpathy", "bilevel", "hpo"],
    ),
    "literature_synthesis": BenchmarkTask(
        name="Literature Synthesis",
        description="Synthesise a research area from 20+ papers into a coherent survey",
        domain="nlp",
        difficulty="hard",
        criteria=EvaluationCriteria(
            min_novelty=0.5,
            min_validity=0.6,
            min_reproducibility=0.3,
            requires_citations=True,
            human_sota_score=1.0,
        ),
        tags=["gpt-researcher", "agentlaboratory"],
    ),
    "hypothesis_generation": BenchmarkTask(
        name="Novel Hypothesis Generation",
        description="Generate and validate 5 novel research hypotheses for a given topic",
        domain="general",
        difficulty="hard",
        criteria=EvaluationCriteria(
            min_novelty=0.7,
            min_validity=0.5,
            requires_citations=True,
            human_sota_score=1.0,
        ),
        tags=["ai-scientist-v2", "mcts"],
    ),
    "self_improvement": BenchmarkTask(
        name="Agent Self-Improvement",
        description="Demonstrate measurable improvement in strategy across 3+ rounds",
        domain="general",
        difficulty="frontier",
        criteria=EvaluationCriteria(
            min_novelty=0.4,
            min_validity=0.6,
            min_reproducibility=0.5,
            human_sota_score=1.0,
        ),
        tags=["hyperagents", "dgm", "bilevel"],
    ),
}


def _top_flags(flags: List[str], n: int = 5) -> List[str]:
    counts: Dict[str, int] = {}
    for f in flags:
        key = f[:60]
        counts[key] = counts.get(key, 0) + 1
    return sorted(counts, key=counts.get, reverse=True)[:n]  # type: ignore[arg-type]
