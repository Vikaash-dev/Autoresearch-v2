"""
Experiment Agent — experiment design, execution, and result tracking.

Inspired by:
  - AI Scientist-v2 (Sakana AI, 2026): Experiment Manager agent
  - AIRS-Bench (2026): reproducibility and correctness metrics
  - DGM-Hyperagents (Meta, 2026): persistent performance tracking
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional

from autoresearch.agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)


class ExperimentStatus(Enum):
    PENDING = auto()
    RUNNING = auto()
    COMPLETED = auto()
    FAILED = auto()
    SKIPPED = auto()


@dataclass
class ExperimentPlan:
    """Structured experiment specification produced by this agent."""

    experiment_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    hypothesis: str = ""
    objective: str = ""
    methodology: str = ""
    variables: Dict[str, Any] = field(default_factory=dict)  # independent / dependent vars
    metrics: List[str] = field(default_factory=list)
    baseline: Optional[str] = None
    estimated_compute: str = "low"          # low | medium | high
    reproducibility_notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "hypothesis": self.hypothesis,
            "objective": self.objective,
            "methodology": self.methodology,
            "variables": self.variables,
            "metrics": self.metrics,
            "baseline": self.baseline,
            "estimated_compute": self.estimated_compute,
            "reproducibility_notes": self.reproducibility_notes,
        }


@dataclass
class ExperimentResult:
    """Output of a completed experiment."""

    experiment_id: str
    status: ExperimentStatus
    metrics: Dict[str, float] = field(default_factory=dict)
    observations: List[str] = field(default_factory=list)
    artifacts: List[str] = field(default_factory=list)   # paths to logs/plots/etc.
    elapsed_seconds: float = 0.0
    reproducibility_score: float = 0.0
    correctness_score: float = 0.0
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "status": self.status.name,
            "metrics": self.metrics,
            "observations": self.observations,
            "artifacts": self.artifacts,
            "elapsed_seconds": self.elapsed_seconds,
            "reproducibility_score": self.reproducibility_score,
            "correctness_score": self.correctness_score,
            "notes": self.notes,
        }


class ExperimentAgent(BaseAgent):
    """
    Designs and (mock-)executes experiments for each accepted hypothesis.

    Responsibilities:
      1. Translate a validated hypothesis into a structured experiment plan.
      2. Execute the experiment (calls a pluggable runner — default is mock).
      3. Score results for reproducibility and correctness (AIRS-Bench style).
      4. Persist results for the experience store and the meta-agent.

    In production: swap ``_run_experiment`` for a real code-execution sandbox
    (e.g. Docker container, Jupyter kernel, or remote compute cluster).
    """

    def __init__(
        self,
        config: Optional[Any] = None,
        runner: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(config=config, name="ExperimentAgent", **kwargs)
        # ``runner`` is a callable(plan) -> dict that performs actual execution.
        # If None, the mock runner is used.
        self._runner = runner
        self._experiment_log: List[ExperimentResult] = []

    def _default_parameters(self) -> Dict[str, Any]:
        return {
            "max_experiments_per_round": 3,
            "min_reproducibility_runs": 2,
            "target_metrics": ["accuracy", "loss", "novelty"],
            "compute_budget": "medium",
        }

    def _execute(self, task: str, context: Dict[str, Any]) -> Dict[str, Any]:
        accepted: List[Dict[str, Any]] = context.get("accepted_hypotheses", [])
        papers: List[Dict[str, Any]] = context.get("papers", [])
        max_exp: int = self._parameters.get("max_experiments_per_round", 3)

        plans: List[ExperimentPlan] = []
        results: List[ExperimentResult] = []

        for hyp_data in accepted[:max_exp]:
            hyp_text = hyp_data.get("hypothesis", "")
            plan = self._design_experiment(hyp_text, task, papers)
            plans.append(plan)
            result = self._execute_experiment(plan)
            self._experiment_log.append(result)
            results.append(result)

        successful = [r for r in results if r.status == ExperimentStatus.COMPLETED]
        avg_correctness = (
            sum(r.correctness_score for r in successful) / len(successful)
            if successful else 0.0
        )
        avg_reproducibility = (
            sum(r.reproducibility_score for r in successful) / len(successful)
            if successful else 0.0
        )

        return {
            "task": task,
            "experiment_plans": [p.to_dict() for p in plans],
            "experiment_results": [r.to_dict() for r in results],
            "successful_count": len(successful),
            "avg_correctness": round(avg_correctness, 3),
            "avg_reproducibility": round(avg_reproducibility, 3),
        }

    # ------------------------------------------------------------------
    # Experiment design
    # ------------------------------------------------------------------

    def _design_experiment(
        self, hypothesis: str, task: str, papers: List[Dict[str, Any]]
    ) -> ExperimentPlan:
        """
        Translate a hypothesis into a structured experiment plan.
        In production: use LLM with a structured output schema.
        """
        baseline = papers[0].get("title", "prior work") if papers else "baseline"
        methodology = papers[0].get("methodology", "standard evaluation") if papers else "ablation study"

        return ExperimentPlan(
            hypothesis=hypothesis,
            objective=f"Empirically validate: {hypothesis[:100]}...",
            methodology=methodology,
            variables={
                "independent": ["learning_rate", "batch_size", "architecture"],
                "dependent": ["accuracy", "loss", "training_time"],
                "controlled": ["random_seed", "dataset_split"],
            },
            metrics=self._parameters.get("target_metrics", ["accuracy", "loss"]),
            baseline=baseline,
            estimated_compute=self._parameters.get("compute_budget", "medium"),
            reproducibility_notes=(
                f"Run {self._parameters.get('min_reproducibility_runs', 2)}× "
                f"with different seeds; report mean ± std."
            ),
        )

    # ------------------------------------------------------------------
    # Experiment execution (mock / pluggable)
    # ------------------------------------------------------------------

    def _execute_experiment(self, plan: ExperimentPlan) -> ExperimentResult:
        start = time.time()
        if self._runner is not None:
            try:
                raw = self._runner(plan)
                return self._parse_runner_output(plan.experiment_id, raw, time.time() - start)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Runner failed for %s: %s", plan.experiment_id, exc)
                return ExperimentResult(
                    experiment_id=plan.experiment_id,
                    status=ExperimentStatus.FAILED,
                    notes=str(exc),
                    elapsed_seconds=time.time() - start,
                )
        return self._mock_execute(plan, time.time() - start)

    def _mock_execute(self, plan: ExperimentPlan, elapsed: float) -> ExperimentResult:
        """
        Simulate experiment execution with plausible but synthetic metrics.
        Scores are derived from plan properties to make evolution meaningful.
        """
        import random  # local import
        rng = random.Random(hash(plan.hypothesis) % (2 ** 31))

        base_acc = 0.5 + rng.random() * 0.4
        metrics = {
            "accuracy": round(base_acc, 4),
            "loss": round(1.0 - base_acc + rng.random() * 0.1, 4),
            "training_time_s": round(rng.uniform(10, 300), 1),
        }
        observations = [
            f"Model converged after {rng.randint(5, 50)} epochs.",
            f"Peak metric: {max(metrics['accuracy'], 0):.3f}.",
            f"Methodology '{plan.methodology}' applied successfully.",
        ]
        reproducibility = round(0.6 + rng.random() * 0.35, 3)
        correctness = round(base_acc, 3)

        return ExperimentResult(
            experiment_id=plan.experiment_id,
            status=ExperimentStatus.COMPLETED,
            metrics=metrics,
            observations=observations,
            elapsed_seconds=elapsed + rng.uniform(0.1, 1.0),
            reproducibility_score=reproducibility,
            correctness_score=correctness,
            notes="Mock execution — replace runner for real experiments.",
        )

    @staticmethod
    def _parse_runner_output(
        experiment_id: str, raw: Dict[str, Any], elapsed: float
    ) -> ExperimentResult:
        return ExperimentResult(
            experiment_id=experiment_id,
            status=ExperimentStatus.COMPLETED,
            metrics=raw.get("metrics", {}),
            observations=raw.get("observations", []),
            artifacts=raw.get("artifacts", []),
            elapsed_seconds=elapsed,
            reproducibility_score=raw.get("reproducibility_score", 0.5),
            correctness_score=raw.get("correctness_score", 0.5),
            notes=raw.get("notes", ""),
        )

    def _score_result(self, result: Any) -> float:
        if result.get("successful_count", 0) == 0:
            return 0.0
        return (result.get("avg_correctness", 0.0) + result.get("avg_reproducibility", 0.0)) / 2.0

    @property
    def experiment_log(self) -> List[ExperimentResult]:
        return list(self._experiment_log)
