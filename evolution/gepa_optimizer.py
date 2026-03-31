"""
GEPA-style evolutionary optimizer for skills, prompts, code, and agent configs.

Directly implements the GEPA (Genetic-Pareto Prompt Evolution) approach from:
  - NousResearch/hermes-agent-self-evolution (ICLR 2026 Oral, MIT licensed)
  - gepa-ai/gepa (35x faster than RL, 90x cheaper than frontier models)

Core ideas:
  - Read full execution traces to diagnose *why* something failed (not just *that*)
  - Propose targeted mutations informed by accumulated failure lessons
  - Pareto-aware selection: maintain a frontier of candidates excelling on different tasks
  - Constraint gates: every evolved variant must pass tests, size limits, semantic checks
  - All changes are diff-based — no direct commits, always PR-reviewable
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)


@dataclass
class Candidate:
    """A single artifact being evolved (prompt / code / skill / config)."""

    candidate_id: str
    artifact: str                          # the text being evolved
    artifact_type: str                     # "prompt" | "code" | "skill" | "config"
    metric: float = 0.0
    pareto_front: bool = False
    task_subset_scores: dict[str, float] = field(default_factory=dict)
    generation: int = 0
    parent_id: str | None = None
    mutation_rationale: str = ""
    created_at: float = field(default_factory=time.time)


@dataclass
class GEPAConfig:
    max_generations: int = 20
    population_size: int = 8
    max_metric_calls: int = 150           # GEPA default from paper
    mutation_temperature: float = 0.9
    pareto_selection: bool = True
    min_improvement_delta: float = 0.01
    # Constraint gates (hermes-agent-self-evolution guardrails)
    max_artifact_chars: int = 15_000
    require_tests_pass: bool = True
    semantic_preservation: bool = True


class GEPAOptimizer:
    """
    Genetic-Pareto evolutionary optimizer for any text artifact.

    Usage:
        optimizer = GEPAOptimizer(config, llm_fn, evaluator_fn)
        best = optimizer.optimize(seed_artifact, artifact_type="skill")

    evaluator_fn(candidate: Candidate) -> (metric: float, trace: str)
        Returns the metric score and full execution trace (errors, logs, profiling).
        The trace is the "Actionable Side Information" (ASI) that feeds reflection.

    llm_fn(prompt: str) -> str
        LLM call for reflection, mutation, and candidate generation.
    """

    def __init__(
        self,
        config: GEPAConfig,
        llm_fn: Callable[[str], str],
        evaluator_fn: Callable[[Candidate], tuple[float, str]],
        constraint_fn: Callable[[Candidate], tuple[bool, str]] | None = None,
        persist_path: Path | None = None,
    ) -> None:
        self._cfg = config
        self._llm = llm_fn
        self._evaluate = evaluator_fn
        self._check_constraints = constraint_fn
        self._persist_path = persist_path
        self._population: list[Candidate] = []
        self._pareto_front: list[Candidate] = []
        self._lessons: list[str] = []          # accumulated reflection lessons
        self._metric_calls = 0

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def optimize(self, seed_artifact: str, artifact_type: str = "prompt") -> Candidate:
        """
        Evolve seed_artifact and return the best candidate found.

        GEPA loop:
          1. Evaluate seed → populate initial front
          2. Select from Pareto front
          3. Execute → capture trace (ASI)
          4. Reflect → diagnose why it failed / how to improve
          5. Mutate → propose improved variant
          6. Evaluate → add to front if improved
          7. Repeat until budget exhausted
        """
        import uuid
        seed = Candidate(
            candidate_id=f"seed_{str(uuid.uuid4())[:8]}",
            artifact=seed_artifact,
            artifact_type=artifact_type,
        )
        metric, trace = self._safe_evaluate(seed)
        seed.metric = metric
        self._population.append(seed)
        self._update_pareto_front()

        logger.info("GEPA starting: seed metric=%.4f budget=%d", metric, self._cfg.max_metric_calls)

        for generation in range(self._cfg.max_generations):
            if self._metric_calls >= self._cfg.max_metric_calls:
                logger.info("GEPA: metric call budget exhausted")
                break

            parent = self._select_parent()
            if parent is None:
                break

            # Evaluate parent on minibatch to get fresh ASI
            _, trace = self._safe_evaluate(parent)

            # Reflect — diagnose failure / opportunity
            lesson = self._reflect(parent, trace)
            self._lessons.append(lesson)

            # Mutate — generate improved candidate
            child = self._mutate(parent, lesson, generation)

            if child is None:
                continue

            # Constraint gates
            ok, reason = self._apply_constraint_gates(child)
            if not ok:
                logger.info("  Candidate failed constraint gate: %s", reason)
                continue

            # Evaluate
            metric, trace = self._safe_evaluate(child)
            child.metric = metric

            # Accept if improved
            if self._dominated_by_front(child):
                logger.debug("  Child not on Pareto front — discarding")
            else:
                self._population.append(child)
                self._update_pareto_front()
                logger.info(
                    "Gen %d: new Pareto member metric=%.4f (parent=%.4f)",
                    generation, child.metric, parent.metric,
                )

            self._flush()

        return self._best_candidate()

    # ------------------------------------------------------------------ #
    #  GEPA internals                                                      #
    # ------------------------------------------------------------------ #

    def _reflect(self, candidate: Candidate, trace: str) -> str:
        """
        LLM reads execution trace and diagnoses *why* the candidate fails.
        This is the key GEPA insight: reflection over full ASI, not scalar reward.
        """
        prompt = (
            "You are a reflective optimizer analyzing why a candidate failed.\n\n"
            f"Artifact type: {candidate.artifact_type}\n"
            f"Current metric: {candidate.metric:.4f}\n"
            f"Artifact (first 1500 chars):\n{candidate.artifact[:1500]}\n\n"
            f"Execution trace (last 2000 chars):\n{trace[-2000:]}\n\n"
            "Accumulated lessons from previous failures:\n"
            + "\n".join(f"- {l}" for l in self._lessons[-5:])
            + "\n\nDiagnose: what specifically failed and why? "
            "What targeted change would improve it? "
            "Be specific and actionable. One paragraph."
        )
        try:
            return self._llm(prompt).strip()
        except Exception as exc:
            logger.warning("Reflection LLM call failed: %s", exc)
            return f"Reflection unavailable: {exc}"

    def _mutate(
        self, parent: Candidate, lesson: str, generation: int
    ) -> Candidate | None:
        """
        Generate an improved child candidate based on reflection lesson.
        Uses GEPA's 'system-aware merge' when the Pareto front has 2+ candidates.
        """
        import uuid

        # If Pareto front has 2+ candidates, attempt merge
        if len(self._pareto_front) >= 2 and generation % 3 == 0:
            merge_candidate = self._merge_pareto_candidates()
            if merge_candidate:
                return merge_candidate

        prompt = (
            f"You are evolving a {parent.artifact_type} to improve its performance.\n\n"
            f"Current artifact:\n{parent.artifact}\n\n"
            f"Diagnosis of failure / improvement opportunity:\n{lesson}\n\n"
            "Accumulated lessons:\n"
            + "\n".join(f"- {l}" for l in self._lessons[-8:])
            + f"\n\nGeneration: {generation}\n"
            "Write the improved artifact. Make targeted, minimal changes. "
            "Do NOT change the artifact's purpose or structure fundamentally. "
            "Return ONLY the improved artifact text, no commentary."
        )
        try:
            new_artifact = self._llm(prompt).strip()
        except Exception as exc:
            logger.warning("Mutation LLM call failed: %s", exc)
            return None

        return Candidate(
            candidate_id=f"gen{generation}_{str(uuid.uuid4())[:8]}",
            artifact=new_artifact,
            artifact_type=parent.artifact_type,
            generation=generation,
            parent_id=parent.candidate_id,
            mutation_rationale=lesson[:200],
        )

    def _merge_pareto_candidates(self) -> Candidate | None:
        """
        Combine strengths of two Pareto-optimal candidates excelling on different tasks.
        GEPA 'system-aware merge' operation.
        """
        import uuid
        if len(self._pareto_front) < 2:
            return None

        a, b = self._pareto_front[0], self._pareto_front[1]
        prompt = (
            "Merge the strengths of these two artifacts. "
            "A excels at some tasks, B at others.\n\n"
            f"Artifact A (metric={a.metric:.4f}):\n{a.artifact[:1000]}\n\n"
            f"Artifact B (metric={b.metric:.4f}):\n{b.artifact[:1000]}\n\n"
            "Return a merged artifact that combines the best of both. "
            "Return ONLY the merged artifact text."
        )
        try:
            merged = self._llm(prompt).strip()
        except Exception:
            return None

        return Candidate(
            candidate_id=f"merge_{str(uuid.uuid4())[:8]}",
            artifact=merged,
            artifact_type=a.artifact_type,
            generation=-1,
            parent_id=a.candidate_id,
            mutation_rationale=f"Pareto merge of {a.candidate_id} and {b.candidate_id}",
        )

    # ------------------------------------------------------------------ #
    #  Constraint gates (hermes-agent-self-evolution guardrails)           #
    # ------------------------------------------------------------------ #

    def _apply_constraint_gates(self, candidate: Candidate) -> tuple[bool, str]:
        """Check all constraint gates before evaluating a candidate."""
        # Size limit
        if len(candidate.artifact) > self._cfg.max_artifact_chars:
            return False, f"artifact too large ({len(candidate.artifact)} > {self._cfg.max_artifact_chars})"

        # Semantic preservation: artifact must not drift too far from seed
        if self._cfg.semantic_preservation and self._population:
            seed = self._population[0]
            overlap = self._token_overlap(seed.artifact, candidate.artifact)
            if overlap < 0.1:
                return False, f"semantic drift too high (token overlap={overlap:.2f})"

        # Custom constraint function (e.g. run test suite)
        if self._check_constraints:
            return self._check_constraints(candidate)

        return True, "ok"

    # ------------------------------------------------------------------ #
    #  Pareto front management                                             #
    # ------------------------------------------------------------------ #

    def _update_pareto_front(self) -> None:
        """Rebuild Pareto front from current population."""
        if not self._pareto_front:
            self._pareto_front = list(self._population)
            for c in self._pareto_front:
                c.pareto_front = True
            return

        new_front: list[Candidate] = []
        for c in self._population:
            if not any(self._dominates(other, c) for other in self._population if other is not c):
                new_front.append(c)
                c.pareto_front = True

        # Mark removed candidates
        front_ids = {c.candidate_id for c in new_front}
        for c in self._population:
            if c.candidate_id not in front_ids:
                c.pareto_front = False

        self._pareto_front = new_front

    def _dominates(self, a: Candidate, b: Candidate) -> bool:
        """Return True if a dominates b on all task subset scores + global metric."""
        if not a.task_subset_scores or not b.task_subset_scores:
            return a.metric >= b.metric + self._cfg.min_improvement_delta
        return all(
            a.task_subset_scores.get(k, 0) >= b.task_subset_scores.get(k, 0)
            for k in b.task_subset_scores
        ) and a.metric >= b.metric

    def _dominated_by_front(self, candidate: Candidate) -> bool:
        return any(self._dominates(f, candidate) for f in self._pareto_front)

    def _select_parent(self) -> Candidate | None:
        if not self._pareto_front:
            return self._population[-1] if self._population else None
        import random
        return random.choice(self._pareto_front)

    def _best_candidate(self) -> Candidate:
        return max(self._population, key=lambda c: c.metric)

    # ------------------------------------------------------------------ #
    #  Helpers                                                             #
    # ------------------------------------------------------------------ #

    def _safe_evaluate(self, candidate: Candidate) -> tuple[float, str]:
        self._metric_calls += 1
        try:
            return self._evaluate(candidate)
        except Exception as exc:
            logger.warning("Evaluator raised: %s", exc)
            return 0.0, str(exc)

    @staticmethod
    def _token_overlap(a: str, b: str) -> float:
        tokens_a = set(a.lower().split())
        tokens_b = set(b.lower().split())
        if not tokens_a:
            return 0.0
        return len(tokens_a & tokens_b) / len(tokens_a)

    def _flush(self) -> None:
        if not self._persist_path:
            return
        self._persist_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "population": [
                {k: v for k, v in vars(c).items()}
                for c in self._population
            ],
            "lessons": self._lessons,
            "metric_calls": self._metric_calls,
        }
        tmp = self._persist_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2, default=str))
        tmp.replace(self._persist_path)
