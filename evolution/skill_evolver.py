"""
Skill Evolver — runs GEPA optimization across all four hermes-agent-self-evolution phases.

Phases (from NousResearch/hermes-agent-self-evolution, ICLR 2026 Oral):
  Phase 1: Skill text files     — evolve SKILL.md prompt artifacts
  Phase 2: Tool descriptions    — evolve Tool-ToM metadata (good_at, bad_at, etc.)
  Phase 3: System prompts       — evolve per-agent _SYSTEM constants
  Phase 4: Code snippets        — evolve small self-contained code functions

Each phase wraps GEPAOptimizer with a phase-appropriate:
  - artifact_type label
  - evaluator function  (how to measure quality of this artifact type)
  - constraint function (what must remain true after evolution)

The SkillEvolver is triggered by the Orchestrator after each run when Self-ToM
reports repeated failures on a specific subtask (failure_count ≥ 3).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable

from evolution.gepa_optimizer import GEPAOptimizer, GEPAConfig, Candidate
from memory.skill_store import SkillStore
from memory.trajectory_log import TrajectoryLog

logger = logging.getLogger(__name__)

# ── Phase labels ──────────────────────────────────────────────────────── #
PHASE_SKILL = "skill"
PHASE_TOOL_DESC = "tool_desc"
PHASE_PROMPT = "prompt"
PHASE_CODE = "code"


class SkillEvolver:
    """
    Orchestrates GEPA-based evolution across all four artifact phases.

    Usage:
        evolver = SkillEvolver(skill_store, trajectory_log, llm_fn)
        new_artifact = evolver.evolve(
            name="hypothesis_system_prompt",
            artifact_type=PHASE_PROMPT,
            eval_dataset=[...],
        )
    """

    def __init__(
        self,
        skill_store: SkillStore,
        trajectory_log: TrajectoryLog,
        llm_fn: Callable[[str], str],
        gepa_config: GEPAConfig | None = None,
        output_dir: Path | None = None,
    ) -> None:
        self._store = skill_store
        self._log = trajectory_log
        self._llm = llm_fn
        self._cfg = gepa_config or GEPAConfig()
        self._output_dir = output_dir or Path("experiments/evolution")
        self._output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    #  Public API                                                           #
    # ------------------------------------------------------------------ #

    def evolve(
        self,
        name: str,
        artifact_type: str,
        eval_dataset: list[dict[str, Any]],
        agent_fn: Callable[[str, dict[str, Any]], dict[str, Any]] | None = None,
    ) -> str:
        """
        Evolve a named artifact using GEPA.

        Args:
            name:          skill name in the SkillStore
            artifact_type: one of PHASE_SKILL, PHASE_TOOL_DESC, PHASE_PROMPT, PHASE_CODE
            eval_dataset:  list of task examples used to evaluate candidates
            agent_fn:      optional callable(artifact, context) → result dict;
                           if None, uses a text-quality heuristic evaluator

        Returns:
            The evolved artifact text (also saved to SkillStore).
        """
        # Load current artifact from skill store
        seed = self._store.load_artifact(name)
        if seed is None:
            raise ValueError(
                f"Skill {name!r} not found in SkillStore. "
                "Register it first with SkillStore.save()."
            )

        logger.info(
            "SkillEvolver: evolving %r (type=%s, dataset_size=%d)",
            name, artifact_type, len(eval_dataset),
        )

        # Build appropriate evaluator for this phase
        evaluator = self._build_evaluator(artifact_type, eval_dataset, agent_fn)

        # Build appropriate constraint checker for this phase
        constraint = self._build_constraint(artifact_type, seed)

        optimizer = GEPAOptimizer(
            config=self._cfg,
            llm_fn=self._llm,
            evaluator_fn=evaluator,
            constraint_fn=constraint,
            persist_path=self._output_dir / f"gepa_{name}.json",
        )

        best = optimizer.optimize(seed, artifact_type=artifact_type)

        # Persist improved artifact back to the skill store
        self._store.save(
            name=name,
            artifact=best.artifact,
            artifact_type=artifact_type,
            metric=best.metric,
            description=f"GEPA-evolved v{best.generation} (metric={best.metric:.4f})",
            tags=["gepa", artifact_type],
        )

        logger.info(
            "SkillEvolver: evolved %r — metric %.4f → %.4f",
            name,
            optimizer._population[0].metric if optimizer._population else 0.0,
            best.metric,
        )

        # Log the evolution event
        self._log.log_event(
            run_id="evolution",
            agent_id="skill_evolver",
            agent_role="evolution",
            action=f"evolve_{artifact_type}",
            inputs={"name": name, "seed_len": len(seed), "dataset_size": len(eval_dataset)},
            outputs={"best_metric": best.metric, "generations": best.generation},
            metric=best.metric,
            status="success",
        )

        return best.artifact

    def evolve_from_failures(
        self,
        failure_analysis: dict[str, Any],
        agent_skill_map: dict[str, tuple[str, str]],
        eval_datasets: dict[str, list[dict[str, Any]]],
        agent_fns: dict[str, Callable] | None = None,
    ) -> dict[str, str]:
        """
        Evolve all skills where failure analysis identifies repeated failures.

        Args:
            failure_analysis:  output of TrajectoryLog.analyze_failures()
            agent_skill_map:   {action_name: (skill_name, artifact_type)}
            eval_datasets:     {skill_name: [eval examples]}
            agent_fns:         optional {skill_name: callable} for evaluation

        Returns:
            {skill_name: evolved_artifact} for every skill that was evolved.
        """
        worst_actions = failure_analysis.get("worst_actions", [])
        evolved: dict[str, str] = {}

        for action in worst_actions:
            if action not in agent_skill_map:
                logger.debug("No skill mapped for action %r — skipping", action)
                continue

            skill_name, artifact_type = agent_skill_map[action]
            dataset = eval_datasets.get(skill_name, [])
            if not dataset:
                logger.warning("No eval dataset for skill %r — skipping evolution", skill_name)
                continue

            agent_fn = (agent_fns or {}).get(skill_name)
            try:
                artifact = self.evolve(
                    name=skill_name,
                    artifact_type=artifact_type,
                    eval_dataset=dataset,
                    agent_fn=agent_fn,
                )
                evolved[skill_name] = artifact
            except Exception as exc:
                logger.error("Evolution of %r failed: %s", skill_name, exc)

        return evolved

    # ------------------------------------------------------------------ #
    #  Phase-specific evaluators                                            #
    # ------------------------------------------------------------------ #

    def _build_evaluator(
        self,
        artifact_type: str,
        eval_dataset: list[dict[str, Any]],
        agent_fn: Callable | None,
    ) -> Callable[[Candidate], tuple[float, str]]:
        """Return a GEPA evaluator appropriate for the artifact type."""

        if agent_fn is not None:
            # Use the provided agent function to measure task performance
            def _agent_evaluator(candidate: Candidate) -> tuple[float, str]:
                scores, traces = [], []
                for example in eval_dataset[:10]:
                    try:
                        result = agent_fn(candidate.artifact, example)
                        scores.append(float(result.get("metric", 0.5)))
                        traces.append(str(result))
                    except Exception as exc:
                        scores.append(0.0)
                        traces.append(str(exc))
                return (
                    sum(scores) / max(len(scores), 1),
                    "\n".join(traces[-5:]),
                )
            return _agent_evaluator

        if artifact_type == PHASE_CODE:
            # For code: compile + run with test inputs, measure pass rate
            return self._code_evaluator(eval_dataset)

        if artifact_type == PHASE_TOOL_DESC:
            # For tool descriptions: measure retrieval accuracy (does it retrieve when relevant?)
            return self._tool_desc_evaluator(eval_dataset)

        # Default: text-quality heuristic for prompts and skill texts
        return self._text_quality_evaluator(eval_dataset)

    def _text_quality_evaluator(
        self, eval_dataset: list[dict[str, Any]]
    ) -> Callable[[Candidate], tuple[float, str]]:
        """
        Heuristic evaluator for prompt/skill text artifacts.
        Scores on: length sufficiency, keyword coverage, structural clarity.
        """
        def _evaluate(candidate: Candidate) -> tuple[float, str]:
            text = candidate.artifact
            score = 0.0
            trace_parts = []

            # Length: 100–3000 chars is healthy
            if 100 <= len(text) <= 3000:
                score += 0.3
                trace_parts.append(f"length={len(text)} ✓")
            else:
                trace_parts.append(f"length={len(text)} ✗ (should be 100–3000)")

            # Structure: has numbered/bulleted points
            has_structure = any(c in text for c in ["1.", "2.", "-", "•", "*"])
            if has_structure:
                score += 0.2
                trace_parts.append("structure ✓")
            else:
                trace_parts.append("structure ✗ (no numbered/bulleted points)")

            # Coverage: expected keywords from dataset
            all_keywords: set[str] = set()
            for ex in eval_dataset[:5]:
                for v in ex.values():
                    if isinstance(v, str):
                        all_keywords.update(v.lower().split()[:10])
            coverage = sum(1 for kw in all_keywords if kw in text.lower())
            cov_score = min(coverage / max(len(all_keywords), 1), 1.0) * 0.3
            score += cov_score
            trace_parts.append(f"keyword_coverage={cov_score:.2f}")

            # LLM self-assessment (if available)
            try:
                assessment_prompt = (
                    f"Rate this {candidate.artifact_type} artifact from 0.0 to 1.0. "
                    "Return only a float.\n\nArtifact:\n" + text[:800]
                )
                raw = self._llm(assessment_prompt).strip()
                llm_score = float(raw.split()[0]) * 0.2
                score += llm_score
                trace_parts.append(f"llm_score={llm_score:.2f}")
            except Exception:
                trace_parts.append("llm_score=unavailable")

            return round(min(score, 1.0), 4), " | ".join(trace_parts)

        return _evaluate

    def _code_evaluator(
        self, eval_dataset: list[dict[str, Any]]
    ) -> Callable[[Candidate], tuple[float, str]]:
        """Evaluate a code artifact by running it against test inputs."""
        import subprocess, tempfile, os

        def _evaluate(candidate: Candidate) -> tuple[float, str]:
            passed, total = 0, 0
            traces = []
            for ex in eval_dataset[:8]:
                test_input = ex.get("input", "")
                expected_output = str(ex.get("expected_output", ""))
                wrapper = (
                    candidate.artifact + "\n\n"
                    f"result = main({test_input!r})\n"
                    "print('OUTPUT:', result)\n"
                )
                with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
                    f.write(wrapper)
                    fpath = f.name
                try:
                    proc = subprocess.run(
                        ["python", fpath], capture_output=True, text=True, timeout=10
                    )
                    output = proc.stdout.strip()
                    passed_this = expected_output.lower() in output.lower()
                    passed += int(passed_this)
                    total += 1
                    traces.append(
                        f"input={test_input!r} → {'✓' if passed_this else '✗'} got={output[:80]}"
                    )
                except Exception as exc:
                    total += 1
                    traces.append(f"input={test_input!r} → ERROR: {exc}")
                finally:
                    os.unlink(fpath)

            score = passed / max(total, 1)
            return round(score, 4), "\n".join(traces)

        return _evaluate

    def _tool_desc_evaluator(
        self, eval_dataset: list[dict[str, Any]]
    ) -> Callable[[Candidate], tuple[float, str]]:
        """
        Evaluate a tool description by checking if it correctly
        predicts when the tool should and should not be used.
        """
        def _evaluate(candidate: Candidate) -> tuple[float, str]:
            import json as _json
            desc = candidate.artifact
            scores, traces = [], []
            for ex in eval_dataset[:10]:
                task = ex.get("task", "")
                should_use = ex.get("should_use", True)
                # Check if the description's good_at/bad_at keywords match the task
                try:
                    data = _json.loads(desc)
                    good_at = " ".join(data.get("good_at", [])).lower()
                    bad_at = " ".join(data.get("bad_at", [])).lower()
                    task_lower = task.lower()
                    predicted_use = any(kw in task_lower for kw in good_at.split())
                    predicted_not_use = any(kw in task_lower for kw in bad_at.split())
                    correct = (predicted_use and should_use) or (predicted_not_use and not should_use)
                    scores.append(1.0 if correct else 0.0)
                    traces.append(f"task={task!r} should_use={should_use} correct={correct}")
                except Exception as exc:
                    scores.append(0.0)
                    traces.append(f"parse_error: {exc}")
            score = sum(scores) / max(len(scores), 1)
            return round(score, 4), "\n".join(traces)

        return _evaluate

    # ------------------------------------------------------------------ #
    #  Phase-specific constraints                                            #
    # ------------------------------------------------------------------ #

    def _build_constraint(
        self, artifact_type: str, seed: str
    ) -> Callable[[Candidate], tuple[bool, str]]:
        """Return a constraint checker appropriate for the artifact type."""

        if artifact_type == PHASE_CODE:
            def _code_constraint(candidate: Candidate) -> tuple[bool, str]:
                # Must be valid Python
                try:
                    compile(candidate.artifact, "<string>", "exec")
                    return True, "ok"
                except SyntaxError as exc:
                    return False, f"SyntaxError: {exc}"
            return _code_constraint

        if artifact_type == PHASE_TOOL_DESC:
            def _json_constraint(candidate: Candidate) -> tuple[bool, str]:
                import json as _json
                try:
                    data = _json.loads(candidate.artifact)
                    required = {"good_at", "bad_at", "description"}
                    missing = required - set(data.keys())
                    if missing:
                        return False, f"Missing required keys: {missing}"
                    return True, "ok"
                except Exception as exc:
                    return False, f"Invalid JSON: {exc}"
            return _json_constraint

        # Default: size and semantic preservation only (handled by GEPAConfig)
        return lambda c: (True, "ok")
