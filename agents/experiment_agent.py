"""
Experiment Agent — executes experiments and evaluates results.

Implements the BFTS evaluate_fn:
  Given an ExperimentNode, generate the code, execute it (sandboxed),
  parse the metric from stdout, and return (metric, error_trace).

Inspired by:
  - AI-Scientist v2 code execution worker
  - karpathy/autoresearch train.py modification + eval loop
  - AIDE "write → execute → evaluate → debug" per node
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

from agents.base_agent import BaseAgent
from core.blackboard import Blackboard, ExperimentNode
from tom.engine import TheoryOfMindEngine

logger = logging.getLogger(__name__)

_EXPERIMENT_SYSTEM = """You are an expert ML engineer writing clean, self-contained experiment code.
The code must:
1. Import only standard Python packages and common ML libraries (torch, numpy, sklearn)
2. Print a line matching: METRIC: <float>  (this is parsed as the evaluation metric)
3. Complete within the time budget (use small models/datasets for initial experiments)
4. Handle errors gracefully and print informative tracebacks
5. Be self-contained — no file paths outside the working directory
"""

_METRIC_PARSE_PREFIX = "METRIC:"


class ExperimentAgent(BaseAgent):
    """Executes experiments and parses their metric."""

    def __init__(
        self,
        blackboard: Blackboard,
        tom_engine: TheoryOfMindEngine,
        llm_fn: Any = None,
        exec_timeout: int = 300,           # 5 minutes per experiment
        work_dir: str | None = None,
    ) -> None:
        super().__init__("experiment", blackboard, tom_engine, llm_fn)
        self._exec_timeout = exec_timeout
        self._work_dir = Path(work_dir) if work_dir else Path(tempfile.mkdtemp(prefix="ar2_"))

    def act(self, context: dict[str, Any]) -> dict[str, Any]:
        """Top-level act: generate code for a hypothesis and run it."""
        node = context.get("node")
        if node is None:
            return {"status": "failed", "error": "No node in context"}

        metric, error = self.evaluate(node, self._bb)
        return {
            "status": "ok" if metric is not None else "failed",
            "metric": metric,
            "error": error,
        }

    def evaluate(
        self, node: ExperimentNode, blackboard: Blackboard
    ) -> tuple[float | None, str]:
        """
        BFTS evaluate_fn: execute experiment for node and return metric.

        Returns:
            (metric, "") on success
            (None, error_trace) on failure
        """
        # Step 1: Generate experiment code from hypothesis
        code = self._generate_code(node)
        if not code:
            self.record_failure("generate_code", "LLM returned empty code")
            return None, "Code generation produced empty result"

        # Step 2: Execute in sandbox
        metric, trace = self._execute_code(code, node.node_id)

        if metric is not None:
            self.record_success("evaluate", metric)
        else:
            self.record_failure("evaluate", trace[:120])

        return metric, trace

    # ------------------------------------------------------------------ #
    #  Code generation                                                     #
    # ------------------------------------------------------------------ #

    def _generate_code(self, node: ExperimentNode) -> str:
        """Generate self-contained experiment code from the hypothesis."""
        # Gather context from blackboard for GEPA-style reflection
        recent_failures = self.get_recent_failures_from_board()
        best_nodes = self._bb.get_best_nodes(n=2)
        best_summary = [
            f"node={n.node_id} metric={n.metric:.4f} hypothesis={n.hypothesis[:80]}"
            for n in best_nodes
        ]

        idea = self._bb.state.idea

        prompt = (
            f"Research idea: {idea.get('Title', 'Unknown')}\n"
            f"Goal: {idea.get('Abstract', '')[:300]}\n\n"
            f"Hypothesis to test: {node.hypothesis}\n"
            f"Code guidance: {node.code_patch}\n"
            f"Stage: {node.stage} (1=baseline, 2=ablation, 3=full, 4=final)\n\n"
            f"Best experiments so far:\n{json.dumps(best_summary)}\n\n"
            f"Recent failures to avoid:\n{json.dumps(recent_failures[:3])}\n\n"
            "Write a complete Python experiment script. "
            f"It MUST print exactly one line: '{_METRIC_PARSE_PREFIX} <float>' "
            "where the float is the primary evaluation metric (higher=better). "
            "Keep it small and fast for stages 1-2. "
            "Return ONLY the Python code, no markdown."
        )

        if self._llm is None:
            return self._dry_run_code(node)

        try:
            code = self.call_llm(prompt, system=_EXPERIMENT_SYSTEM)
            # Strip markdown fences
            if "```python" in code:
                code = code.split("```python")[1].split("```")[0]
            elif "```" in code:
                code = code.split("```")[1].split("```")[0]
            return code.strip()
        except Exception as exc:
            logger.error("Code generation failed: %s", exc)
            return ""

    def _dry_run_code(self, node: ExperimentNode) -> str:
        """
        Deterministic experiment code used when no LLM is configured (dry-run mode).

        The metric is derived from the node ID hash so:
          - the same node always produces the same metric (reproducible)
          - different nodes produce different metrics (realistic spread)
          - no random state, no side effects
        """
        # Hash the node_id to get a stable float in [0.50, 0.95]
        node_hash = int(hashlib.md5(node.node_id.encode()).hexdigest(), 16)
        metric = round(0.50 + (node_hash % 10_000) / 22_222, 4)  # maps to [0.50, 0.95)
        hypothesis_safe = node.hypothesis[:60].replace('"', "'")
        return (
            "# Dry-run experiment (no LLM configured)\n"
            f'# Hypothesis: "{hypothesis_safe}"\n'
            f"metric = {metric}  # deterministic from node_id hash\n"
            f"print('METRIC:', metric)\n"
        )

    # ------------------------------------------------------------------ #
    #  Sandboxed execution                                                 #
    # ------------------------------------------------------------------ #

    def _execute_code(
        self, code: str, node_id: str
    ) -> tuple[float | None, str]:
        """
        Execute code in a subprocess (sandboxed) and parse the METRIC: line.
        """
        work_dir = self._work_dir / node_id
        work_dir.mkdir(parents=True, exist_ok=True)

        code_path = work_dir / "runfile.py"
        code_path.write_text(code)

        env = {**os.environ, "PYTHONPATH": str(work_dir)}

        start = time.time()
        try:
            result = subprocess.run(
                ["python", str(code_path)],
                capture_output=True,
                text=True,
                timeout=self._exec_timeout,
                cwd=str(work_dir),
                env=env,
            )
        except subprocess.TimeoutExpired:
            elapsed = time.time() - start
            error = f"TimeoutExpired after {elapsed:.0f}s"
            logger.warning("Experiment %s timed out", node_id)
            return None, error
        except Exception as exc:
            return None, str(exc)

        stdout = result.stdout or ""
        stderr = result.stderr or ""
        combined = stdout + ("\n" + stderr if stderr else "")

        # Parse metric
        metric = self._parse_metric(stdout)
        if metric is not None:
            return metric, ""

        # If no metric printed → failure; return combined output as trace
        return None, combined[-3000:]  # last 3000 chars for GEPA reflection

    @staticmethod
    def _parse_metric(stdout: str) -> float | None:
        """Parse the 'METRIC: <float>' line from stdout."""
        for line in reversed(stdout.splitlines()):
            line = line.strip()
            if line.startswith(_METRIC_PARSE_PREFIX):
                try:
                    return float(line[len(_METRIC_PARSE_PREFIX):].strip())
                except ValueError:
                    continue  # line matched prefix but value is not a float; try the next
        return None
