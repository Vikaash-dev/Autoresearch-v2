"""
Core modify-evaluate-keep/discard loop for autonomous research.

Directly inspired by karpathy/autoresearch (March 2026):
  "give an AI agent a small but real setup and let it experiment autonomously.
   It modifies the code, evaluates the result, keeps or discards, and repeats."

Extended with:
  - BFTS (Best-First Tree Search) from SakanaAI/AI-Scientist v2
  - GEPA-style reflection: reads *why* something failed, not just *that* it failed
  - Self-ToM: agent monitors its own success rate and adapts
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Callable

from core.blackboard import Blackboard, ExperimentNode

logger = logging.getLogger(__name__)


@dataclass
class LoopConfig:
    max_iterations: int = 50
    time_budget_seconds: float = 3600.0   # 1 hour default
    min_improvement_delta: float = 0.001  # stop if gain < this
    patience: int = 10                    # stop after N iters with no improvement
    debug_on_fail: bool = True
    max_debug_depth: int = 3              # max times to retry a failing node
    debug_prob: float = 0.5              # probability of attempting debug vs prune


class ResearchLoop:
    """
    The fundamental autonomous research loop.

    Usage:
        loop = ResearchLoop(blackboard, config, propose_fn, evaluate_fn)
        best_node = loop.run()

    Callbacks:
        propose_fn(blackboard) -> ExperimentNode
            Called each iteration to propose the next experiment node.
        evaluate_fn(node, blackboard) -> (metric: float, error_trace: str)
            Executes the node and returns its metric (higher=better) or error.
        on_improvement_fn(node, blackboard) -> None   [optional]
            Called whenever a new best is found.
    """

    def __init__(
        self,
        blackboard: Blackboard,
        config: LoopConfig,
        propose_fn: Callable[[Blackboard], ExperimentNode],
        evaluate_fn: Callable[[ExperimentNode, Blackboard], tuple[float | None, str]],
        on_improvement_fn: Callable[[ExperimentNode, Blackboard], None] | None = None,
    ) -> None:
        self._bb = blackboard
        self._cfg = config
        self._propose = propose_fn
        self._evaluate = evaluate_fn
        self._on_improvement = on_improvement_fn
        self._no_improve_count = 0
        self._start_time: float = 0.0

    def run(self) -> ExperimentNode | None:
        """
        Run the modify-evaluate-keep/discard loop until a stopping criterion.

        Returns the best ExperimentNode found, or None if no successful node.
        """
        self._start_time = time.time()
        logger.info(
            "ResearchLoop starting: max_iters=%d, budget=%.0fs",
            self._cfg.max_iterations,
            self._cfg.time_budget_seconds,
        )

        for iteration in range(self._cfg.max_iterations):
            if self._should_stop(iteration):
                break

            logger.info("--- Iteration %d ---", iteration + 1)

            # 1. Propose next experiment
            try:
                node = self._propose(self._bb)
            except Exception as exc:
                logger.warning("propose_fn raised: %s — skipping iteration", exc)
                continue

            self._bb.add_node(node)

            # 2. Evaluate with optional debug retries
            node, improved = self._evaluate_with_retry(node)

            # 3. Keep or discard
            if improved:
                self._no_improve_count = 0
                logger.info(
                    "✓ New best: node=%s metric=%.4f", node.node_id, node.metric
                )
                if self._on_improvement:
                    self._on_improvement(node, self._bb)
            else:
                self._no_improve_count += 1
                logger.info(
                    "✗ No improvement (patience %d/%d)",
                    self._no_improve_count,
                    self._cfg.patience,
                )

        best_id = self._bb.state.best_node_id
        return self._bb.get_node(best_id) if best_id else None

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _evaluate_with_retry(
        self, node: ExperimentNode
    ) -> tuple[ExperimentNode, bool]:
        """
        Run evaluate_fn with up to max_debug_depth retries on failure.

        On failure, the error trace is fed back to propose_fn for a debug patch
        (GEPA-style reflection: *why* it failed informs the next mutation).
        """
        import random

        for attempt in range(self._cfg.max_debug_depth + 1):
            self._bb.update_node(node.node_id, status="running")
            metric, error_trace = self._evaluate(node, self._bb)

            if metric is not None:
                self._bb.update_node(
                    node.node_id,
                    metric=metric,
                    status="success",
                    error_trace="",
                )
                improved = (
                    self._bb.state.best_metric is None
                    or metric > self._bb.state.best_metric + self._cfg.min_improvement_delta
                )
                if improved:
                    self._bb.update_best(node.node_id, metric)
                return node, improved

            # Failed — decide whether to debug or prune
            self._bb.update_node(
                node.node_id,
                status="failed",
                error_trace=error_trace,
            )

            if (
                attempt < self._cfg.max_debug_depth
                and self._cfg.debug_on_fail
                and random.random() < self._cfg.debug_prob
            ):
                logger.info(
                    "  Debug attempt %d for node %s", attempt + 1, node.node_id
                )
                # Create a child debug node — propose_fn will see the error trace
                # on the blackboard and generate a targeted fix
                debug_node = self._propose(self._bb)
                debug_node.parent_id = node.node_id
                debug_node.depth = node.depth + 1
                self._bb.add_node(debug_node)
                node = debug_node
            else:
                self._bb.update_node(node.node_id, status="pruned")
                break

        return node, False

    def _should_stop(self, iteration: int) -> bool:
        elapsed = time.time() - self._start_time
        if elapsed >= self._cfg.time_budget_seconds:
            logger.info("Time budget exhausted after %.0fs", elapsed)
            return True
        if self._no_improve_count >= self._cfg.patience:
            logger.info("Early stopping: no improvement for %d iters", self._cfg.patience)
            return True
        return False

    @property
    def elapsed(self) -> float:
        return time.time() - self._start_time if self._start_time else 0.0
