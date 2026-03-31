"""
Best-First Tree Search (BFTS) over the hypothesis/experiment space.

Synthesises:
  - SakanaAI/AI-Scientist v2 BFTS (stage-gated parallel tree expansion)
  - WecoAI/AIDE (LLM-guided agentic tree search, 4x more medals than linear agents)
  - karpathy/autoresearch (fixed time budget, keep/discard loop as tree nodes)

The tree is stored on the Blackboard so all agents share the same view.
"""

from __future__ import annotations

import heapq
import logging
import threading
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

from core.blackboard import Blackboard, ExperimentNode

logger = logging.getLogger(__name__)


@dataclass
class BFTSConfig:
    num_workers: int = 4          # parallel expansion workers (AI-Scientist default)
    num_seeds: int = 3            # number of independent root nodes (Stage 1 drafts)
    max_nodes: int = 50           # hard cap on total nodes explored
    max_depth: int = 8            # maximum tree depth before forcing writeup
    stages: dict[str, int] = field(
        default_factory=lambda: {
            "stage1_max_iters": 20,  # baseline establishment
            "stage2_max_iters": 12,  # ablations
            "stage3_max_iters": 12,  # full experiments
            "stage4_max_iters": 18,  # final tuning + writeup
        }
    )
    exploration_weight: float = 1.41  # UCB-style exploration constant
    prune_threshold: float = 0.0      # prune nodes below this metric
    debug_prob: float = 0.5
    max_debug_depth: int = 3


class BestFirstTreeSearch:
    """
    Parallel BFTS over experiment nodes.

    The priority queue scores each frontier node by:
        score = metric + exploration_weight * sqrt(log(N) / n_i)
    where N = total visits, n_i = visits to this node's subtree.

    This is a UCB1-style balance between exploitation (high metric nodes)
    and exploration (under-visited branches) — exactly the "progressive
    agentic tree search" described in AI-Scientist v2.
    """

    def __init__(
        self,
        blackboard: Blackboard,
        config: BFTSConfig,
        expand_fn: Callable[[ExperimentNode, Blackboard], list[ExperimentNode]],
        evaluate_fn: Callable[[ExperimentNode, Blackboard], tuple[float | None, str]],
    ) -> None:
        self._bb = blackboard
        self._cfg = config
        self._expand = expand_fn      # generates child hypothesis nodes
        self._evaluate = evaluate_fn  # runs the experiment and returns metric
        self._heap: list[tuple[float, str]] = []  # (neg_priority, node_id)
        self._lock = threading.Lock()
        self._visits: dict[str, int] = {}
        self._total_visits = 0

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def run(self) -> ExperimentNode | None:
        """Execute BFTS and return the best node found."""
        logger.info(
            "BFTS starting: workers=%d seeds=%d max_nodes=%d",
            self._cfg.num_workers,
            self._cfg.num_seeds,
            self._cfg.max_nodes,
        )

        # Stage 1: seed the tree with num_seeds independent root nodes
        roots = self._seed_roots()
        if not roots:
            logger.error("No seed nodes could be evaluated — aborting")
            return None

        # Stage 2-4: expand frontier nodes using parallel workers
        self._expand_loop()

        best_id = self._bb.state.best_node_id
        return self._bb.get_node(best_id) if best_id else None

    # ------------------------------------------------------------------ #
    #  Seeding                                                             #
    # ------------------------------------------------------------------ #

    def _seed_roots(self) -> list[ExperimentNode]:
        """Create and evaluate num_seeds independent root nodes (Stage 1 drafts)."""
        roots: list[ExperimentNode] = []
        for i in range(self._cfg.num_seeds):
            root = ExperimentNode(
                node_id=f"root_{i}_{str(uuid.uuid4())[:6]}",
                depth=0,
                stage=1,
            )
            self._bb.add_node(root)
            metric, error = self._evaluate(root, self._bb)
            if metric is not None:
                self._bb.update_node(root.node_id, metric=metric, status="success")
                if (
                    self._bb.state.best_metric is None
                    or metric > self._bb.state.best_metric
                ):
                    self._bb.update_best(root.node_id, metric)
                self._push(root.node_id, metric)
                roots.append(root)
            else:
                self._bb.update_node(root.node_id, status="failed", error_trace=error)
                logger.warning("Seed root_%d failed: %s", i, error[:120])
        return roots

    # ------------------------------------------------------------------ #
    #  Main expansion loop                                                 #
    # ------------------------------------------------------------------ #

    def _expand_loop(self) -> None:
        """Single-threaded expansion loop (parallelism handled by caller via workers)."""
        while self._bb.state.total_nodes_explored < self._cfg.max_nodes:
            node_id = self._pop()
            if node_id is None:
                logger.info("Frontier exhausted — stopping BFTS")
                break

            parent = self._bb.get_node(node_id)
            if parent is None or parent.depth >= self._cfg.max_depth:
                continue

            # Expand: generate children hypotheses
            children = self._expand(parent, self._bb)
            if not children:
                continue

            for child in children[: self._cfg.num_workers]:
                self._bb.add_node(child)
                metric, error = self._evaluate(child, self._bb)
                self._total_visits += 1
                self._visits[child.node_id] = self._visits.get(child.node_id, 0) + 1

                if metric is not None and metric > self._cfg.prune_threshold:
                    self._bb.update_node(child.node_id, metric=metric, status="success")
                    if (
                        self._bb.state.best_metric is None
                        or metric > self._bb.state.best_metric
                    ):
                        self._bb.update_best(child.node_id, metric)
                    priority = self._ucb_priority(child.node_id, metric)
                    self._push(child.node_id, priority)
                else:
                    self._bb.update_node(
                        child.node_id, status="failed", error_trace=error
                    )

    # ------------------------------------------------------------------ #
    #  Priority queue helpers                                              #
    # ------------------------------------------------------------------ #

    def _ucb_priority(self, node_id: str, metric: float) -> float:
        import math
        n_total = max(1, self._total_visits)
        n_node = max(1, self._visits.get(node_id, 1))
        exploration = self._cfg.exploration_weight * math.sqrt(
            math.log(n_total) / n_node
        )
        return metric + exploration

    def _push(self, node_id: str, priority: float) -> None:
        with self._lock:
            heapq.heappush(self._heap, (-priority, node_id))

    def _pop(self) -> str | None:
        with self._lock:
            while self._heap:
                _, node_id = heapq.heappop(self._heap)
                node = self._bb.get_node(node_id)
                if node and node.status == "success":
                    return node_id
            return None

    # ------------------------------------------------------------------ #
    #  Visualisation                                                        #
    # ------------------------------------------------------------------ #

    def render_ascii(self) -> str:
        """Render a simple ASCII tree of all nodes for logging/debugging."""
        lines: list[str] = []
        nodes = self._bb.get_all_nodes()
        roots = [n for n in nodes if n.parent_id is None]

        def _render(node: ExperimentNode, prefix: str, is_last: bool) -> None:
            connector = "└── " if is_last else "├── "
            metric_str = f"{node.metric:.4f}" if node.metric is not None else "N/A"
            status_sym = {"success": "✓", "failed": "✗", "running": "⟳", "pruned": "✂"}.get(
                node.status, "?"
            )
            lines.append(
                f"{prefix}{connector}[{node.node_id}] {status_sym} metric={metric_str} "
                f"depth={node.depth} stage={node.stage}"
            )
            child_prefix = prefix + ("    " if is_last else "│   ")
            children = [self._bb.get_node(cid) for cid in node.children if self._bb.get_node(cid)]
            for i, child in enumerate(children):
                if child:
                    _render(child, child_prefix, i == len(children) - 1)

        for i, root in enumerate(roots):
            _render(root, "", i == len(roots) - 1)

        return "\n".join(lines)
