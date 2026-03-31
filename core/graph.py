"""
ResearchGraph — typed, directed graph over all knowledge produced in a research run.

Nodes represent entities (papers, hypotheses, experiments, findings, concepts).
Edges represent typed relationships (supports, contradicts, extends, cites, derives_from).

Used by:
  - Blackboard: attaches the graph alongside its flat node list so agents see relations
  - LiteratureAgent: adds paper → concept edges, paper ← cites ← paper edges
  - HypothesisAgent: adds hypothesis → derives_from → paper edges
  - ExperimentAgent: adds experiment → tests → hypothesis edges
  - WriterAgent: traverses graph to extract citation chains and finding flow
  - Community-ToM: queries graph for consensus / active-debate structure

Design lineage:
  - AI-Scientist v2 "unified_tree_viz" (visual graph of the search tree)
  - Voyager's skill graph (skills as nodes, dependencies as edges)
  - NousResearch/hermes-agent FTS5 memory graph
  - karpathy/autoresearch "nanochat parent repository" tree structure
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Any, Iterator


# ======================================================================= #
#  Node types                                                               #
# ======================================================================= #

class NodeType(str, Enum):
    PAPER        = "paper"         # arXiv / S2 paper
    HYPOTHESIS   = "hypothesis"    # agent-generated hypothesis
    EXPERIMENT   = "experiment"    # BFTS experiment node
    FINDING      = "finding"       # confirmed experimental result
    CONCEPT      = "concept"       # abstract concept / keyword
    SKILL        = "skill"         # evolved agent skill
    TOOL         = "tool"          # registered tool
    PLAN         = "plan"          # orchestrator plan step


class EdgeType(str, Enum):
    CITES        = "cites"         # paper → cites → paper
    SUPPORTS     = "supports"      # finding / experiment → supports → hypothesis
    CONTRADICTS  = "contradicts"   # finding → contradicts → hypothesis
    EXTENDS      = "extends"       # hypothesis → extends → paper/concept
    DERIVES_FROM = "derives_from"  # hypothesis → derives_from → paper
    TESTS        = "tests"         # experiment → tests → hypothesis
    PRODUCES     = "produces"      # experiment → produces → finding
    USES         = "uses"          # experiment/agent → uses → tool/skill
    IS_ABOUT     = "is_about"      # paper/hypothesis → is_about → concept
    PART_OF      = "part_of"       # plan_step → part_of → plan
    EVOLVED_FROM = "evolved_from"  # skill → evolved_from → skill (GEPA lineage)


# ======================================================================= #
#  Data classes                                                             #
# ======================================================================= #

@dataclass
class GraphNode:
    """A vertex in the ResearchGraph."""

    node_id: str = field(default_factory=lambda: str(uuid.uuid4())[:10])
    node_type: NodeType = NodeType.CONCEPT
    label: str = ""            # short human-readable label
    content: str = ""          # full text content (abstract, code, prompt, etc.)
    metadata: dict[str, Any] = field(default_factory=dict)
    score: float | None = None # quality / relevance score when applicable
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def __hash__(self) -> int:
        return hash(self.node_id)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, GraphNode):
            return self.node_id == other.node_id
        return NotImplemented


@dataclass
class GraphEdge:
    """A directed, typed edge between two GraphNodes."""

    edge_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    source_id: str = ""
    target_id: str = ""
    edge_type: EdgeType = EdgeType.IS_ABOUT
    weight: float = 1.0        # strength of the relationship (0–1)
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)


# ======================================================================= #
#  ResearchGraph                                                            #
# ======================================================================= #

class ResearchGraph:
    """
    Thread-safe directed multigraph over research knowledge.

    Core operations are O(1) amortized using adjacency sets.
    Provides typed traversal, subgraph extraction, ASCII rendering,
    and JSON serialisation for persistence on the Blackboard.

    Example:
        g = ResearchGraph()
        paper = g.add_node(NodeType.PAPER, "Attention Is All You Need",
                           content="...", metadata={"url": "..."})
        hyp   = g.add_node(NodeType.HYPOTHESIS, "Sparse attention improves efficiency")
        g.add_edge(hyp.node_id, paper.node_id, EdgeType.DERIVES_FROM)
        findings = g.neighbors(paper.node_id, edge_type=EdgeType.CITES)
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._nodes: dict[str, GraphNode] = {}
        # Adjacency: source_id → {edge_id} (outgoing)
        self._out_edges: dict[str, set[str]] = {}
        # Adjacency: target_id → {edge_id} (incoming)
        self._in_edges: dict[str, set[str]] = {}
        self._edges: dict[str, GraphEdge] = {}

    # ------------------------------------------------------------------ #
    #  Node operations                                                     #
    # ------------------------------------------------------------------ #

    def add_node(
        self,
        node_type: NodeType,
        label: str,
        content: str = "",
        metadata: dict[str, Any] | None = None,
        score: float | None = None,
        node_id: str | None = None,
    ) -> GraphNode:
        """Add a new node. Returns the created GraphNode."""
        node = GraphNode(
            node_id=node_id or str(uuid.uuid4())[:10],
            node_type=node_type,
            label=label,
            content=content,
            metadata=metadata or {},
            score=score,
        )
        with self._lock:
            self._nodes[node.node_id] = node
            self._out_edges.setdefault(node.node_id, set())
            self._in_edges.setdefault(node.node_id, set())
        return node

    def get_node(self, node_id: str) -> GraphNode | None:
        with self._lock:
            return self._nodes.get(node_id)

    def update_node(self, node_id: str, **kwargs: Any) -> None:
        with self._lock:
            node = self._nodes.get(node_id)
            if node is None:
                raise KeyError(f"Node {node_id!r} not in graph")
            for k, v in kwargs.items():
                setattr(node, k, v)
            node.updated_at = time.time()

    def remove_node(self, node_id: str) -> None:
        """Remove a node and all its edges."""
        with self._lock:
            if node_id not in self._nodes:
                return
            # Remove all edges touching this node
            for eid in list(self._out_edges.get(node_id, set())):
                self._remove_edge_unsafe(eid)
            for eid in list(self._in_edges.get(node_id, set())):
                self._remove_edge_unsafe(eid)
            del self._nodes[node_id]
            self._out_edges.pop(node_id, None)
            self._in_edges.pop(node_id, None)

    def find_nodes(
        self,
        node_type: NodeType | None = None,
        label_contains: str | None = None,
        min_score: float | None = None,
    ) -> list[GraphNode]:
        """Search nodes by type, label substring, and/or minimum score."""
        with self._lock:
            results = []
            for node in self._nodes.values():
                if node_type and node.node_type != node_type:
                    continue
                if label_contains and label_contains.lower() not in node.label.lower():
                    continue
                if min_score is not None and (node.score is None or node.score < min_score):
                    continue
                results.append(node)
            return sorted(results, key=lambda n: n.score or 0.0, reverse=True)

    # ------------------------------------------------------------------ #
    #  Edge operations                                                     #
    # ------------------------------------------------------------------ #

    def add_edge(
        self,
        source_id: str,
        target_id: str,
        edge_type: EdgeType,
        weight: float = 1.0,
        metadata: dict[str, Any] | None = None,
    ) -> GraphEdge:
        """Add a directed edge. Both nodes must already exist."""
        with self._lock:
            if source_id not in self._nodes:
                raise KeyError(f"Source node {source_id!r} not in graph")
            if target_id not in self._nodes:
                raise KeyError(f"Target node {target_id!r} not in graph")
            edge = GraphEdge(
                source_id=source_id,
                target_id=target_id,
                edge_type=edge_type,
                weight=weight,
                metadata=metadata or {},
            )
            self._edges[edge.edge_id] = edge
            self._out_edges[source_id].add(edge.edge_id)
            self._in_edges[target_id].add(edge.edge_id)
            return edge

    def get_edge(self, edge_id: str) -> GraphEdge | None:
        with self._lock:
            return self._edges.get(edge_id)

    def remove_edge(self, edge_id: str) -> None:
        with self._lock:
            self._remove_edge_unsafe(edge_id)

    def _remove_edge_unsafe(self, edge_id: str) -> None:
        """Remove an edge — must be called under self._lock."""
        edge = self._edges.pop(edge_id, None)
        if edge:
            self._out_edges.get(edge.source_id, set()).discard(edge_id)
            self._in_edges.get(edge.target_id, set()).discard(edge_id)

    # ------------------------------------------------------------------ #
    #  Traversal                                                            #
    # ------------------------------------------------------------------ #

    def neighbors(
        self,
        node_id: str,
        edge_type: EdgeType | None = None,
        direction: str = "out",   # "out" | "in" | "both"
    ) -> list[GraphNode]:
        """
        Return all neighbors reachable from node_id.

        direction="out"  → nodes that node_id points TO
        direction="in"   → nodes that point TO node_id
        direction="both" → union of both
        """
        with self._lock:
            edge_ids: set[str] = set()
            if direction in ("out", "both"):
                edge_ids |= self._out_edges.get(node_id, set())
            if direction in ("in", "both"):
                edge_ids |= self._in_edges.get(node_id, set())

            result = []
            for eid in edge_ids:
                edge = self._edges[eid]
                if edge_type and edge.edge_type != edge_type:
                    continue
                neighbor_id = (
                    edge.target_id if edge.source_id == node_id else edge.source_id
                )
                node = self._nodes.get(neighbor_id)
                if node:
                    result.append(node)
            return result

    def edges_between(
        self, source_id: str, target_id: str
    ) -> list[GraphEdge]:
        """Return all edges from source to target."""
        with self._lock:
            result = []
            for eid in self._out_edges.get(source_id, set()):
                e = self._edges[eid]
                if e.target_id == target_id:
                    result.append(e)
            return result

    def bfs(
        self,
        start_id: str,
        direction: str = "out",
        edge_type: EdgeType | None = None,
        max_depth: int = 10,
    ) -> Iterator[tuple[int, GraphNode]]:
        """
        Breadth-first traversal starting from start_id.
        Yields (depth, node) pairs.
        """
        visited = {start_id}
        queue: list[tuple[int, str]] = [(0, start_id)]
        while queue:
            depth, current_id = queue.pop(0)
            node = self._nodes.get(current_id)
            if node:
                yield depth, node
            if depth >= max_depth:
                continue
            for neighbor in self.neighbors(current_id, edge_type=edge_type, direction=direction):
                if neighbor.node_id not in visited:
                    visited.add(neighbor.node_id)
                    queue.append((depth + 1, neighbor.node_id))

    def subgraph(
        self,
        node_ids: list[str],
        include_connecting_edges: bool = True,
    ) -> "ResearchGraph":
        """Extract a subgraph containing only the given node IDs."""
        sub = ResearchGraph()
        id_set = set(node_ids)
        with self._lock:
            for nid in node_ids:
                node = self._nodes.get(nid)
                if node:
                    sub.add_node(
                        node.node_type, node.label,
                        content=node.content, metadata=node.metadata,
                        score=node.score, node_id=node.node_id,
                    )
            if include_connecting_edges:
                for edge in self._edges.values():
                    if edge.source_id in id_set and edge.target_id in id_set:
                        sub.add_edge(
                            edge.source_id, edge.target_id,
                            edge.edge_type, edge.weight, edge.metadata,
                        )
        return sub

    def citation_chain(self, node_id: str, max_depth: int = 4) -> list[GraphNode]:
        """Return all papers reachable via CITES edges (bibliography traversal)."""
        return [
            n for _, n in self.bfs(node_id, edge_type=EdgeType.CITES, max_depth=max_depth)
            if n.node_type == NodeType.PAPER
        ]

    def evidence_for(self, hypothesis_id: str) -> list[GraphNode]:
        """Return all findings/experiments that support a hypothesis."""
        return self.neighbors(hypothesis_id, edge_type=EdgeType.SUPPORTS, direction="in")

    def evidence_against(self, hypothesis_id: str) -> list[GraphNode]:
        """Return all findings that contradict a hypothesis."""
        return self.neighbors(hypothesis_id, edge_type=EdgeType.CONTRADICTS, direction="in")

    # ------------------------------------------------------------------ #
    #  Stats                                                                #
    # ------------------------------------------------------------------ #

    @property
    def node_count(self) -> int:
        return len(self._nodes)

    @property
    def edge_count(self) -> int:
        return len(self._edges)

    def stats(self) -> dict[str, Any]:
        with self._lock:
            type_counts: dict[str, int] = {}
            edge_counts: dict[str, int] = {}
            for n in self._nodes.values():
                type_counts[n.node_type.value] = type_counts.get(n.node_type.value, 0) + 1
            for e in self._edges.values():
                edge_counts[e.edge_type.value] = edge_counts.get(e.edge_type.value, 0) + 1
            return {
                "total_nodes": len(self._nodes),
                "total_edges": len(self._edges),
                "nodes_by_type": type_counts,
                "edges_by_type": edge_counts,
            }

    # ------------------------------------------------------------------ #
    #  Rendering                                                            #
    # ------------------------------------------------------------------ #

    def render_ascii(
        self,
        root_id: str | None = None,
        max_depth: int = 4,
    ) -> str:
        """
        Render a simple ASCII representation of the graph (or a subtree).
        If root_id is None, renders the full graph grouped by node type.
        """
        if root_id:
            lines = [f"Graph from root [{root_id}]:"]
            for depth, node in self.bfs(root_id, max_depth=max_depth):
                indent = "  " * depth
                score = f" score={node.score:.3f}" if node.score is not None else ""
                lines.append(
                    f"{indent}[{node.node_type.value}] {node.label[:60]}{score}"
                )
            return "\n".join(lines)

        # Full graph grouped by type
        lines = [f"ResearchGraph ({self.node_count} nodes, {self.edge_count} edges):"]
        with self._lock:
            by_type: dict[str, list[GraphNode]] = {}
            for node in self._nodes.values():
                by_type.setdefault(node.node_type.value, []).append(node)
            for ntype, nodes in sorted(by_type.items()):
                lines.append(f"\n  [{ntype.upper()}] ({len(nodes)})")
                for node in nodes[:10]:
                    score = f" ({node.score:.3f})" if node.score is not None else ""
                    n_out = len(self._out_edges.get(node.node_id, set()))
                    n_in  = len(self._in_edges.get(node.node_id, set()))
                    lines.append(
                        f"    ├── {node.node_id} {node.label[:50]}{score} "
                        f"[out={n_out} in={n_in}]"
                    )
                if len(nodes) > 10:
                    lines.append(f"    └── ... and {len(nodes)-10} more")
        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    #  Serialisation                                                        #
    # ------------------------------------------------------------------ #

    def to_dict(self) -> dict[str, Any]:
        with self._lock:
            return {
                "nodes": {nid: asdict(n) for nid, n in self._nodes.items()},
                "edges": {eid: asdict(e) for eid, e in self._edges.items()},
            }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ResearchGraph":
        g = cls()
        for nid, nd in data.get("nodes", {}).items():
            nd["node_type"] = NodeType(nd["node_type"])
            node = GraphNode(**nd)
            g._nodes[nid] = node
            g._out_edges.setdefault(nid, set())
            g._in_edges.setdefault(nid, set())
        for eid, ed in data.get("edges", {}).items():
            ed["edge_type"] = EdgeType(ed["edge_type"])
            edge = GraphEdge(**ed)
            g._edges[eid] = edge
            g._out_edges.setdefault(edge.source_id, set()).add(eid)
            g._in_edges.setdefault(edge.target_id, set()).add(eid)
        return g

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.to_dict(), indent=2))
        tmp.replace(path)

    @classmethod
    def load(cls, path: Path) -> "ResearchGraph":
        if not path.exists():
            return cls()
        return cls.from_dict(json.loads(path.read_text()))

    def __repr__(self) -> str:
        return f"ResearchGraph(nodes={self.node_count}, edges={self.edge_count})"
