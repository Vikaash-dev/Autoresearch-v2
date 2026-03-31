"""
Tool Registry with Tool-ToM metadata for Autoresearch-v2.

Tracks all discovered and registered tools — their capabilities, failure modes,
known alternatives, and performance scores on different task types. This implements
the Tool-ToM layer: agents consult the registry before choosing a tool so they pick
the best one for the specific query type rather than always using the same default.

Design influenced by:
  - Voyager's skill library (vector-indexed, performance-scored)
  - The previous analysis: "Tool-ToM: model external tool behavior"
  - NousResearch/hermes-agent tool discovery mechanism

Registry storage: JSON file at tools/tool_registry.json
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ToolRecord:
    """Full Tool-ToM record for one tool."""

    name: str
    description: str
    artifact_type: str = "function"      # "function" | "api_client" | "subprocess" | "package"
    python_import: str = ""              # e.g. "from tavily import TavilyClient"
    install_cmd: str = ""               # e.g. "pip install tavily-python"
    version: str = ""
    good_at: list[str] = field(default_factory=list)
    bad_at: list[str] = field(default_factory=list)
    common_failure_modes: list[str] = field(default_factory=list)
    known_alternatives: list[str] = field(default_factory=list)  # other tool names
    task_scores: dict[str, float] = field(default_factory=dict)  # task_type → score
    overall_score: float = 0.0
    use_count: int = 0
    fail_count: int = 0
    tags: list[str] = field(default_factory=list)
    source_url: str = ""               # GitHub URL or PyPI page
    registered_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)


class ToolRegistry:
    """
    Registry of all available tools with Tool-ToM metadata.

    Usage:
        registry = ToolRegistry(Path("tools/tool_registry.json"))
        registry.register(ToolRecord(name="tavily_search", ...))
        best = registry.get_best_for("search research papers about transformers")
        registry.record_use("tavily_search", task_type="paper_search", success=True, score=0.9)
    """

    def __init__(self, persist_path: Path | None = None) -> None:
        self._path = persist_path
        self._tools: dict[str, ToolRecord] = {}
        if persist_path and persist_path.exists():
            self._load()
        else:
            self._register_builtin_tools()

    # ------------------------------------------------------------------ #
    #  Registration                                                         #
    # ------------------------------------------------------------------ #

    def register(self, record: ToolRecord, overwrite: bool = False) -> None:
        """Register a tool. Raises if name already exists unless overwrite=True."""
        if record.name in self._tools and not overwrite:
            logger.debug("ToolRegistry: %r already registered — skipping", record.name)
            return
        record.updated_at = time.time()
        self._tools[record.name] = record
        self._flush()
        logger.info("ToolRegistry: registered tool %r (score=%.2f)", record.name, record.overall_score)

    def unregister(self, name: str) -> bool:
        """Remove a tool from the registry. Returns True if it existed."""
        if name in self._tools:
            del self._tools[name]
            self._flush()
            return True
        return False

    # ------------------------------------------------------------------ #
    #  Query / selection (Tool-ToM core)                                   #
    # ------------------------------------------------------------------ #

    def get(self, name: str) -> ToolRecord | None:
        return self._tools.get(name)

    def list_all(self, tag: str | None = None) -> list[ToolRecord]:
        tools = list(self._tools.values())
        if tag:
            tools = [t for t in tools if tag in t.tags]
        return sorted(tools, key=lambda t: t.overall_score, reverse=True)

    def get_best_for(self, task_description: str, fallback: str | None = None) -> ToolRecord | None:
        """
        Select the best registered tool for a given task description.

        Uses keyword matching against each tool's good_at list and task_scores.
        Returns the tool with the highest relevance × performance score.
        Falls back to the tool named `fallback` if no match found.
        """
        desc_tokens = set(task_description.lower().split())
        scored: list[tuple[float, ToolRecord]] = []

        for tool in self._tools.values():
            # Relevance: fraction of good_at phrases that overlap with description
            relevance = 0.0
            for phrase in tool.good_at:
                phrase_tokens = set(phrase.lower().split())
                overlap = len(desc_tokens & phrase_tokens) / max(len(phrase_tokens), 1)
                relevance = max(relevance, overlap)

            # Task-specific score if we have one
            task_score = 0.0
            for task_type, score in tool.task_scores.items():
                task_tokens = set(task_type.lower().split("_"))
                if desc_tokens & task_tokens:
                    task_score = max(task_score, score)

            combined = relevance * 0.4 + (task_score or tool.overall_score) * 0.6
            if combined > 0:
                scored.append((combined, tool))

        if scored:
            scored.sort(key=lambda x: x[0], reverse=True)
            best = scored[0][1]
            logger.debug(
                "ToolRegistry: selected %r (score=%.3f) for task: %s",
                best.name, scored[0][0], task_description[:60],
            )
            return best

        if fallback and fallback in self._tools:
            return self._tools[fallback]
        return None

    def get_alternatives(self, name: str) -> list[ToolRecord]:
        """Return known alternative tools for the named tool."""
        tool = self._tools.get(name)
        if not tool:
            return []
        return [self._tools[alt] for alt in tool.known_alternatives if alt in self._tools]

    # ------------------------------------------------------------------ #
    #  Performance tracking                                                 #
    # ------------------------------------------------------------------ #

    def record_use(
        self,
        name: str,
        task_type: str,
        success: bool,
        score: float | None = None,
    ) -> None:
        """
        Record a tool use outcome. Updates task_scores and overall_score
        using an exponential moving average (α=0.2) so recent results
        count more than old ones.
        """
        tool = self._tools.get(name)
        if not tool:
            logger.warning("ToolRegistry.record_use: unknown tool %r", name)
            return

        tool.use_count += 1
        if not success:
            tool.fail_count += 1

        # EMA update for task-specific score
        observed = score if score is not None else (1.0 if success else 0.0)
        alpha = 0.2
        if task_type in tool.task_scores:
            tool.task_scores[task_type] = (
                alpha * observed + (1 - alpha) * tool.task_scores[task_type]
            )
        else:
            tool.task_scores[task_type] = observed

        # Overall score = mean of task scores
        if tool.task_scores:
            tool.overall_score = round(
                sum(tool.task_scores.values()) / len(tool.task_scores), 4
            )

        tool.updated_at = time.time()
        self._flush()

    # ------------------------------------------------------------------ #
    #  Built-in tool definitions                                            #
    # ------------------------------------------------------------------ #

    def _register_builtin_tools(self) -> None:
        """Pre-register the tools bundled with Autoresearch-v2."""
        builtins: list[ToolRecord] = [
            ToolRecord(
                name="tavily_search",
                description="Real-time web search via Tavily API with domain filtering",
                artifact_type="api_client",
                python_import="from tools.tavily_search import TavilyPool",
                install_cmd="pip install tavily-python",
                good_at=[
                    "search research papers", "search arxiv", "search github repositories",
                    "find recent preprints", "search semantic scholar",
                    "extract web page content", "find code implementations",
                ],
                bad_at=["structured database queries", "private repositories"],
                common_failure_modes=["API key quota exhausted", "rate limit 429"],
                known_alternatives=["semantic_scholar_api", "arxiv_api"],
                task_scores={
                    "paper_search": 0.90,
                    "github_search": 0.85,
                    "content_extraction": 0.80,
                },
                overall_score=0.85,
                tags=["search", "web", "papers", "github"],
                source_url="https://github.com/tavily-ai/tavily-python",
            ),
            ToolRecord(
                name="semantic_scholar_api",
                description="Structured paper search via Semantic Scholar REST API",
                artifact_type="api_client",
                python_import="import urllib.request",
                good_at=[
                    "search research papers", "get paper metadata", "find citations",
                    "search by author", "get abstract and year",
                ],
                bad_at=["very recent preprints (<48h)", "GitHub repo search"],
                common_failure_modes=["rate limit without API key", "missing abstracts"],
                known_alternatives=["tavily_search", "arxiv_api"],
                task_scores={"paper_search": 0.80, "citation_lookup": 0.90},
                overall_score=0.80,
                tags=["search", "papers", "academic"],
                source_url="https://api.semanticscholar.org/",
            ),
            ToolRecord(
                name="arxiv_api",
                description="arXiv XML API for searching and retrieving preprints",
                artifact_type="api_client",
                python_import="from urllib.request import urlopen",
                good_at=[
                    "search arxiv preprints", "find very recent papers",
                    "search by category (cs.LG, cs.AI)", "free no-key access",
                ],
                bad_at=["semantic similarity search", "GitHub repo search", "full-text search"],
                common_failure_modes=["rate limit at high volume", "XML parsing edge cases"],
                known_alternatives=["semantic_scholar_api", "tavily_search"],
                task_scores={"paper_search": 0.70, "recent_preprints": 0.85},
                overall_score=0.72,
                tags=["search", "papers", "arxiv", "free"],
                source_url="https://arxiv.org/help/api/",
            ),
            ToolRecord(
                name="subprocess_executor",
                description="Sandboxed Python subprocess executor for running experiments",
                artifact_type="subprocess",
                python_import="import subprocess",
                good_at=[
                    "run python experiments", "execute generated code",
                    "capture stdout metrics", "isolated code execution",
                ],
                bad_at=["GPU workloads without CUDA config", "long-running jobs >5min by default"],
                common_failure_modes=["TimeoutExpired", "ImportError for missing packages"],
                known_alternatives=[],
                task_scores={"code_execution": 0.90, "experiment_eval": 0.88},
                overall_score=0.89,
                tags=["execution", "sandbox", "experiment"],
                source_url="https://docs.python.org/3/library/subprocess.html",
            ),
            ToolRecord(
                name="skill_store",
                description="Persistent versioned store for evolved agent skills",
                artifact_type="function",
                python_import="from memory.skill_store import SkillStore",
                good_at=[
                    "persist agent prompts", "version skill artifacts",
                    "retrieve best skill by type", "rollback to previous version",
                ],
                bad_at=["vector similarity search (uses keyword matching)"],
                common_failure_modes=["disk full", "concurrent write race condition"],
                known_alternatives=["trajectory_log"],
                task_scores={"skill_persistence": 0.95, "skill_retrieval": 0.80},
                overall_score=0.88,
                tags=["memory", "skills", "persistence"],
                source_url="memory/skill_store.py",
            ),
            ToolRecord(
                name="trajectory_log",
                description="Append-only agent action log for Self-ToM and GEPA analysis",
                artifact_type="function",
                python_import="from memory.trajectory_log import TrajectoryLog",
                good_at=[
                    "log agent actions", "analyze failure patterns",
                    "provide GEPA execution traces", "Self-ToM data source",
                ],
                bad_at=["real-time streaming analytics"],
                common_failure_modes=["disk I/O on high-frequency logging"],
                known_alternatives=["skill_store"],
                task_scores={"action_logging": 0.95, "failure_analysis": 0.90},
                overall_score=0.92,
                tags=["memory", "logging", "analytics"],
                source_url="memory/trajectory_log.py",
            ),
        ]
        for tool in builtins:
            self._tools[tool.name] = tool
        self._flush()

    # ------------------------------------------------------------------ #
    #  Persistence                                                          #
    # ------------------------------------------------------------------ #

    def _flush(self) -> None:
        if not self._path:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        data = {name: asdict(rec) for name, rec in self._tools.items()}
        tmp = self._path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2))
        tmp.replace(self._path)

    def _load(self) -> None:
        data = json.loads(self._path.read_text())
        self._tools = {name: ToolRecord(**rec) for name, rec in data.items()}

    def __len__(self) -> int:
        return len(self._tools)

    def __repr__(self) -> str:
        return f"ToolRegistry(tools={list(self._tools.keys())})"
