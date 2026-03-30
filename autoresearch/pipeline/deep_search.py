"""
Deep Search Pipeline — Python port of dzhng/deep-research + GPT-Researcher planner.

Core algorithm (from dzhng/deep-research):
  deepSearch(query, breadth, depth):
    queries = generate_serp_queries(query, learnings_so_far, n=breadth)
    for each query (concurrent, up to ConcurrencyLimit):
        results = search(query)
        learnings, follow_ups = process_results(results)
        if depth > 1:
            next_query = build_next_query(follow_ups, prior_goals)
            deepSearch(next_query, breadth//2, depth-1)   ← recurse
    return all_learnings, all_urls

Key insight (Karpathy): fixed budget forces fair comparison across different strategies.
Key insight (dzhng): breadth halves at every depth level — prevents exponential blowup.
Key insight (GPT-Researcher): generate clarifying questions BEFORE the search tree.
Key insight (Tongyi): IterResearch — each depth level builds on ALL prior learnings.
"""

from __future__ import annotations

import concurrent.futures
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class SearchQuery:
    query: str
    research_goal: str
    depth: int = 0


@dataclass
class SearchResult:
    url: str
    title: str
    content: str
    score: float = 1.0  # relevance score (Tavily / Firecrawl style)


@dataclass
class ProcessedResult:
    learnings: List[str] = field(default_factory=list)
    follow_up_questions: List[str] = field(default_factory=list)
    urls: List[str] = field(default_factory=list)


@dataclass
class DeepSearchProgress:
    current_depth: int = 0
    total_depth: int = 0
    current_breadth: int = 0
    total_breadth: int = 0
    current_query: str = ""
    total_queries: int = 0
    completed_queries: int = 0
    all_learnings: List[str] = field(default_factory=list)
    all_urls: List[str] = field(default_factory=list)


@dataclass
class DeepSearchOutput:
    learnings: List[str]
    urls: List[str]
    total_queries_run: int
    elapsed_seconds: float
    depth_reached: int


# ---------------------------------------------------------------------------
# Provider interfaces (pluggable — swap real APIs in production)
# ---------------------------------------------------------------------------

class SearchProvider:
    """Abstract web/paper search provider.  Override search() for real APIs."""

    def search(self, query: str, limit: int = 5) -> List[SearchResult]:  # noqa: ARG002
        """Return up to `limit` results for `query`.  Mock by default."""
        words = re.findall(r"\w+", query.lower())
        return [
            SearchResult(
                url=f"https://mock.example.com/{i}/{words[0] if words else 'result'}",
                title=f"Result {i}: {query[:40]}",
                content=(
                    f"This document discusses {query}. "
                    f"Key finding {i}: the relationship between {' and '.join(words[:3])} "
                    f"suggests a promising research direction. "
                    f"Methodology: systematic review of {20 + i * 5} prior studies. "
                    f"Conclusion: further investigation is warranted."
                ),
                score=round(1.0 - i * 0.1, 2),
            )
            for i in range(1, limit + 1)
        ]


class LLMProvider:
    """Abstract LLM provider.  Override generate() for real models."""

    def generate(self, prompt: str, system: str = "") -> str:  # noqa: ARG002
        """Return a text response.  Mock by default."""
        words = re.findall(r"\w+", prompt.lower())
        unique = list(dict.fromkeys(words))[:8]
        return " ".join(unique)


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

class DeepSearchEngine:
    """
    Recursive deep search engine.

    Mirrors dzhng/deep-research's deepResearch() function but in Python,
    extended with:
    - GPT-Researcher style clarifying questions before the search tree
    - Tongyi IterResearch: every depth level receives ALL prior learnings
    - KEA-style: multiple queries are run concurrently and cross-validated
    - Budget tracking (Karpathy: fixed budget per run)
    """

    CONCURRENCY_LIMIT = 3  # raise for paid API tiers

    def __init__(
        self,
        search_provider: Optional[SearchProvider] = None,
        llm_provider: Optional[LLMProvider] = None,
        max_budget_seconds: float = 300.0,  # Karpathy: fixed 5-min budget
    ) -> None:
        self.search = search_provider or SearchProvider()
        self.llm = llm_provider or LLMProvider()
        self.max_budget_seconds = max_budget_seconds
        self._start_time: float = 0.0
        self._queries_run: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def clarify(self, query: str, n: int = 3) -> List[str]:
        """
        GPT-Researcher step 0: generate clarifying follow-up questions
        before starting the search tree.
        """
        prompt = (
            f"Given this research query: '{query}'\n"
            f"Generate {n} clarifying questions that would help focus the research.\n"
            f"Return one question per line."
        )
        raw = self.llm.generate(prompt, system="You are an expert research planner.")
        lines = [l.strip() for l in raw.split("\n") if l.strip()]
        # Fallback mock questions if LLM returns garbage
        if len(lines) < n:
            lines = [
                f"What is the most important aspect of {query}?",
                f"What existing approaches address {query}?",
                f"What metrics best evaluate progress on {query}?",
            ][:n]
        return lines[:n]

    def run(
        self,
        query: str,
        breadth: int = 4,
        depth: int = 2,
        prior_learnings: Optional[List[str]] = None,
        on_progress: Optional[Callable[[DeepSearchProgress], None]] = None,
    ) -> DeepSearchOutput:
        """
        Entry point.  Runs the recursive search tree with a fixed time budget.
        """
        self._start_time = time.time()
        self._queries_run = 0
        learnings: List[str] = list(prior_learnings or [])
        urls: List[str] = []

        progress = DeepSearchProgress(
            total_depth=depth,
            total_breadth=breadth,
        )

        self._recurse(
            query=query,
            breadth=breadth,
            depth=depth,
            learnings=learnings,
            urls=urls,
            progress=progress,
            on_progress=on_progress,
        )

        elapsed = time.time() - self._start_time
        return DeepSearchOutput(
            learnings=list(dict.fromkeys(learnings)),   # dedup, preserve order
            urls=list(dict.fromkeys(urls)),
            total_queries_run=self._queries_run,
            elapsed_seconds=elapsed,
            depth_reached=depth - progress.current_depth,
        )

    # ------------------------------------------------------------------
    # Internal recursion
    # ------------------------------------------------------------------

    def _recurse(
        self,
        query: str,
        breadth: int,
        depth: int,
        learnings: List[str],
        urls: List[str],
        progress: DeepSearchProgress,
        on_progress: Optional[Callable],
    ) -> None:
        if depth <= 0:
            return
        if self._over_budget():
            logger.info("Budget exhausted — stopping search early.")
            return

        progress.current_depth = depth
        progress.current_breadth = breadth

        # Generate SERP queries for this level
        serp_queries = self._generate_serp_queries(query, learnings, n=breadth)
        progress.total_queries += len(serp_queries)

        if on_progress:
            on_progress(progress)

        # Run queries concurrently (dzhng: pLimit style)
        limit = min(self.CONCURRENCY_LIMIT, len(serp_queries))
        with concurrent.futures.ThreadPoolExecutor(max_workers=limit) as pool:
            futures = {
                pool.submit(self._process_query, sq, learnings): sq
                for sq in serp_queries
            }
            for future in concurrent.futures.as_completed(futures):
                sq = futures[future]
                if self._over_budget():
                    break
                try:
                    processed = future.result()
                    learnings.extend(processed.learnings)
                    urls.extend(processed.urls)
                    progress.completed_queries += 1
                    progress.all_learnings = list(learnings)
                    progress.all_urls = list(urls)

                    if on_progress:
                        progress.current_query = sq.query
                        on_progress(progress)

                    # Recurse: breadth halves, depth decrements (dzhng pattern)
                    if depth > 1 and processed.follow_up_questions:
                        next_query = self._build_next_query(
                            prior_goal=sq.research_goal,
                            follow_ups=processed.follow_up_questions,
                            learnings=learnings,
                        )
                        self._recurse(
                            query=next_query,
                            breadth=max(1, breadth // 2),
                            depth=depth - 1,
                            learnings=learnings,
                            urls=urls,
                            progress=progress,
                            on_progress=on_progress,
                        )
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Query '%s' failed: %s", sq.query, exc)

    def _over_budget(self) -> bool:
        return (time.time() - self._start_time) >= self.max_budget_seconds

    # ------------------------------------------------------------------
    # Helper methods (LLM-backed, mock in base implementation)
    # ------------------------------------------------------------------

    def _generate_serp_queries(
        self, query: str, learnings: List[str], n: int
    ) -> List[SearchQuery]:
        """Generate n diverse SERP queries for this research step."""
        prompt = (
            f"Research goal: {query}\n"
            f"Prior learnings: {'; '.join(learnings[-5:])}\n"  # last 5 learnings
            f"Generate {n} specific search queries. One per line."
        )
        raw = self.llm.generate(prompt)
        lines = [l.strip() for l in raw.split("\n") if l.strip()]

        # Build SearchQuery objects; fall back to topic variants if LLM thin
        queries: List[SearchQuery] = []
        base_words = re.findall(r"\w+", query)[:4]
        fallbacks = [
            f"{query} overview",
            f"{query} state of the art",
            f"{query} methods comparison",
            f"{query} limitations challenges",
        ]
        for i in range(n):
            text = lines[i] if i < len(lines) else fallbacks[i % len(fallbacks)]
            queries.append(SearchQuery(
                query=text,
                research_goal=f"Understand {text} in context of {query}",
                depth=i,
            ))
        return queries

    def _process_query(
        self, sq: SearchQuery, prior_learnings: List[str]
    ) -> ProcessedResult:
        """Search → extract learnings + follow-up questions."""
        self._queries_run += 1
        results = self.search.search(sq.query, limit=5)

        if not results:
            return ProcessedResult()

        # Build context from search results
        context = "\n\n".join(
            f"[{r.title}]({r.url})\n{r.content[:500]}"
            for r in results
        )
        prompt = (
            f"Query: {sq.query}\n"
            f"Prior learnings: {'; '.join(prior_learnings[-3:])}\n"
            f"Search results:\n{context}\n\n"
            f"Extract 3 key learnings and 2 follow-up questions. "
            f"Format: LEARNING: ... on its own line, FOLLOWUP: ... on its own line."
        )
        raw = self.llm.generate(prompt)

        learnings: List[str] = []
        follow_ups: List[str] = []

        for line in raw.split("\n"):
            line = line.strip()
            if line.startswith("LEARNING:"):
                learnings.append(line[len("LEARNING:"):].strip())
            elif line.startswith("FOLLOWUP:"):
                follow_ups.append(line[len("FOLLOWUP:"):].strip())

        # Fallback: extract sentences from content as learnings
        if not learnings:
            for r in results[:3]:
                sentences = re.split(r"[.!?]", r.content)
                for s in sentences[:2]:
                    if len(s.strip()) > 30:
                        learnings.append(s.strip())

        if not follow_ups:
            words = re.findall(r"\w+", sq.query)[:3]
            follow_ups = [f"What are the limitations of {' '.join(words)}?"]

        return ProcessedResult(
            learnings=learnings[:3],
            follow_up_questions=follow_ups[:2],
            urls=[r.url for r in results],
        )

    @staticmethod
    def _build_next_query(
        prior_goal: str, follow_ups: List[str], learnings: List[str]
    ) -> str:
        """Build the next level's query from prior goal + follow-up questions."""
        follow_up_str = " | ".join(follow_ups[:2])
        recent = "; ".join(learnings[-2:]) if learnings else ""
        return (
            f"Prior goal: {prior_goal[:100]}. "
            f"Follow-up: {follow_up_str}. "
            f"Context: {recent[:200]}"
        )
