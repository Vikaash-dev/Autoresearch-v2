"""
Research Agent — paper discovery, retrieval and structured analysis.

Inspired by:
  - AutoResearcher (2025): knowledge-grounded literature retrieval
  - AI Scientist-v2 (2026): prior-work integration before hypothesis generation
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

from autoresearch.agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)


class Paper:
    """Lightweight representation of a research paper."""

    def __init__(
        self,
        title: str,
        authors: List[str],
        year: int,
        abstract: str,
        url: str = "",
        relevance_score: float = 0.0,
        key_findings: Optional[List[str]] = None,
        methodology: str = "",
        limitations: Optional[List[str]] = None,
    ) -> None:
        self.title = title
        self.authors = authors
        self.year = year
        self.abstract = abstract
        self.url = url
        self.relevance_score = relevance_score
        self.key_findings: List[str] = key_findings or []
        self.methodology = methodology
        self.limitations: List[str] = limitations or []

    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "authors": self.authors,
            "year": self.year,
            "abstract": self.abstract,
            "url": self.url,
            "relevance_score": self.relevance_score,
            "key_findings": self.key_findings,
            "methodology": self.methodology,
            "limitations": self.limitations,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Paper":
        return cls(**{k: v for k, v in d.items() if k in cls.__init__.__code__.co_varnames})


class ResearchAgent(BaseAgent):
    """
    Searches, retrieves and analyses research papers relevant to a given task.

    In production: connect to arXiv / Semantic Scholar / PubMed APIs.
    The mock backend demonstrates the interface without requiring API keys.
    """

    # Curated seed papers reflecting the problem statement's literature
    _SEED_PAPERS: List[Dict[str, Any]] = [
        {
            "title": "Bilevel Autoresearch: Multi-batch Persistent Experience for Hyperparameter Optimisation",
            "authors": ["Anonymous et al."],
            "year": 2026,
            "abstract": (
                "Addresses the single-track iteration limitation of prior autoresearch proposals "
                "by introducing a multi-batch persistent experience model. Casts hyperparameter "
                "search as a bilevel optimisation problem, recycling failed runs as training signal."
            ),
            "key_findings": [
                "Multi-batch parallelism reduces wall-clock time by 3×",
                "Failure recycling improves sample efficiency by 40%",
                "Bilevel formulation converges faster than single-level baselines",
            ],
            "methodology": "Bilevel optimisation + persistent experience replay",
            "limitations": ["Requires warm-start corpus", "Computationally heavy outer loop"],
        },
        {
            "title": "The AI Scientist-v2: Workshop-Level Automated Scientific Discovery via Agentic Tree Search",
            "authors": ["Sakana AI"],
            "year": 2026,
            "abstract": (
                "Removes human-coded templates and introduces a progressive agentic tree-search "
                "for hypothesis space exploration. Produced the first fully AI-generated paper "
                "accepted at a scientific workshop."
            ),
            "key_findings": [
                "Template-free hypothesis generation",
                "Tree search covers 5× more hypothesis space vs. linear approaches",
                "First AI paper accepted at NeurIPS workshop",
            ],
            "methodology": "Progressive agentic tree search + experiment manager agent",
            "limitations": ["High compute per run", "Still requires human review of safety"],
        },
        {
            "title": "Centaur: Can LLMs Beat Classical Hyperparameter Optimisation?",
            "authors": ["Anonymous et al."],
            "year": 2026,
            "abstract": (
                "Hybrid approach sharing CMA-ES internal state (covariance matrix, step-size) "
                "with an LLM, combining domain knowledge with mathematical precision."
            ),
            "key_findings": [
                "Outperforms pure LLM-based HyperAgents on 12/15 benchmarks",
                "CMA-ES covariance sharing reduces LLM sampling variance",
                "Achieves near-optimal solutions 30% faster",
            ],
            "methodology": "CMA-ES + LLM hybrid HPO",
            "limitations": ["CMA-ES overhead in discrete search spaces"],
        },
        {
            "title": "AIRS-Bench: A Suite of Tasks for Frontier AI Research Science Agents",
            "authors": ["Meta AI Research"],
            "year": 2026,
            "abstract": (
                "Establishes a benchmark for Frontier AI Research Science Agents highlighting "
                "that current agents often fail to match human SOTA, providing a roadmap for "
                "next-generation agents."
            ),
            "key_findings": [
                "Best current agent achieves 61% of human SOTA on average",
                "Identified key bottlenecks: reliability and reproducibility",
                "Defines performance ceilings per task category",
            ],
            "methodology": "Benchmark suite with 42 research tasks",
            "limitations": ["Benchmark may not cover all research domains"],
        },
        {
            "title": "AutoResearcher: Knowledge-Grounded Automated Research with Multi-Agent Systems",
            "authors": ["Anonymous et al."],
            "year": 2025,
            "abstract": (
                "Uses a multi-agent system with Knowledge Grounding and Novelty agents to "
                "ensure generated ideas are scientifically sound and original."
            ),
            "key_findings": [
                "Knowledge-grounded ideas are 2× more likely to be novel",
                "Novelty agent reduces duplicate ideas by 78%",
                "Multi-agent coordination improves scientific soundness",
            ],
            "methodology": "Multi-agent: Knowledge Grounding + Novelty + Research agents",
            "limitations": ["Dependent on quality of knowledge base"],
        },
        {
            "title": "DGM-Hyperagents: Self-Referential Meta-Improvement of Research Agents",
            "authors": ["Meta Research"],
            "year": 2026,
            "abstract": (
                "Extends the Darwin Gödel Machine framework, allowing the improvement procedure "
                "itself to evolve. Agents improve their improvement strategies, achieving "
                "meta-level improvements without explicit programming."
            ),
            "key_findings": [
                "Meta-improvement yields 2.3× performance over base HyperAgent",
                "Persistent memory emerges naturally without explicit programming",
                "Performance tracking allows targeted meta-improvement",
            ],
            "methodology": "Darwin Gödel Machine + evolutionary self-improvement",
            "limitations": ["Risk of self-improvement divergence without safety constraints"],
        },
    ]

    def __init__(self, config: Optional[Any] = None, **kwargs: Any) -> None:
        super().__init__(config=config, name="ResearchAgent", **kwargs)
        self._paper_cache: List[Paper] = []

    def _default_parameters(self) -> Dict[str, Any]:
        return {
            "max_papers": 10,
            "min_relevance": 0.3,
            "include_limitations": True,
        }

    def _execute(self, task: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Retrieve and analyse papers relevant to ``task``.

        Returns a dict with ``papers`` (list of Paper dicts) and ``synthesis``.
        """
        max_papers: int = self._parameters.get("max_papers", 10)
        min_relevance: float = self._parameters.get("min_relevance", 0.3)

        papers = self._retrieve(task, max_papers, min_relevance)
        self._paper_cache = papers
        synthesis = self._synthesise(task, papers)

        return {
            "papers": [p.to_dict() for p in papers],
            "synthesis": synthesis,
            "task": task,
        }

    def _retrieve(self, task: str, max_papers: int, min_relevance: float) -> List[Paper]:
        """Score and filter seed papers by relevance to the task."""
        scored: List[Paper] = []
        task_lower = task.lower()
        task_tokens = set(re.split(r"\W+", task_lower))

        for raw in self._SEED_PAPERS:
            paper = Paper(**{k: v for k, v in raw.items() if k in Paper.__init__.__code__.co_varnames})
            relevance = self._compute_relevance(task_tokens, paper)
            paper.relevance_score = relevance
            if relevance >= min_relevance:
                scored.append(paper)

        scored.sort(key=lambda p: p.relevance_score, reverse=True)
        return scored[:max_papers]

    @staticmethod
    def _compute_relevance(task_tokens: set, paper: Paper) -> float:
        """Simple keyword overlap relevance (to be replaced by embedding similarity)."""
        text = " ".join([
            paper.title,
            paper.abstract,
            paper.methodology,
            " ".join(paper.key_findings),
        ]).lower()
        text_tokens = set(re.split(r"\W+", text))
        if not task_tokens or not text_tokens:
            return 0.0
        overlap = len(task_tokens & text_tokens)
        return min(overlap / max(len(task_tokens), 1), 1.0)

    @staticmethod
    def _synthesise(task: str, papers: List[Paper]) -> str:
        if not papers:
            return f"No relevant papers found for task: {task}"
        lines = [f"Literature synthesis for: '{task}'", ""]
        for i, p in enumerate(papers, 1):
            lines.append(f"{i}. [{p.year}] {p.title}")
            lines.append(f"   Methodology: {p.methodology}")
            if p.key_findings:
                lines.append(f"   Key finding: {p.key_findings[0]}")
            lines.append(f"   Relevance: {p.relevance_score:.2f}")
            lines.append("")
        return "\n".join(lines)

    def _score_result(self, result: Any) -> float:
        papers = result.get("papers", [])
        if not papers:
            return 0.0
        avg_relevance = sum(p["relevance_score"] for p in papers) / len(papers)
        return min(avg_relevance * 2, 1.0)

    @property
    def paper_cache(self) -> List[Paper]:
        return list(self._paper_cache)
