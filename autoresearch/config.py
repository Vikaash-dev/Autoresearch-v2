"""
Central configuration for Autoresearch v2 — SERA Architecture.

Provides typed, validated, serialisable config for every layer of the
Self-Evolving Research Architecture:
  - LLM backend (OpenAI / Anthropic / local)
  - Persistent memory (SQLite / JSON / in-memory)
  - Agentic tree-search (AI Scientist-v2 style)
  - Bilevel + CMA-ES hybrid HPO (Centaur style)
  - AIRS-Bench evaluation metrics
  - Per-agent self-evolution hyperparameters
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class LLMConfig:
    """Configuration for the language model backend."""

    provider: str = "openai"
    model: str = "gpt-4o"
    api_key: Optional[str] = field(default_factory=lambda: os.getenv("OPENAI_API_KEY"))
    max_tokens: int = 4096
    temperature: float = 0.7
    timeout: int = 120

    # Fallback provider (used by hybrid router when cloud cost is high)
    fallback_provider: Optional[str] = None
    fallback_model: Optional[str] = None


@dataclass
class MemoryConfig:
    """Configuration for persistent memory / experience store."""

    backend: str = "sqlite"          # "sqlite" | "json" | "in_memory"
    db_path: str = "autoresearch_memory.db"
    max_experiences: int = 10_000
    recycling_threshold: float = 0.3  # Bilevel: recycle failed runs below this score


@dataclass
class SearchConfig:
    """Agentic tree-search settings (AI Scientist-v2 style)."""

    max_depth: int = 5
    branching_factor: int = 3
    exploration_weight: float = 1.4   # UCB exploration constant
    max_nodes: int = 50
    pruning_threshold: float = 0.2    # prune branches below this value score


@dataclass
class OptimizationConfig:
    """Bilevel / Centaur hybrid-HPO settings."""

    # Bilevel
    outer_iterations: int = 10
    inner_iterations: int = 5
    batch_size: int = 8

    # Centaur CMA-ES
    cmaes_sigma0: float = 0.5
    cmaes_popsize: int = 10
    cmaes_max_iter: int = 100
    use_hybrid_hpo: bool = True        # combine LLM + CMA-ES


@dataclass
class EvaluationConfig:
    """AIRS-Bench style evaluation settings."""

    metrics: List[str] = field(
        default_factory=lambda: [
            "novelty",
            "correctness",
            "reproducibility",
            "impact_score",
        ]
    )
    human_sota_baseline: float = 1.0   # normalised SOTA ceiling
    min_acceptable_score: float = 0.6


@dataclass
class AgentConfig:
    """Per-agent tunables."""

    max_self_evolution_rounds: int = 5
    evolution_lr: float = 0.1          # parameter update rate during self-evolution
    meta_improvement_interval: int = 3  # how often DGM-style meta-agent runs (rounds)
    enable_knowledge_grounding: bool = True
    enable_novelty_check: bool = True


@dataclass
class AutoresearchConfig:
    """Top-level configuration object."""

    llm: LLMConfig = field(default_factory=LLMConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)

    # General
    task: str = ""
    output_dir: str = "output"
    verbose: bool = False
    seed: int = 42

    extra: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self) -> None:
        """Raise ValueError for any obviously invalid setting."""
        if not (0.0 <= self.llm.temperature <= 2.0):
            raise ValueError(f"llm.temperature must be in [0, 2], got {self.llm.temperature}")
        if self.llm.max_tokens < 1:
            raise ValueError("llm.max_tokens must be >= 1")
        if not (0.0 <= self.memory.recycling_threshold <= 1.0):
            raise ValueError("memory.recycling_threshold must be in [0, 1]")
        if self.search.branching_factor < 1:
            raise ValueError("search.branching_factor must be >= 1")
        if self.search.max_depth < 1:
            raise ValueError("search.max_depth must be >= 1")
        if not (0.0 <= self.agent.evolution_lr <= 1.0):
            raise ValueError("agent.evolution_lr must be in [0, 1]")
        if self.optimization.batch_size < 1:
            raise ValueError("optimization.batch_size must be >= 1")

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "AutoresearchConfig":
        cfg = cls()
        for key, value in d.items():
            if hasattr(cfg, key) and isinstance(value, dict):
                sub = getattr(cfg, key)
                for k, v in value.items():
                    if hasattr(sub, k):
                        setattr(sub, k, v)
            elif hasattr(cfg, key):
                setattr(cfg, key, value)
        return cfg

    @classmethod
    def from_json(cls, json_str: str) -> "AutoresearchConfig":
        return cls.from_dict(json.loads(json_str))

    @classmethod
    def from_file(cls, path: str) -> "AutoresearchConfig":
        """Load config from a JSON file."""
        text = Path(path).read_text(encoding="utf-8")
        return cls.from_json(text)
