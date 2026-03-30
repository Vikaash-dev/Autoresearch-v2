"""
Autoresearch v2 — Self-Evolving Multi-Agent Research Framework

Architecture inspired by:
  - Bilevel Autoresearch (2026): multi-batch persistent experience, bilevel optimisation
  - The AI Scientist-v2 (Sakana AI, 2026): agentic tree search, full research lifecycle
  - Centaur (2026): hybrid LLM + classical HPO (CMA-ES)
  - AIRS-Bench (2026): frontier research-agent evaluation
  - AutoResearcher (2025): knowledge-grounded multi-agent generation
  - DGM-Hyperagents (Meta, 2026): self-referential meta-improvement
"""

from autoresearch.orchestrator import ResearchOrchestrator
from autoresearch.config import AutoresearchConfig

__all__ = ["ResearchOrchestrator", "AutoresearchConfig"]
__version__ = "2.0.0"
