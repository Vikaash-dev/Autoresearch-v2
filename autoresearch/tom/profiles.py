"""Reviewer persona profiles for ToM layer."""
from __future__ import annotations
from typing import Dict, List, Optional
from pydantic import BaseModel


class ReviewerPersona(BaseModel):
    name: str
    label: str
    description: str
    priorities: List[str]
    rejection_triggers: List[str]
    acceptance_criteria: List[str]
    bias_profile: Dict[str, float]


_DEFAULT_PERSONAS = {
    "methodological_purist": ReviewerPersona(
        name="methodological_purist",
        label="Methodological Purist",
        description="Demands rigorous statistical methodology, proper controls, and reproducibility.",
        priorities=["statistical_rigor", "reproducibility", "control_groups"],
        rejection_triggers=["no_ablation", "small_sample_size", "missing_baseline"],
        acceptance_criteria=["solid_methodology", "clear_baselines", "reproducible_code"],
        bias_profile={"novelty": 0.3, "rigor": 0.9, "clarity": 0.7},
    ),
    "empirical_skeptic": ReviewerPersona(
        name="empirical_skeptic",
        label="Empirical Skeptic",
        description="Skeptical of theoretical claims without empirical validation.",
        priorities=["empirical_evidence", "real_world_data", "falsifiability"],
        rejection_triggers=["only_synthetic_data", "unfalsifiable_claims", "cherry_picked_results"],
        acceptance_criteria=["diverse_datasets", "negative_results_included", "effect_size_reported"],
        bias_profile={"novelty": 0.5, "rigor": 0.8, "clarity": 0.6},
    ),
    "novelty_maximizer": ReviewerPersona(
        name="novelty_maximizer",
        label="Novelty Maximizer",
        description="Values groundbreaking ideas and novel contributions above all.",
        priorities=["novelty", "impact", "open_problems"],
        rejection_triggers=["incremental_work", "known_results", "limited_scope"],
        acceptance_criteria=["new_research_direction", "high_impact_potential", "open_questions"],
        bias_profile={"novelty": 0.95, "rigor": 0.5, "clarity": 0.6},
    ),
    "benchmark_enforcer": ReviewerPersona(
        name="benchmark_enforcer",
        label="Benchmark Enforcer",
        description="Insists on standard benchmark comparisons and SOTA comparisons.",
        priorities=["benchmark_performance", "fair_comparison", "compute_efficiency"],
        rejection_triggers=["missing_sota_comparison", "non_standard_metrics", "unfair_baselines"],
        acceptance_criteria=["standard_benchmarks", "efficient_compute", "honest_comparison"],
        bias_profile={"novelty": 0.4, "rigor": 0.7, "clarity": 0.8},
    ),
}


class ReviewerPersonaRegistry:
    def __init__(self) -> None:
        self._personas = dict(_DEFAULT_PERSONAS)

    def get(self, name: str) -> Optional[ReviewerPersona]:
        return self._personas.get(name)

    def all(self) -> List[ReviewerPersona]:
        return list(self._personas.values())

    def register(self, persona: ReviewerPersona) -> None:
        self._personas[persona.name] = persona
