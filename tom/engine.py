"""
Theory of Mind module for Autoresearch-v2.

Implements all 7 ToM layers identified in the architecture:
  1. User-ToM      — model user intent, style, domain, trust
  2. Agent-ToM     — inter-agent belief modeling (who knows what)
  3. Self-ToM      — agent monitors its own failure patterns
  4. Reviewer-ToM  — adversarial reviewer simulation (pre-submit)
  5. Task-ToM      — decompose underspecified task descriptions
  6. Tool-ToM      — model tool capabilities and failure modes
  7. Community-ToM — model collective field knowledge / consensus

Data backed by:
  - plastic-labs/honcho  (dialectic user modeling, Pareto frontier of agent memory)
  - NousResearch/hermes-agent (user profiles built from session history)
  - ICLR 2026 insights: ToM as gradient-free inference over mental state space
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ======================================================================= #
#  Shared data structures                                                   #
# ======================================================================= #

@dataclass
class UserModel:
    """Persistent model of who the user is and what they want."""

    user_id: str = "default"
    research_style: str = "exploratory"   # exploratory | rigorous | publishable
    depth_preference: str = "deep"        # quick | medium | deep
    domain_vocabulary: list[str] = field(default_factory=list)
    trust_level: float = 0.5             # 0=low trust (approve everything), 1=full auto
    preferred_output_format: str = "icbinb"  # icbinb | icml | arxiv
    known_dislikes: list[str] = field(default_factory=list)
    session_history_summary: str = ""
    last_updated: float = field(default_factory=time.time)


@dataclass
class ReviewerModel:
    """Model of an adversarial scientific reviewer at a given venue."""

    venue: str = "NeurIPS"
    domain: str = "machine_learning"
    hot_topics: list[str] = field(default_factory=list)
    pet_peeves: list[str] = field(
        default_factory=lambda: [
            "overclaiming", "missing ablations", "weak baselines",
            "no significance tests", "cherry-picked results",
        ]
    )
    required_sections: list[str] = field(
        default_factory=lambda: ["abstract", "introduction", "related_work",
                                  "method", "experiments", "conclusion"]
    )
    likely_rejection_reasons: list[str] = field(default_factory=list)


@dataclass
class CommunityModel:
    """Model of current collective field knowledge."""

    domain: str = ""
    consensus_beliefs: list[str] = field(default_factory=list)
    active_debates: list[str] = field(default_factory=list)
    open_problems: list[str] = field(default_factory=list)
    recent_pivots: list[str] = field(default_factory=list)
    overcrowded_areas: list[str] = field(default_factory=list)
    high_impact_areas: list[str] = field(default_factory=list)
    last_updated: float = field(default_factory=time.time)


@dataclass
class TaskDecomposition:
    """Explicit decomposition of user's task intent."""

    raw_description: str = ""
    stated_goal: str = ""
    implicit_goal: str = ""
    implicit_constraints: list[str] = field(default_factory=list)
    implicit_audience: str = ""
    time_horizon: str = ""           # hours | days | weeks
    success_criteria: list[str] = field(default_factory=list)
    likely_pitfalls: list[str] = field(default_factory=list)


# ======================================================================= #
#  ToM Engine                                                               #
# ======================================================================= #

class TheoryOfMindEngine:
    """
    Central ToM reasoning engine.

    Wraps an LLM call for each ToM layer. Falls back to heuristic defaults
    when no LLM client is configured (useful for testing).
    """

    def __init__(
        self,
        llm_fn: Any | None = None,           # callable(prompt: str) -> str
        persist_path: Path | None = None,
    ) -> None:
        self._llm = llm_fn
        self._persist_path = persist_path
        self._user_model = UserModel()
        self._reviewer_model = ReviewerModel()
        self._community_model = CommunityModel()
        self._self_observations: list[dict[str, Any]] = []

        if persist_path and persist_path.exists():
            self._load()

    # ------------------------------------------------------------------ #
    #  1. User-ToM                                                         #
    # ------------------------------------------------------------------ #

    def update_user_model(self, utterance: str, context: dict[str, Any] | None = None) -> UserModel:
        """
        Update the user model based on a new utterance.
        Inspired by Honcho's dialectic user modeling.
        """
        prompt = (
            "You are modeling the intent and preferences of a researcher.\n"
            f"Current model: {json.dumps(asdict(self._user_model), indent=2)}\n\n"
            f"New utterance: {utterance!r}\n\n"
            "Update the model fields. Return ONLY a JSON object with the updated fields.\n"
            "Fields: research_style, depth_preference, domain_vocabulary (list), "
            "trust_level (0-1), preferred_output_format, known_dislikes (list), "
            "session_history_summary.\n"
        )
        updated = self._call_llm_json(prompt, fallback={})
        for k, v in updated.items():
            if hasattr(self._user_model, k):
                setattr(self._user_model, k, v)
        self._user_model.last_updated = time.time()
        self._save()
        return self._user_model

    def infer_user_intent(self, task_description: str) -> TaskDecomposition:
        """
        Decompose an underspecified task description into explicit goals and constraints.
        Task-ToM layer.
        """
        prompt = (
            "Decompose this research task description into explicit components.\n"
            f"User model: {json.dumps(asdict(self._user_model), indent=2)}\n"
            f"Task: {task_description!r}\n\n"
            "Return JSON with keys: stated_goal, implicit_goal, implicit_constraints (list), "
            "implicit_audience, time_horizon, success_criteria (list), likely_pitfalls (list)."
        )
        decomposed = self._call_llm_json(prompt, fallback={
            "stated_goal": task_description,
            "implicit_goal": task_description,
            "implicit_constraints": [],
            "implicit_audience": "ML researchers",
            "time_horizon": "hours",
            "success_criteria": ["experiment completes", "metric improves"],
            "likely_pitfalls": [],
        })
        return TaskDecomposition(raw_description=task_description, **decomposed)

    @property
    def user_model(self) -> UserModel:
        return self._user_model

    # ------------------------------------------------------------------ #
    #  3. Self-ToM                                                         #
    # ------------------------------------------------------------------ #

    def observe_self(self, action: str, outcome: str, metric: float | None = None) -> None:
        """Record a self-observation for pattern analysis."""
        self._self_observations.append({
            "action": action,
            "outcome": outcome,
            "metric": metric,
            "timestamp": time.time(),
        })
        self._save()

    def analyze_self(self) -> dict[str, Any]:
        """
        Identify own blind spots, failure patterns, and strengths.
        Self-ToM layer — feeds directly into the Evolution Engine.
        """
        if not self._self_observations:
            return {"summary": "No observations yet", "recommendations": []}

        success_count = sum(1 for o in self._self_observations if o["outcome"] == "success")
        total = len(self._self_observations)
        rate = success_count / total if total > 0 else 0.0

        prompt = (
            "Analyze these self-observations of an AI research agent.\n"
            f"Observations (last 20): {json.dumps(self._self_observations[-20:], indent=2)}\n\n"
            "Return JSON with keys: "
            "success_rate (float), common_failure_patterns (list), blind_spots (list), "
            "strengths (list), recommendations (list of config changes to make)."
        )
        analysis = self._call_llm_json(prompt, fallback={
            "success_rate": rate,
            "common_failure_patterns": [],
            "blind_spots": [],
            "strengths": [],
            "recommendations": [],
        })
        return analysis

    # ------------------------------------------------------------------ #
    #  4. Reviewer-ToM                                                     #
    # ------------------------------------------------------------------ #

    def build_reviewer_model(
        self, venue: str, domain: str, recent_papers: list[str] | None = None
    ) -> ReviewerModel:
        """
        Build a model of an adversarial reviewer at venue/domain.
        Called before paper writing starts — converts reactive correction to proactive guide.
        """
        prompt = (
            f"Model the typical reviewer at {venue} for {domain} papers in 2026.\n"
            f"Recent accepted papers (for context): {recent_papers or []}\n\n"
            "Return JSON with keys: hot_topics (list), pet_peeves (list), "
            "required_sections (list), likely_rejection_reasons (list)."
        )
        fields = self._call_llm_json(prompt, fallback={
            "hot_topics": ["scaling", "emergent behavior", "efficiency"],
            "pet_peeves": ["overclaiming", "missing ablations", "weak baselines"],
            "required_sections": ["abstract", "introduction", "related_work",
                                   "method", "experiments", "conclusion"],
            "likely_rejection_reasons": ["incremental contribution", "missing baselines"],
        })
        self._reviewer_model = ReviewerModel(venue=venue, domain=domain, **fields)
        self._save()
        return self._reviewer_model

    def predict_review(self, paper_draft: str) -> dict[str, Any]:
        """
        Predict what a reviewer at the modeled venue will say about a draft.
        Returns predicted score, major concerns, and required changes.
        """
        prompt = (
            f"Reviewer profile: {json.dumps(asdict(self._reviewer_model), indent=2)}\n\n"
            f"Paper draft (first 3000 chars):\n{paper_draft[:3000]}\n\n"
            "Predict the review. Return JSON with keys: "
            "predicted_score (1-10), accept_probability (0-1), "
            "major_concerns (list), required_changes (list), "
            "would_accept (bool), reasoning."
        )
        return self._call_llm_json(prompt, fallback={
            "predicted_score": 5,
            "accept_probability": 0.3,
            "major_concerns": ["insufficient evaluation"],
            "required_changes": ["add baselines", "add ablations"],
            "would_accept": False,
            "reasoning": "Insufficient review data",
        })

    @property
    def reviewer_model(self) -> ReviewerModel:
        return self._reviewer_model

    # ------------------------------------------------------------------ #
    #  7. Community-ToM                                                    #
    # ------------------------------------------------------------------ #

    def update_community_model(
        self, domain: str, paper_abstracts: list[str]
    ) -> CommunityModel:
        """
        Derive the field's collective knowledge state from recent paper abstracts.
        Guides hypothesis generation away from overcrowded areas.
        """
        prompt = (
            f"Analyze these recent paper abstracts from the {domain} field.\n"
            f"Abstracts: {json.dumps(paper_abstracts[:20], indent=2)}\n\n"
            "Return JSON with keys: consensus_beliefs (list), active_debates (list), "
            "open_problems (list), recent_pivots (list), "
            "overcrowded_areas (list), high_impact_areas (list)."
        )
        fields = self._call_llm_json(prompt, fallback={
            "consensus_beliefs": [],
            "active_debates": [],
            "open_problems": [],
            "recent_pivots": [],
            "overcrowded_areas": [],
            "high_impact_areas": [],
        })
        self._community_model = CommunityModel(domain=domain, **fields)
        self._community_model.last_updated = time.time()
        self._save()
        return self._community_model

    @property
    def community_model(self) -> CommunityModel:
        return self._community_model

    # ------------------------------------------------------------------ #
    #  LLM helpers                                                         #
    # ------------------------------------------------------------------ #

    def _call_llm_json(self, prompt: str, fallback: dict[str, Any]) -> dict[str, Any]:
        if self._llm is None:
            logger.debug("No LLM configured — using fallback for ToM call")
            return fallback
        try:
            raw = self._llm(prompt)
            # Strip markdown code fences if present
            raw = raw.strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            return json.loads(raw)
        except Exception as exc:
            logger.warning("ToM LLM call failed: %s — using fallback", exc)
            return fallback

    # ------------------------------------------------------------------ #
    #  Persistence                                                         #
    # ------------------------------------------------------------------ #

    def _save(self) -> None:
        if not self._persist_path:
            return
        self._persist_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "user_model": asdict(self._user_model),
            "reviewer_model": asdict(self._reviewer_model),
            "community_model": asdict(self._community_model),
            "self_observations": self._self_observations,
        }
        tmp = self._persist_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2))
        tmp.replace(self._persist_path)

    def _load(self) -> None:
        data = json.loads(self._persist_path.read_text())
        if "user_model" in data:
            self._user_model = UserModel(**data["user_model"])
        if "reviewer_model" in data:
            self._reviewer_model = ReviewerModel(**data["reviewer_model"])
        if "community_model" in data:
            self._community_model = CommunityModel(**data["community_model"])
        if "self_observations" in data:
            self._self_observations = data["self_observations"]
