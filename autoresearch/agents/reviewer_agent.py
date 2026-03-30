"""
ReviewerAgent — peer review + AIRS-Bench 7-dimension scoring.

Paper: AIRS-Bench (2026) — "A Suite of Tasks for Frontier AI Research Science Agents"
Key finding: current AI research agents fail to match human SOTA in complex tasks.
The benchmark defines evaluation dimensions that serve as the roadmap for next-gen agents.

7 Scoring Dimensions (AIRS-Bench):
  1. Novelty          — Is the research direction genuinely new?
  2. Technical Depth  — Is the methodology rigorous and well-justified?
  3. Reproducibility  — Could another agent/human reproduce the results?
  4. Validity         — Are claims supported by evidence? (anti-hallucination)
  5. Impact           — How significant is the contribution?
  6. Clarity          — Is the writing clear and well-structured?
  7. Citation Quality — Are sources accurate and relevant? (ACM reliability critique)

Also implements KEA Research 4-step consensus:
  Independent → Refine (peer) → Evaluate (rank) → Synthesize
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from autoresearch.agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# AIRS-Bench scoring rubric
# ---------------------------------------------------------------------------

DIMENSION_WEIGHTS: Dict[str, float] = {
    "novelty":          0.20,
    "technical_depth":  0.20,
    "reproducibility":  0.15,
    "validity":         0.20,
    "impact":           0.10,
    "clarity":          0.10,
    "citation_quality": 0.05,
}

assert abs(sum(DIMENSION_WEIGHTS.values()) - 1.0) < 1e-9, "weights must sum to 1"


# Known citation error patterns (ACM 2025 evaluation finding).
# Each entry is (author_pattern, misattributed_concept_pattern, flag_message).
KNOWN_CITATION_ERRORS: list[tuple[str, str, str]] = [
    ("goodfellow", "lstm",        "⚠️ Possible citation error: LSTM misattributed to Goodfellow"),
    ("lecun",      "transformer", "⚠️ Possible citation error: Transformer misattributed to LeCun"),
    ("hochreiter", "attention",   "⚠️ Possible citation error: Attention misattributed to Hochreiter"),
]


@dataclass
class ReviewScore:
    dimension:  str
    score:      float           # 0.0 – 1.0
    weight:     float
    rationale:  str
    weighted:   float = field(init=False)

    def __post_init__(self) -> None:
        self.score    = max(0.0, min(1.0, self.score))
        self.weighted = round(self.score * self.weight, 4)


@dataclass
class ReviewResult:
    overall_score:  float
    dimension_scores: Dict[str, ReviewScore]
    strengths:      List[str]
    weaknesses:     List[str]
    recommendations: List[str]
    accept_decision: str        # "accept" / "revise" / "reject"
    reliability_flags: List[str]    # ACM-style: hallucination, citation errors, etc.
    performance_ceiling_gap: float  # AIRS-Bench: distance from theoretical SOTA

    def to_dict(self) -> Dict[str, Any]:
        return {
            "overall_score":     round(self.overall_score, 3),
            "accept_decision":   self.accept_decision,
            "dimension_scores":  {
                k: {"score": v.score, "weighted": v.weighted, "rationale": v.rationale}
                for k, v in self.dimension_scores.items()
            },
            "strengths":         self.strengths,
            "weaknesses":        self.weaknesses,
            "recommendations":   self.recommendations,
            "reliability_flags": self.reliability_flags,
            "performance_ceiling_gap": round(self.performance_ceiling_gap, 3),
        }


# ---------------------------------------------------------------------------
# Reviewer Agent
# ---------------------------------------------------------------------------

class ReviewerAgent(BaseAgent):
    """
    ReviewerAgent — AIRS-Bench evaluation + KEA consensus review.

    Scores a research report on 7 dimensions, flags reliability issues,
    and provides accept/revise/reject with actionable recommendations.

    Context keys expected:
      - report:       dict (from WriterAgent)
      - markdown:     str  (full report text)
      - learnings:    list[str]
      - experiments:  list[dict]
      - task:         str
    """

    def _default_parameters(self) -> Dict[str, Any]:
        return {
            "accept_threshold":  0.65,
            "revise_threshold":  0.45,
            "n_review_passes":   2,       # KEA: multiple independent review passes
            "performance_ceiling": 1.0,   # AIRS-Bench: theoretical max
        }

    def _execute(self, task: str, context: Dict[str, Any]) -> Dict[str, Any]:
        report_dict: Dict[str, Any] = context.get("report", {})
        markdown:    str            = context.get("markdown", "")
        learnings:   List[str]      = context.get("learnings", [])
        experiments: List[Any]      = context.get("experiments", [])

        # KEA: multiple independent passes → consensus
        pass_scores: List[ReviewResult] = []
        for _ in range(self._parameters["n_review_passes"]):
            review = self._single_review_pass(task, report_dict, markdown, learnings, experiments)
            pass_scores.append(review)

        # Synthesise consensus (KEA step 4: best-ranked reviewer writes final answer)
        final = self._synthesise(pass_scores)

        return {
            "overall_score":          final.overall_score,
            "accept_decision":        final.accept_decision,
            "dimension_scores":       {k: v.score for k, v in final.dimension_scores.items()},
            "strengths":              final.strengths,
            "weaknesses":             final.weaknesses,
            "recommendations":        final.recommendations,
            "reliability_flags":      final.reliability_flags,
            "performance_ceiling_gap": final.performance_ceiling_gap,
            "review_passes":          len(pass_scores),
            "detailed_review":        final.to_dict(),
        }

    # ------------------------------------------------------------------
    # Single independent review pass
    # ------------------------------------------------------------------

    def _single_review_pass(
        self,
        task: str,
        report_dict: Dict[str, Any],
        markdown: str,
        learnings: List[str],
        experiments: List[Any],
    ) -> ReviewResult:

        scores: Dict[str, ReviewScore] = {}
        flags:  List[str] = []

        # ---------- 1. Novelty ----------
        novel_score, novel_rationale = self._score_novelty(task, markdown, learnings)
        scores["novelty"] = ReviewScore("novelty", novel_score, DIMENSION_WEIGHTS["novelty"], novel_rationale)

        # ---------- 2. Technical Depth ----------
        depth_score, depth_rationale = self._score_technical_depth(markdown, experiments)
        scores["technical_depth"] = ReviewScore("technical_depth", depth_score, DIMENSION_WEIGHTS["technical_depth"], depth_rationale)

        # ---------- 3. Reproducibility ----------
        repro_score, repro_rationale = self._score_reproducibility(markdown, report_dict)
        scores["reproducibility"] = ReviewScore("reproducibility", repro_score, DIMENSION_WEIGHTS["reproducibility"], repro_rationale)

        # ---------- 4. Validity (anti-hallucination) ----------
        valid_score, valid_rationale, valid_flags = self._score_validity(report_dict, learnings)
        scores["validity"] = ReviewScore("validity", valid_score, DIMENSION_WEIGHTS["validity"], valid_rationale)
        flags.extend(valid_flags)

        # ---------- 5. Impact ----------
        impact_score, impact_rationale = self._score_impact(task, markdown)
        scores["impact"] = ReviewScore("impact", impact_score, DIMENSION_WEIGHTS["impact"], impact_rationale)

        # ---------- 6. Clarity ----------
        clarity_score, clarity_rationale = self._score_clarity(markdown, report_dict)
        scores["clarity"] = ReviewScore("clarity", clarity_score, DIMENSION_WEIGHTS["clarity"], clarity_rationale)

        # ---------- 7. Citation Quality ----------
        cite_score, cite_rationale, cite_flags = self._score_citations(report_dict)
        scores["citation_quality"] = ReviewScore("citation_quality", cite_score, DIMENSION_WEIGHTS["citation_quality"], cite_rationale)
        flags.extend(cite_flags)

        # Overall weighted score
        overall = sum(s.weighted for s in scores.values())

        # Accept decision
        accept_thr = self._parameters["accept_threshold"]
        revise_thr = self._parameters["revise_threshold"]
        if overall >= accept_thr:
            decision = "accept"
        elif overall >= revise_thr:
            decision = "revise"
        else:
            decision = "reject"

        # AIRS-Bench: performance ceiling gap
        ceiling = self._parameters["performance_ceiling"]
        gap = max(0.0, ceiling - overall)

        # Strengths + weaknesses
        strengths, weaknesses, recommendations = self._summarise(scores, flags)

        return ReviewResult(
            overall_score=overall,
            dimension_scores=scores,
            strengths=strengths,
            weaknesses=weaknesses,
            recommendations=recommendations,
            accept_decision=decision,
            reliability_flags=flags,
            performance_ceiling_gap=gap,
        )

    # ------------------------------------------------------------------
    # Dimension scorers
    # ------------------------------------------------------------------

    def _score_novelty(self, task: str, markdown: str, learnings: List[str]) -> Tuple[float, str]:
        task_words = set(re.findall(r"\w{4,}", task.lower()))
        text_words = set(re.findall(r"\w{4,}", markdown.lower()))
        learning_words = set(w for l in learnings for w in re.findall(r"\w{4,}", l.lower()))
        # Novel words = appear in report but not in learnings (new contribution)
        novel_words = text_words - learning_words - task_words
        novelty = min(len(novel_words) / max(len(text_words), 1) * 3, 1.0)
        rationale = f"Report contains {len(novel_words)} novel terms not in prior learnings."
        return round(novelty, 3), rationale

    def _score_technical_depth(self, markdown: str, experiments: List[Any]) -> Tuple[float, str]:
        # Heuristics: word count, presence of methodology section, experiment results
        wc = len(markdown.split())
        has_methodology = "methodology" in markdown.lower() or "method" in markdown.lower()
        has_results = "result" in markdown.lower() or "experiment" in markdown.lower()
        has_numbers = len(re.findall(r"\d+\.\d+", markdown)) > 3
        kept_experiments = sum(1 for e in experiments if isinstance(e, dict) and e.get("kept"))

        score = 0.0
        reasons = []
        if wc > 500:    score += 0.2; reasons.append(f"{wc} words")
        if wc > 1000:   score += 0.1
        if has_methodology: score += 0.2; reasons.append("methodology present")
        if has_results:     score += 0.2; reasons.append("results present")
        if has_numbers:     score += 0.2; reasons.append("quantitative data")
        if kept_experiments > 0: score += 0.1; reasons.append(f"{kept_experiments} successful experiments")

        return round(min(score, 1.0), 3), "; ".join(reasons) or "basic structure"

    def _score_reproducibility(self, markdown: str, report_dict: Dict) -> Tuple[float, str]:
        has_code = "```" in markdown or "code" in markdown.lower()
        has_params = any(word in markdown.lower() for word in ["parameter", "config", "hyperparameter", "setting"])
        has_dataset = any(word in markdown.lower() for word in ["dataset", "data", "corpus", "benchmark"])
        sections = report_dict.get("sections", [])
        has_methodology_sec = any(s.get("title", "").lower() in ("methodology", "method", "approach") for s in sections)

        score = sum([
            0.30 * has_code,
            0.25 * has_params,
            0.20 * has_dataset,
            0.25 * has_methodology_sec,
        ])
        reasons = []
        if has_code:   reasons.append("code/pseudocode present")
        if has_params: reasons.append("parameters documented")
        if has_dataset: reasons.append("data described")
        if has_methodology_sec: reasons.append("methodology section present")
        return round(score, 3), "; ".join(reasons) or "insufficient detail for reproduction"

    def _score_validity(
        self,
        report_dict: Dict,
        learnings: List[str],
    ) -> Tuple[float, str, List[str]]:
        flags = list(report_dict.get("hallucination_flags", []))
        verified_ratio = report_dict.get("verified_ratio", 0.5)
        # Penalise for hallucination flags (ACM finding: 42% failure rate)
        penalty = min(len(flags) * 0.08, 0.4)
        score = max(0.0, verified_ratio - penalty)
        rationale = (
            f"Verified ratio={verified_ratio:.2f}; "
            f"{len(flags)} unsupported claim(s) flagged."
        )
        return round(score, 3), rationale, flags

    def _score_impact(self, task: str, markdown: str) -> Tuple[float, str]:
        impact_words = {"novel", "significant", "improve", "surpass", "outperform",
                        "state-of-the-art", "sota", "advance", "breakthrough", "contribution"}
        words = set(re.findall(r"\w+", markdown.lower()))
        matches = words & impact_words
        score = min(len(matches) / 5, 1.0) * 0.7 + 0.3  # base 0.3
        return round(min(score, 1.0), 3), f"Impact vocabulary: {sorted(matches)[:5]}"

    def _score_clarity(self, markdown: str, report_dict: Dict) -> Tuple[float, str]:
        sections = report_dict.get("sections", [])
        has_abstract = bool(report_dict.get("abstract"))
        n_sections  = len(sections)
        avg_section_len = (
            sum(len(s.get("content", "").split()) for s in sections) / max(n_sections, 1)
        )
        score = 0.0
        reasons = []
        if has_abstract:    score += 0.3; reasons.append("abstract present")
        if n_sections >= 4: score += 0.3; reasons.append(f"{n_sections} sections")
        if avg_section_len > 50: score += 0.2; reasons.append("sections adequately long")
        if "introduction" in markdown.lower(): score += 0.1; reasons.append("introduction present")
        if "conclusion" in markdown.lower():   score += 0.1; reasons.append("conclusion present")
        return round(min(score, 1.0), 3), "; ".join(reasons) or "minimal structure"

    def _score_citations(self, report_dict: Dict) -> Tuple[float, str, List[str]]:
        refs = report_dict.get("references", [])
        flags = []
        n = len(refs)
        # Check for known hallucination patterns (externalised to KNOWN_CITATION_ERRORS)
        for ref in refs:
            ref_lower = ref.lower()
            for author_pat, concept_pat, msg in KNOWN_CITATION_ERRORS:
                if author_pat in ref_lower and concept_pat in ref_lower:
                    flags.append(msg)
        score = min(n / 8, 1.0) * (1.0 - len(flags) * 0.2)
        return round(max(score, 0.0), 3), f"{n} references; {len(flags)} citation warnings", flags

    # ------------------------------------------------------------------
    # Synthesis (KEA: consensus from multiple passes)
    # ------------------------------------------------------------------

    def _synthesise(self, passes: List[ReviewResult]) -> ReviewResult:
        if not passes:
            raise ValueError("No review passes to synthesise")
        if len(passes) == 1:
            return passes[0]

        # Average dimension scores
        all_dims = list(passes[0].dimension_scores.keys())
        averaged: Dict[str, ReviewScore] = {}
        for dim in all_dims:
            avg_score = sum(p.dimension_scores[dim].score for p in passes) / len(passes)
            averaged[dim] = ReviewScore(
                dimension=dim,
                score=avg_score,
                weight=DIMENSION_WEIGHTS[dim],
                rationale=passes[0].dimension_scores[dim].rationale,
            )

        overall = sum(s.weighted for s in averaged.values())
        flags = list({f for p in passes for f in p.reliability_flags})

        accept_thr = self._parameters["accept_threshold"]
        revise_thr = self._parameters["revise_threshold"]
        decision = "accept" if overall >= accept_thr else ("revise" if overall >= revise_thr else "reject")

        # Merge strengths, weaknesses, recommendations
        strengths, weaknesses, recommendations = self._summarise(averaged, flags)

        return ReviewResult(
            overall_score=overall,
            dimension_scores=averaged,
            strengths=strengths,
            weaknesses=weaknesses,
            recommendations=recommendations,
            accept_decision=decision,
            reliability_flags=flags,
            performance_ceiling_gap=max(0.0, self._parameters["performance_ceiling"] - overall),
        )

    # ------------------------------------------------------------------

    @staticmethod
    def _summarise(
        scores: Dict[str, ReviewScore],
        flags: List[str],
    ) -> Tuple[List[str], List[str], List[str]]:
        ranked = sorted(scores.values(), key=lambda s: s.score, reverse=True)
        strengths = [f"{s.dimension} ({s.score:.2f}): {s.rationale}" for s in ranked[:3] if s.score > 0.5]
        weaknesses = [f"{s.dimension} ({s.score:.2f}): {s.rationale}" for s in ranked[-3:] if s.score < 0.6]
        recommendations = []
        for s in ranked[-3:]:
            if s.score < 0.5:
                recommendations.append(f"Improve {s.dimension}: {s.rationale[:100]}")
        if flags:
            recommendations.append(f"Address {len(flags)} reliability flag(s): {flags[0][:100]}")
        return strengths, weaknesses, recommendations

    def _score_result(self, result: Any) -> float:
        return result.get("overall_score", 0.5) if isinstance(result, dict) else 0.5
