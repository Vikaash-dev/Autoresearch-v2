"""Post-run self-review module."""
from __future__ import annotations
import logging
from typing import Any, Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from ..core.state import RunState
    from ..config.schema import AutoResearchConfig

logger = logging.getLogger(__name__)


class SelfReviewModule:
    """Analyzes a completed run to identify bottlenecks and propose improvements."""

    def __init__(self, config: "AutoResearchConfig") -> None:
        self.config = config

    def analyze(self, state: "RunState") -> Dict[str, Any]:
        bottlenecks = self._find_bottlenecks(state)
        proposed_updates = self._propose_policy_updates(bottlenecks, state)
        quality_scores = self._score_quality(state)

        reflection = {
            "run_id": state.run_id,
            "iteration": state.iteration,
            "bottlenecks": bottlenecks,
            "quality_scores": quality_scores,
            "proposed_policy_updates": proposed_updates,
            "summary": self._summarize(bottlenecks, quality_scores),
        }
        logger.info("Self-review complete: %d bottlenecks, quality=%.2f",
                    len(bottlenecks), quality_scores.get("overall", 0))
        return reflection

    def _find_bottlenecks(self, state: "RunState") -> list[Dict[str, Any]]:
        bottlenecks = []
        if len(state.evidence) < 5:
            bottlenecks.append({
                "module": "literature_miner",
                "issue": "Insufficient evidence collected",
                "severity": "high",
                "metric": len(state.evidence),
                "threshold": 5,
            })
        failed_exps = [e for e in state.experiments if e.get("status") == "failed"]
        if failed_exps:
            bottlenecks.append({
                "module": "executor",
                "issue": f"{len(failed_exps)} experiments failed",
                "severity": "medium",
                "metric": len(failed_exps),
                "threshold": 0,
            })
        if len(state.hypotheses) < 2:
            bottlenecks.append({
                "module": "hypothesis_generator",
                "issue": "Too few hypotheses generated",
                "severity": "medium",
                "metric": len(state.hypotheses),
                "threshold": 2,
            })
        return bottlenecks

    def _propose_policy_updates(self, bottlenecks: list, state: "RunState") -> list[Dict[str, Any]]:
        updates = []
        for b in bottlenecks:
            module = b["module"]
            if module == "literature_miner":
                updates.append({
                    "module": module,
                    "parameter": "arxiv_max_results",
                    "current": self.config.literature.arxiv_max_results,
                    "proposed": self.config.literature.arxiv_max_results * 2,
                    "rationale": "Increase search breadth to find more evidence",
                    "safe": True,
                })
            elif module == "executor":
                updates.append({
                    "module": module,
                    "parameter": "max_retries",
                    "current": self.config.experiment.max_retries,
                    "proposed": self.config.experiment.max_retries + 2,
                    "rationale": "More retries for flaky experiments",
                    "safe": True,
                })
            elif module == "hypothesis_generator":
                updates.append({
                    "module": module,
                    "parameter": "num_hypotheses",
                    "current": 3,
                    "proposed": 5,
                    "rationale": "Generate more diverse hypotheses",
                    "safe": True,
                })
        return updates

    def _score_quality(self, state: "RunState") -> Dict[str, float]:
        evidence_score = min(1.0, len(state.evidence) / 10)
        hyp_score = min(1.0, len(state.hypotheses) / 3)
        exp_score = min(1.0, len([e for e in state.experiments if e.get("status") == "success"]) / max(len(state.experiments), 1))
        review_score = min(1.0, len(state.reviews) / 5)
        overall = (evidence_score + hyp_score + exp_score + review_score) / 4
        return {
            "evidence": round(evidence_score, 2),
            "hypotheses": round(hyp_score, 2),
            "experiments": round(exp_score, 2),
            "reviews": round(review_score, 2),
            "overall": round(overall, 2),
        }

    def _summarize(self, bottlenecks: list, quality: Dict[str, float]) -> str:
        if not bottlenecks:
            return f"Run completed successfully. Overall quality: {quality.get('overall', 0):.0%}"
        issues = ", ".join(b["issue"] for b in bottlenecks)
        return f"Run completed with issues: {issues}. Overall quality: {quality.get('overall', 0):.0%}"
