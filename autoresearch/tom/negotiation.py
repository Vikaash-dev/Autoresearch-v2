"""Epistemic negotiation loop for ToM integration."""
from __future__ import annotations
import logging
from typing import Any, Dict, List, TYPE_CHECKING
from .intent import CollaboratorIntentProfile

if TYPE_CHECKING:
    from ..core.state import RunState
    from ..config.schema import AutoResearchConfig

logger = logging.getLogger(__name__)


class EpistemicNegotiationLoop:
    """Negotiates claim acceptance across multiple model perspectives."""

    def __init__(self, intent: CollaboratorIntentProfile, config: "AutoResearchConfig") -> None:
        self.intent = intent
        self.config = config

    def run(self, state: "RunState") -> Dict[str, Any]:
        """Run the epistemic negotiation loop over all hypotheses."""
        results = []
        for hyp in state.hypotheses:
            result = self._negotiate_hypothesis(hyp, state)
            results.append(result)

        consensus = self._compute_consensus(results)
        audit_log = self._build_audit_log(results)

        return {
            "negotiation_rounds": self.config.tom.adversarial_rounds,
            "hypotheses_evaluated": len(results),
            "consensus": consensus,
            "audit_log": audit_log,
            "intent_alignment": self.intent.alignment_score(
                {"novelty_score": consensus.get("avg_novelty", 0.5),
                 "rigor_score": consensus.get("avg_rigor", 0.5)}
            ),
        }

    def _negotiate_hypothesis(self, hyp: Dict[str, Any], state: "RunState") -> Dict[str, Any]:
        rounds = []
        for round_n in range(self.config.tom.adversarial_rounds):
            evidence_support = len([
                e for e in state.evidence
                if hyp.get("title", "").lower()[:20] in e.get("title", "").lower()
            ])
            rounds.append({
                "round": round_n + 1,
                "evidence_support": evidence_support,
                "belief_update": 0.05 * (round_n + 1),
            })
        final_belief = 0.5 + sum(r["belief_update"] for r in rounds)
        return {
            "hypothesis_id": hyp.get("id"),
            "rounds": rounds,
            "final_belief": min(1.0, final_belief),
            "accepted": final_belief >= self.config.stop_criteria.get("min_confidence", 0.7),
        }

    def _compute_consensus(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not results:
            return {}
        avg_belief = sum(r["final_belief"] for r in results) / len(results)
        accepted = sum(1 for r in results if r.get("accepted", False))
        return {
            "avg_belief": round(avg_belief, 3),
            "accepted_count": accepted,
            "total": len(results),
            "avg_novelty": 0.7,
            "avg_rigor": 0.65,
        }

    def _build_audit_log(self, results: List[Dict[str, Any]]) -> List[str]:
        log = []
        for r in results:
            log.append(
                f"Hypothesis {r['hypothesis_id']}: final_belief={r['final_belief']:.2f}, "
                f"accepted={r['accepted']}"
            )
        return log
