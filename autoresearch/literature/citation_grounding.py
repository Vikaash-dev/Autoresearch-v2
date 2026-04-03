"""Citation grounding subsystem: verify claim-to-source linkage."""
from __future__ import annotations
import logging
from typing import Dict, List, Optional
from .evidence_registry import EvidenceItem

logger = logging.getLogger(__name__)


class CitationGrounding:
    """Verifies that claims are grounded in evidence items."""

    def __init__(self, evidence_items: List[EvidenceItem]) -> None:
        self.evidence = evidence_items

    def verify_claim(self, claim: str) -> Dict[str, object]:
        """Check if a claim is supported by any evidence item."""
        claim_lower = claim.lower()
        supporting = []
        for item in self.evidence:
            overlap = self._keyword_overlap(claim_lower, item.title.lower() + " " + item.abstract.lower())
            if overlap > 0.1:
                supporting.append({"id": item.id, "title": item.title, "overlap": round(overlap, 3)})
        verified = len(supporting) > 0
        return {
            "claim": claim,
            "verified": verified,
            "supporting_evidence": sorted(supporting, key=lambda x: -x["overlap"])[:3],
            "confidence": min(1.0, sum(s["overlap"] for s in supporting)),
        }

    def ground_all(self, claims: List[str]) -> List[Dict[str, object]]:
        return [self.verify_claim(c) for c in claims]

    @staticmethod
    def _keyword_overlap(text_a: str, text_b: str) -> float:
        words_a = set(text_a.split())
        words_b = set(text_b.split())
        if not words_a:
            return 0.0
        return len(words_a & words_b) / len(words_a)
