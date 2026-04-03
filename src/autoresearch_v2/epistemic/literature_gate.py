from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class LiteratureGateResult:
    passed: bool
    note: str


class LiteratureGate:
    def __init__(self, min_results: int = 1, min_relevance: float = 0.05) -> None:
        self.min_results = max(1, min_results)
        self.min_relevance = max(0.0, min_relevance)

    def verify(
        self,
        claim: str,
        citations: list[str],
        evidence: list[dict[str, object]] | None = None,
    ) -> LiteratureGateResult:
        if not citations:
            return LiteratureGateResult(passed=False, note="no citations provided")
        ranked = evidence or []
        if not ranked:
            return LiteratureGateResult(
                passed=False,
                note="citations present but missing ranked evidence",
            )
        strong = [e for e in ranked if float(e.get("score", 0.0)) >= self.min_relevance]
        if len(strong) < self.min_results:
            return LiteratureGateResult(
                passed=False,
                note=(
                    f"insufficient relevant evidence: "
                    f"{len(strong)}/{self.min_results} above score {self.min_relevance}"
                ),
            )
        contradictions = [e for e in strong if "contradict" in str(e.get("chunk", "")).lower()]
        supports = [e for e in strong if e not in contradictions]
        return LiteratureGateResult(
            passed=len(supports) > 0,
            note=(
                f"evidence_ok={len(strong)} supports={len(supports)} "
                f"contradictions={len(contradictions)}"
            ),
        )
