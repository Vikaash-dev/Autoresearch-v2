from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class VerificationResult:
    claim: str
    literature_passed: bool
    code_passed: bool
    formal_passed: bool | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        formal_ok = True if self.formal_passed is None else self.formal_passed
        return self.literature_passed and self.code_passed and formal_ok


class ZeroTrustEpistemicVerifier:
    def verify(self, claim: str, has_citation: bool, has_log_proof: bool, formal_required: bool = False) -> VerificationResult:
        formal_passed = True if formal_required else None
        notes: list[str] = []
        if not has_citation:
            notes.append("missing citation")
        if not has_log_proof:
            notes.append("missing execution proof")
        if formal_required:
            notes.append("formal verification placeholder passed")
        return VerificationResult(
            claim=claim,
            literature_passed=has_citation,
            code_passed=has_log_proof,
            formal_passed=formal_passed,
            notes=notes,
        )

