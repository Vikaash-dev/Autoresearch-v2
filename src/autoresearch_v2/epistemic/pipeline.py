from __future__ import annotations

from dataclasses import dataclass

from .code_gate import CodeGate
from .formal_gate import FormalGate
from .literature_gate import LiteratureGate
from .provenance import ClaimProvenanceGraph, ProvenanceEntry


@dataclass(slots=True)
class EpistemicVerificationResult:
    claim: str
    passed: bool
    details: dict[str, str]


class EpistemicPipeline:
    def __init__(self) -> None:
        self.literature_gate = LiteratureGate()
        self.code_gate = CodeGate()
        self.formal_gate = FormalGate()
        self.provenance = ClaimProvenanceGraph()

    def verify_claim(
        self,
        claim: str,
        citations: list[str],
        execution_log: list[str],
        formal_required: bool = False,
    ) -> EpistemicVerificationResult:
        l = self.literature_gate.verify(claim, citations)
        c = self.code_gate.verify(claim, execution_log)
        f = self.formal_gate.verify(claim, required=formal_required)
        passed = l.passed and c.passed and f.passed
        self.provenance.add_entry(
            ProvenanceEntry(
                claim=claim,
                literature_note=l.note,
                code_note=c.note,
                formal_note=f.note,
            )
        )
        return EpistemicVerificationResult(
            claim=claim,
            passed=passed,
            details={"literature": l.note, "code": c.note, "formal": f.note},
        )

    def get_provenance_graph(self) -> ClaimProvenanceGraph:
        return self.provenance

