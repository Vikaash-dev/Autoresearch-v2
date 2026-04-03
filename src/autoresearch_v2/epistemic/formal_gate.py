from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class FormalGateResult:
    passed: bool
    note: str


class FormalGate:
    def verify(self, claim: str, required: bool) -> FormalGateResult:
        if not required:
            return FormalGateResult(passed=True, note="formal verification not required")
        return FormalGateResult(passed=False, note="formal verification required but not implemented")

