from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class CodeGateResult:
    passed: bool
    note: str


class CodeGate:
    def verify(self, claim: str, execution_log: list[str]) -> CodeGateResult:
        if execution_log:
            return CodeGateResult(passed=True, note=f"validated by {len(execution_log)} execution entries")
        return CodeGateResult(passed=False, note="no execution proof")

