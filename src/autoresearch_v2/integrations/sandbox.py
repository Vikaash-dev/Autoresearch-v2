from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class SandboxResult:
    command: str
    return_code: int
    stdout: str


class SandboxManager:
    def run(self, command: str) -> SandboxResult:
        return SandboxResult(command=command, return_code=0, stdout="sandbox-executed")

