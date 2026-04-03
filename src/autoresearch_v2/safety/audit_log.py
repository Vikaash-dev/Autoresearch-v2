from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class AuditLog:
    events: list[str] = field(default_factory=list)

    def record(self, event: str) -> None:
        self.events.append(event)

