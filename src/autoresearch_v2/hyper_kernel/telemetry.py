from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class RunTelemetry:
    agent_latency: dict[str, float] = field(default_factory=dict)
    failure_rates: dict[str, float] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

