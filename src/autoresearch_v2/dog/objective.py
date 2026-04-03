from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ObjectiveStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETE = "complete"
    FAILED = "failed"
    PRUNED = "pruned"


@dataclass(slots=True)
class Objective:
    id: str
    description: str
    status: ObjectiveStatus = ObjectiveStatus.PENDING
    priority: float = 1.0
    assigned_agent: str | None = None
    dependencies: list[str] = field(default_factory=list)
    outputs: dict[str, Any] = field(default_factory=dict)
    compute_budget: float = 1.0
    verification_required: bool = True
