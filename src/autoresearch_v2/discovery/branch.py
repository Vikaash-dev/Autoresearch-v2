from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class BranchState(str, Enum):
    SEEDED = "seeded"
    GROWING = "growing"
    PROMISING = "promising"
    STAGNANT = "stagnant"
    PRUNED = "pruned"
    GRAFTED = "grafted"
    HARVESTED = "harvested"


@dataclass(slots=True)
class Branch:
    id: str
    hypothesis: str
    budget: float = 1.0
    state: BranchState = BranchState.SEEDED
    critic_score: float = 0.0

    def update_score(self, score: float) -> None:
        self.critic_score = score
        if self.state == BranchState.PRUNED:
            return
        if score >= 0.8:
            self.state = BranchState.PROMISING
        elif score <= 0.2:
            self.state = BranchState.STAGNANT
        else:
            self.state = BranchState.GROWING

    def prune(self) -> None:
        self.state = BranchState.PRUNED

