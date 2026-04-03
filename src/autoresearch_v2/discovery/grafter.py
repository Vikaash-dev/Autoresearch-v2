from __future__ import annotations

from dataclasses import dataclass

from .branch import BranchState
from .forest import DiscoveryForest


@dataclass(slots=True)
class BranchGrafter:
    min_score: float = 0.7

    def graft(self, forest: DiscoveryForest, source_id: str, target_id: str) -> bool:
        if source_id not in forest.branches or target_id not in forest.branches:
            return False
        source = forest.branches[source_id]
        target = forest.branches[target_id]
        if source.critic_score < self.min_score or target.state == BranchState.PRUNED:
            return False
        target.hypothesis = f"{target.hypothesis} + graft({source.hypothesis})"
        target.state = BranchState.GRAFTED
        return True

