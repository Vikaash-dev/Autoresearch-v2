from __future__ import annotations

from dataclasses import dataclass

from .forest import DiscoveryForest


@dataclass(slots=True)
class BranchPruner:
    prune_floor: float = 0.2

    def prune(self, forest: DiscoveryForest) -> list[str]:
        pruned: list[str] = []
        for branch_id, branch in forest.branches.items():
            if branch.critic_score <= self.prune_floor:
                branch.prune()
                pruned.append(branch_id)
        return pruned

