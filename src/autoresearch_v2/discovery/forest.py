from __future__ import annotations

from dataclasses import dataclass, field

from .branch import Branch, BranchState


@dataclass(slots=True)
class DiscoveryForest:
    branches: dict[str, Branch] = field(default_factory=dict)
    prune_threshold: float = 0.2

    def add_branch(self, branch: Branch) -> None:
        if branch.id in self.branches:
            raise ValueError(f"Branch already exists: {branch.id}")
        self.branches[branch.id] = branch

    def score_branch(self, branch_id: str, score: float) -> Branch:
        branch = self.branches[branch_id]
        branch.update_score(score)
        if score <= self.prune_threshold:
            branch.prune()
        return branch

    def active_branches(self) -> list[Branch]:
        return [b for b in self.branches.values() if b.state != BranchState.PRUNED]

