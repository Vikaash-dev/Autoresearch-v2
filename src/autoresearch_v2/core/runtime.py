from __future__ import annotations

from dataclasses import dataclass

from autoresearch_v2.discovery.branch import Branch
from autoresearch_v2.discovery.forest import DiscoveryForest
from autoresearch_v2.dog.graph import DynamicObjectiveGraph
from autoresearch_v2.dog.objective import Objective
from autoresearch_v2.epistemic.verification import ZeroTrustEpistemicVerifier
from autoresearch_v2.tom.council import ToMReviewerCouncil


@dataclass(slots=True)
class RunSummary:
    topic: str
    branch_count: int
    ready_objectives: int
    accepted_by_council: bool
    verification_passed: bool


def run_bootstrap(topic: str, branches: int) -> RunSummary:
    graph = DynamicObjectiveGraph()
    graph.add_objective(Objective(id="literature", description=f"Ground topic: {topic}", priority=1.0))
    graph.add_objective(
        Objective(
            id="hypothesis",
            description="Generate hypotheses",
            dependencies=["literature"],
            priority=0.9,
        )
    )
    graph.mark_complete("literature", outputs={"papers": 10})
    ready = graph.ready_objectives()

    forest = DiscoveryForest()
    for i in range(branches):
        forest.add_branch(Branch(id=f"branch-{i+1}", hypothesis=f"H{i+1} for {topic}", budget=1.0))
        forest.score_branch(f"branch-{i+1}", score=0.7 if i == 0 else 0.1)

    council = ToMReviewerCouncil()
    review = council.review(f"Topic: {topic}. Active branches: {len(forest.active_branches())}")

    verifier = ZeroTrustEpistemicVerifier()
    verification = verifier.verify(
        claim=f"Bootstrap run for {topic}",
        has_citation=True,
        has_log_proof=True,
        formal_required=False,
    )

    return RunSummary(
        topic=topic,
        branch_count=branches,
        ready_objectives=len(ready),
        accepted_by_council=review.accepted,
        verification_passed=verification.passed,
    )
