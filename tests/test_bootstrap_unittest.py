import unittest

from autoresearch_v2.core.runtime import run_bootstrap
from autoresearch_v2.discovery.branch import Branch, BranchState
from autoresearch_v2.discovery.forest import DiscoveryForest
from autoresearch_v2.dog.graph import DynamicObjectiveGraph
from autoresearch_v2.dog.objective import Objective
from autoresearch_v2.epistemic.verification import ZeroTrustEpistemicVerifier
from autoresearch_v2.tom.council import ToMReviewerCouncil


class TestDOG(unittest.TestCase):
    def test_ready_objectives_respect_dependencies(self) -> None:
        graph = DynamicObjectiveGraph()
        graph.add_objective(Objective(id="a", description="a"))
        graph.add_objective(Objective(id="b", description="b", dependencies=["a"]))
        self.assertEqual([o.id for o in graph.ready_objectives()], ["a"])
        graph.mark_complete("a")
        self.assertEqual([o.id for o in graph.ready_objectives()], ["b"])


class TestDiscovery(unittest.TestCase):
    def test_branch_pruning_threshold(self) -> None:
        forest = DiscoveryForest(prune_threshold=0.2)
        forest.add_branch(Branch(id="b1", hypothesis="h1"))
        result = forest.score_branch("b1", 0.1)
        self.assertEqual(result.state, BranchState.PRUNED)
        self.assertEqual(forest.active_branches(), [])


class TestToMEpistemic(unittest.TestCase):
    def test_council_returns_scores(self) -> None:
        council = ToMReviewerCouncil()
        outcome = council.review("Short manuscript")
        self.assertEqual(len(outcome.scores), 5)
        self.assertIsInstance(outcome.accepted, bool)

    def test_zero_trust_verifier_fails_without_evidence(self) -> None:
        verifier = ZeroTrustEpistemicVerifier()
        result = verifier.verify("claim", has_citation=False, has_log_proof=True)
        self.assertFalse(result.passed)
        self.assertIn("missing citation", result.notes)


class TestRuntime(unittest.TestCase):
    def test_run_bootstrap_summary(self) -> None:
        summary = run_bootstrap("test topic", 2)
        self.assertEqual(summary.topic, "test topic")
        self.assertEqual(summary.branch_count, 2)
        self.assertGreaterEqual(summary.ready_objectives, 1)
        self.assertTrue(summary.verification_passed)


if __name__ == "__main__":
    unittest.main()

