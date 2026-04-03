import unittest

from autoresearch_v2.compute.planner import ComputePlanner
from autoresearch_v2.compute.simulator import ComputeSimulator
from autoresearch_v2.discovery.grafter import BranchGrafter
from autoresearch_v2.discovery.pruner import BranchPruner
from autoresearch_v2.core.runtime import run_bootstrap
from autoresearch_v2.discovery.branch import Branch, BranchState
from autoresearch_v2.discovery.forest import DiscoveryForest
from autoresearch_v2.dog.allocator import ComputeBudgetAllocator
from autoresearch_v2.dog.graph import DynamicObjectiveGraph
from autoresearch_v2.dog.objective import Objective
from autoresearch_v2.dog.scheduler import ObjectiveScheduler
from autoresearch_v2.epistemic.pipeline import EpistemicPipeline
from autoresearch_v2.epistemic.verification import ZeroTrustEpistemicVerifier
from autoresearch_v2.hyper_kernel.kernel import HyperKernel
from autoresearch_v2.hyper_kernel.modifier import BoundedModifier
from autoresearch_v2.hyper_kernel.telemetry import RunTelemetry
from autoresearch_v2.tom.dialogue import AdversarialDialogueProtocol
from autoresearch_v2.tom.intent_model import CollaboratorIntentModel
from autoresearch_v2.tom.council import ToMReviewerCouncil
from autoresearch_v2.retrieval.semantic import LocalCorpusIngestor, LocalSemanticRetriever


class TestDOG(unittest.TestCase):
    def test_ready_objectives_respect_dependencies(self) -> None:
        graph = DynamicObjectiveGraph()
        graph.add_objective(Objective(id="a", description="a"))
        graph.add_objective(Objective(id="b", description="b", dependencies=["a"]))
        self.assertEqual([o.id for o in graph.ready_objectives()], ["a"])
        graph.mark_complete("a")
        self.assertEqual([o.id for o in graph.ready_objectives()], ["b"])

    def test_scheduler_and_allocator(self) -> None:
        graph = DynamicObjectiveGraph()
        graph.add_objective(Objective(id="a", description="a", priority=1.0))
        scheduler = ObjectiveScheduler(graph)
        assignments = scheduler.schedule(["agent-x"], limit=1)
        self.assertEqual(len(assignments), 1)
        allocator = ComputeBudgetAllocator(total_budget=10.0)
        budget = allocator.allocate([a.objective_id for a in assignments])
        self.assertEqual(budget["a"], 10.0)


class TestDiscovery(unittest.TestCase):
    def test_branch_pruning_threshold(self) -> None:
        forest = DiscoveryForest(prune_threshold=0.2)
        forest.add_branch(Branch(id="b1", hypothesis="h1"))
        result = forest.score_branch("b1", 0.1)
        self.assertEqual(result.state, BranchState.PRUNED)
        self.assertEqual(forest.active_branches(), [])

    def test_pruner_and_grafter(self) -> None:
        forest = DiscoveryForest(prune_threshold=0.1)
        forest.add_branch(Branch(id="b1", hypothesis="h1"))
        forest.add_branch(Branch(id="b2", hypothesis="h2"))
        forest.score_branch("b1", 0.8)
        forest.score_branch("b2", 0.4)
        grafter = BranchGrafter(min_score=0.6)
        self.assertTrue(grafter.graft(forest, "b1", "b2"))
        self.assertEqual(forest.branches["b2"].state, BranchState.GRAFTED)
        pruner = BranchPruner(prune_floor=0.5)
        pruned = pruner.prune(forest)
        self.assertIn("b2", pruned)


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

    def test_zero_trust_verifier_fails_when_formal_required_placeholder(self) -> None:
        verifier = ZeroTrustEpistemicVerifier()
        result = verifier.verify("claim", has_citation=True, has_log_proof=True, formal_required=True)
        self.assertFalse(result.passed)
        self.assertIn("formal verification required but not yet implemented", result.notes)

    def test_dialogue_intent_and_epistemic_pipeline(self) -> None:
        council = ToMReviewerCouncil()
        dialogue = AdversarialDialogueProtocol(council=council, max_rounds=2)
        dialogue_result = dialogue.run("draft")
        self.assertGreaterEqual(len(dialogue_result.rounds), 1)

        cim = CollaboratorIntentModel(risk_appetite=0.5, rigor_vs_speed=0.8, venue_bias=0.5, collaboration_style=0.5)
        adjusted = cim.adjust_budgets({"verification": 10.0, "exploration": 10.0})
        self.assertGreater(adjusted["verification"], 10.0)

        pipeline = EpistemicPipeline()
        result = pipeline.verify_claim(
            "claim",
            citations=["c1"],
            evidence=[{"score": 0.6, "chunk": "supports claim with evidence"}],
            execution_log=["ok"],
            formal_required=False,
        )
        self.assertTrue(result.passed)
        self.assertEqual(len(pipeline.get_provenance_graph().entries), 1)

    def test_compute_and_hyperkernel(self) -> None:
        plan = ComputePlanner(total_budget=90).plan({"a": 2, "b": 1})
        self.assertAlmostEqual(plan["a"] + plan["b"], 90.0)
        sim = ComputeSimulator()
        self.assertEqual(sim.estimate_duration(3, avg_seconds_per_objective=2), 6)

        hk = HyperKernel(modifier=BoundedModifier())
        telemetry = RunTelemetry(agent_latency={"writer": 12.0}, failure_rates={"writer": 0.0}, notes=[])
        mod = hk.optimize_agent("writer", telemetry)
        self.assertIsNotNone(mod)
        self.assertTrue(hk.apply_modification(mod))


class TestRuntime(unittest.TestCase):
    def test_run_bootstrap_summary(self) -> None:
        summary = run_bootstrap("test topic", 2)
        self.assertEqual(summary.topic, "test topic")
        self.assertEqual(summary.branch_count, 2)
        self.assertGreaterEqual(summary.ready_objectives, 1)
        self.assertTrue(summary.verification_passed)
        self.assertGreaterEqual(summary.provenance_entries, 1)
        self.assertIn("retrieval_mode=offline_local_semantic", summary.telemetry_notes)


class TestRetrieval(unittest.TestCase):
    def test_local_retriever_ranks_hyperagents(self) -> None:
        retriever = LocalSemanticRetriever(documents=LocalCorpusIngestor().default_corpus())
        results = retriever.search("self-referential self-improving hyperagents", top_k=3)
        self.assertGreaterEqual(len(results), 1)
        self.assertEqual(results[0].doc_id, "arxiv:2603.19461")

    def test_index_persistence_round_trip(self) -> None:
        retriever = LocalSemanticRetriever(
            index_path="artifacts/test_offline_semantic_index.json",
            documents=LocalCorpusIngestor().default_corpus(),
        )
        first = retriever.search("software engineering coding tasks", top_k=1)
        reloaded = LocalSemanticRetriever(index_path="artifacts/test_offline_semantic_index.json")
        second = reloaded.search("software engineering coding tasks", top_k=1)
        self.assertEqual(len(first), len(second))
        self.assertEqual(first[0].doc_id, second[0].doc_id)


if __name__ == "__main__":
    unittest.main()
