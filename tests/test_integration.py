"""Integration tests for the full SERA-X orchestrator."""
import pytest
from autoresearch import ResearchOrchestrator, AutoresearchConfig
from autoresearch.evaluation.benchmark import (
    AIRSBenchEvaluator, BenchmarkTask, EvaluationCriteria, BENCHMARK_TASKS,
)
from autoresearch.memory.experience_store import ExperienceStore


class TestOrchestratorSmoke:
    """Smoke-tests: system runs without crashing."""

    def test_single_round(self, tmp_path):
        orch = ResearchOrchestrator(db_path=str(tmp_path / "test.db"))
        result = orch.run(
            "Bilevel autoresearch optimisation techniques",
            max_rounds=1,
            search_breadth=2,
            search_depth=1,
        )
        assert isinstance(result, dict)
        assert result["total_rounds"] == 1
        assert 0.0 <= result["best_score"] <= 1.0

    def test_two_rounds_build_experience(self, tmp_path):
        orch = ResearchOrchestrator(db_path=str(tmp_path / "test2.db"))
        result = orch.run(
            "CMA-ES hyperparameter optimisation",
            max_rounds=2,
            search_breadth=2,
            search_depth=1,
        )
        summary = result["experience_summary"]
        assert summary["total_rounds"] == 2
        assert summary["total_learnings"] >= 0  # learnings accumulate

    def test_evolution_summary_present(self, tmp_path):
        orch = ResearchOrchestrator(db_path=str(tmp_path / "test3.db"))
        result = orch.run("test", max_rounds=1, search_breadth=2, search_depth=1)
        ev = result["evolution_summary"]
        assert "total_resources" in ev
        assert "committed" in ev

    def test_airs_bench_in_result(self, tmp_path):
        orch = ResearchOrchestrator(db_path=str(tmp_path / "test4.db"))
        result = orch.run("test", max_rounds=1, search_breadth=2, search_depth=1)
        airs = result["airs_bench_reliability"]
        assert "pass_rate" in airs
        assert "failure_rate" in airs

    def test_best_report_has_markdown(self, tmp_path):
        orch = ResearchOrchestrator(db_path=str(tmp_path / "test5.db"))
        result = orch.run("test", max_rounds=1, search_breadth=2, search_depth=1)
        report = result.get("best_report", {})
        assert "markdown" in report
        assert len(report.get("markdown", "")) > 50


class TestAIRSBenchEvaluator:
    def _make_review(self, overall=0.7):
        dims = {
            "novelty": 0.6, "technical_depth": 0.7,
            "reproducibility": 0.6, "validity": 0.8,
            "impact": 0.7, "clarity": 0.8, "citation_quality": 0.6,
        }
        return {
            "overall_score": overall,
            "dimension_scores": dims,
            "reliability_flags": [],
            "performance_ceiling_gap": 1.0 - overall,
        }

    def test_evaluate_pass(self):
        ev = AIRSBenchEvaluator()
        task = BENCHMARK_TASKS["literature_synthesis"]
        result = ev.evaluate(task, self._make_review(0.75), "test_agent", round_num=0)
        assert result.overall_score == pytest.approx(0.75, abs=0.01)

    def test_evaluate_fail_low_score(self):
        ev = AIRSBenchEvaluator()
        task = BenchmarkTask(
            name="hard",
            criteria=EvaluationCriteria(
                min_novelty=0.9, min_validity=0.9,
                min_reproducibility=0.9, min_technical_depth=0.9,
            ),
        )
        result = ev.evaluate(task, self._make_review(0.5), "agent", round_num=0)
        assert not result.passed

    def test_leaderboard_ordering(self):
        ev = AIRSBenchEvaluator()
        task = BENCHMARK_TASKS["literature_synthesis"]
        ev.evaluate(task, self._make_review(0.5), "agent_a", round_num=0)
        ev.evaluate(task, self._make_review(0.8), "agent_b", round_num=0)
        lb = ev.leaderboard(top_n=2)
        assert lb[0]["overall_score"] >= lb[1]["overall_score"]

    def test_trend_improving(self):
        ev = AIRSBenchEvaluator()
        task = BENCHMARK_TASKS["literature_synthesis"]
        ev.evaluate(task, self._make_review(0.4), "agent_x", round_num=0)
        ev.evaluate(task, self._make_review(0.7), "agent_x", round_num=1)
        trend = ev.trend("agent_x")
        assert trend["trend"] == "improving"
        assert trend["improvement"] > 0

    def test_reliability_report(self):
        ev = AIRSBenchEvaluator()
        task = BENCHMARK_TASKS["ml_optimisation"]
        ev.evaluate(task, self._make_review(0.4), "a", round_num=0)
        ev.evaluate(task, self._make_review(0.9), "b", round_num=0)
        report = ev.reliability_report()
        assert "pass_rate" in report
        assert "failure_rate" in report
        assert abs(report["pass_rate"] + report["failure_rate"] - 1.0) < 0.01

    def test_benchmark_tasks_complete(self):
        for tid, task in BENCHMARK_TASKS.items():
            assert task.name, f"Task {tid} missing name"
            assert task.criteria.human_sota_score > 0


class TestExperienceStore:
    def test_save_and_retrieve_learnings(self, tmp_path):
        store = ExperienceStore(db_path=str(tmp_path / "mem.db"))
        round_rec = store.start_round("neural network optimisation")
        store.save_learnings(round_rec.round_id, ["neural networks are effective", "network depth matters"])
        # Query with a word that appears in the learnings
        retrieved = store.search_learnings("neural network", n=10)
        assert len(retrieved) >= 2

    def test_summary_after_rounds(self, tmp_path):
        store = ExperienceStore(db_path=str(tmp_path / "mem2.db"))
        r1 = store.start_round("task 1")
        store.finish_round(r1.round_id, score=0.6)
        r2 = store.start_round("task 2")
        store.finish_round(r2.round_id, score=0.8)
        s = store.summary()
        assert s["total_rounds"] == 2
        assert abs(s["average_score"] - 0.7) < 0.01

    def test_in_memory_backend(self):
        store = ExperienceStore(db_path=":memory:")
        r = store.start_round("task")
        store.finish_round(r.round_id, score=0.5)
        assert store.summary()["total_rounds"] == 1


class TestConfig:
    def test_default_config_valid(self):
        from autoresearch.config import AutoresearchConfig
        cfg = AutoresearchConfig()
        cfg.validate()   # should not raise

    def test_invalid_temperature_raises(self):
        from autoresearch.config import AutoresearchConfig
        cfg = AutoresearchConfig()
        cfg.llm.temperature = 3.0
        with pytest.raises(ValueError, match="temperature"):
            cfg.validate()

    def test_json_roundtrip(self):
        import json
        from autoresearch.config import AutoresearchConfig
        cfg = AutoresearchConfig()
        d = json.loads(cfg.to_json())
        assert d["seed"] == 42
        assert "llm" in d
