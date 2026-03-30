"""Tests for the Bilevel Autoresearch optimization engine (arXiv:2603.23420)."""
import math
import pytest
from autoresearch.optimization.bilevel import (
    BilevelEngine, BilevelConfig, TabuSearchManager,
    MechanismRegistry, SearchStrategy, _next_strategy, ExperimentRecord,
)


# ---------------------------------------------------------------------------
# TabuSearchManager
# ---------------------------------------------------------------------------

class TestTabuSearchManager:
    def test_not_tabu_empty(self):
        tsm = TabuSearchManager()
        assert not tsm.is_tabu({"lr": 0.01})

    def test_identical_is_tabu(self):
        tsm = TabuSearchManager(similarity_threshold=0.8)
        tsm.add({"lr": 0.01, "bs": 32})
        assert tsm.is_tabu({"lr": 0.01, "bs": 32})
        assert tsm.blocks == 1

    def test_different_not_tabu(self):
        tsm = TabuSearchManager()
        tsm.add({"lr": 0.01})
        assert not tsm.is_tabu({"lr": 0.1, "wd": 0.001})

    def test_max_tabu_size_evicts_oldest(self):
        tsm = TabuSearchManager(max_tabu_size=3)
        for i in range(4):
            tsm.add({f"key_{i}": i})
        assert len(tsm._tabu_list) == 3

    def test_similarity_same_dict(self):
        assert TabuSearchManager._similarity({"a": 1}, {"a": 1}) == 1.0

    def test_similarity_no_overlap(self):
        assert TabuSearchManager._similarity({"a": 1}, {"b": 2}) == 0.0


# ---------------------------------------------------------------------------
# MechanismRegistry
# ---------------------------------------------------------------------------

class TestMechanismRegistry:
    def test_builtins_loaded(self):
        reg = MechanismRegistry()
        mechs = reg.list_mechanisms()
        assert "tabu_search" in mechs
        assert "orthogonal_exploration" in mechs

    def test_apply_all_returns_dict(self):
        reg = MechanismRegistry()
        result = reg.apply_all({"lr": 0.01}, [])
        assert isinstance(result, dict)

    def test_generate_and_load_custom(self):
        reg = MechanismRegistry()
        ok = reg.generate_and_load("my_mechanism", "a simple passthrough mechanism")
        assert ok
        assert "my_mechanism" in reg.list_mechanisms()

    def test_summaries(self):
        reg = MechanismRegistry()
        s = reg.summaries()
        assert isinstance(s, dict)
        assert "tabu_search" in s


# ---------------------------------------------------------------------------
# Adaptive strategy transitions (CORAL)
# ---------------------------------------------------------------------------

class TestAdaptiveStrategy:
    def _make_records(self, n_kept, n_total, n_crashed=0):
        recs = []
        for i in range(n_total):
            kept = i < n_kept
            status = "crashed" if i < n_crashed else "completed"
            recs.append(ExperimentRecord(
                round_num=i, changes={f"k{i}": i},
                score=0.8 if kept else 0.3,
                improvement=0.1 if kept else -0.1,
                kept=kept, status=status,
            ))
        return recs

    def test_explore_when_few_records(self):
        assert _next_strategy([], SearchStrategy.EXPLORE) == SearchStrategy.EXPLORE

    def test_exploit_high_keep_rate(self):
        # High keep rate AND recent keeps (last 5 have kept=True) → EXPLOIT
        recs = []
        for i in range(10):
            kept = True  # all kept = high keep rate with recent success
            recs.append(ExperimentRecord(
                round_num=i, changes={f"k{i}": i},
                score=0.8, improvement=0.1,
                kept=kept, status="completed",
            ))
        result = _next_strategy(recs, SearchStrategy.EXPLORE)
        assert result == SearchStrategy.EXPLOIT

    def test_exploit_on_crash(self):
        recs = self._make_records(n_kept=0, n_total=10, n_crashed=7)  # 70% crash
        result = _next_strategy(recs, SearchStrategy.EXPLORE)
        assert result == SearchStrategy.EXPLOIT

    def test_ablation_on_plateau_no_nearmiss(self):
        recs = self._make_records(n_kept=0, n_total=10)
        # Make improvements very negative (not near misses)
        for r in recs:
            r.improvement = -0.5
        result = _next_strategy(recs, SearchStrategy.EXPLORE)
        assert result == SearchStrategy.ABLATION


# ---------------------------------------------------------------------------
# BilevelEngine full loop
# ---------------------------------------------------------------------------

class TestBilevelEngine:
    def _make_engine(self, budget=5.0, max_iter=10):
        scores = [0.5, 0.52, 0.49, 0.55, 0.53, 0.58, 0.56, 0.60, 0.59, 0.62]
        call_count = [0]

        def executor(changes):
            idx = min(call_count[0], len(scores) - 1)
            call_count[0] += 1
            return scores[idx], "completed"

        def proposer(history, strategy, config):
            import random
            return {"lr": random.uniform(1e-4, 1e-1), "bs": random.choice([16, 32, 64])}

        cfg = BilevelConfig(
            inner_max_iterations=max_iter,
            inner_budget_seconds=budget,
            mechanism_generation_every=5,
            plateau_threshold=3,
            verbose=False,
        )
        return BilevelEngine(executor=executor, proposer=proposer, config=cfg)

    def test_run_returns_dict(self):
        engine = self._make_engine()
        result = engine.run("test task", baseline_score=0.48)
        assert isinstance(result, dict)
        assert "best_score" in result
        assert "history" in result

    def test_best_score_improves(self):
        engine = self._make_engine(max_iter=10)
        result = engine.run("optimise", baseline_score=0.48)
        assert result["best_score"] >= 0.48

    def test_mechanisms_loaded(self):
        engine = self._make_engine(max_iter=8, budget=30.0)
        result = engine.run("test", baseline_score=0.0)
        assert len(result["mechanisms_used"]) >= 1

    def test_tabu_summary_present(self):
        engine = self._make_engine()
        result = engine.run("test", baseline_score=0.0)
        assert "tabu_summary" in result

    def test_keep_rate_reasonable(self):
        engine = self._make_engine(max_iter=10, budget=30.0)
        result = engine.run("test", baseline_score=0.48)
        assert 0.0 <= result["keep_rate"] <= 1.0
