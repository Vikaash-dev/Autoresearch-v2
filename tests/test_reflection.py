"""Tests for reflection/self-improvement module."""
import pytest
from autoresearch.config.schema import AutoResearchConfig
from autoresearch.core.state import RunState
from autoresearch.reflection.self_review import SelfReviewModule
from autoresearch.reflection.improvement import SelfImprovementEngine


def test_self_review_empty_state():
    cfg = AutoResearchConfig()
    reviewer = SelfReviewModule(cfg)
    state = RunState(run_id="test", topic="AI")
    reflection = reviewer.analyze(state)
    assert "bottlenecks" in reflection
    assert "quality_scores" in reflection
    assert "proposed_policy_updates" in reflection


def test_self_review_quality_scores():
    cfg = AutoResearchConfig()
    reviewer = SelfReviewModule(cfg)
    state = RunState(run_id="test", topic="AI")
    state.evidence = [{"id": str(i)} for i in range(10)]
    state.hypotheses = [{"id": f"h{i}"} for i in range(3)]
    reflection = reviewer.analyze(state)
    assert reflection["quality_scores"]["evidence"] == 1.0


def test_improvement_engine_safe():
    cfg = AutoResearchConfig()
    engine = SelfImprovementEngine(cfg)
    state = RunState(run_id="test", topic="AI")
    updates = [
        {"module": "literature_miner", "parameter": "arxiv_max_results", "current": 20, "proposed": 40, "safe": True, "rationale": "more results"},
        {"module": "executor", "parameter": "unsafe_param", "current": 1, "proposed": 99, "safe": False},
    ]
    applied = engine.apply_updates(updates, state)
    assert len(applied) == 1
    assert applied[0]["parameter"] == "arxiv_max_results"


def test_improvement_engine_validate():
    cfg = AutoResearchConfig()
    engine = SelfImprovementEngine(cfg)
    assert engine.validate_update({"safe": True, "parameter": "max_retries", "proposed": 5})
    assert not engine.validate_update({"safe": False, "parameter": "max_retries", "proposed": 5})
    assert not engine.validate_update({"safe": True, "parameter": "secret_param", "proposed": "hack"})
