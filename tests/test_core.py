"""Tests for core module."""
import pytest
from autoresearch.config.schema import AutoResearchConfig
from autoresearch.core.objective_graph import ObjectiveGraph, Objective, ObjectiveStatus
from autoresearch.core.state import RunState


def test_objective_graph_ready():
    graph = ObjectiveGraph()
    obj_a = Objective(id="a", name="A")
    obj_b = Objective(id="b", name="B", dependencies=["a"])
    graph.add(obj_a, lambda *a: None)
    graph.add(obj_b, lambda *a: None)

    ready = graph.get_ready()
    assert len(ready) == 1
    assert ready[0].id == "a"


def test_objective_graph_completion():
    graph = ObjectiveGraph()
    obj_a = Objective(id="a", name="A")
    obj_b = Objective(id="b", name="B", dependencies=["a"])
    graph.add(obj_a, lambda *a: None)
    graph.add(obj_b, lambda *a: None)

    graph.mark_completed("a")
    ready = graph.get_ready()
    assert ready[0].id == "b"


def test_objective_graph_is_done():
    graph = ObjectiveGraph()
    obj = Objective(id="a", name="A")
    graph.add(obj, lambda *a: None)
    assert not graph.is_done()
    graph.mark_completed("a")
    assert graph.is_done()


def test_run_state_save_load(tmp_path):
    state = RunState(run_id="test123", topic="AI testing")
    state.evidence.append({"id": "e1", "title": "Test paper"})
    saved = state.save(tmp_path)
    loaded = RunState.load(tmp_path)
    assert loaded.run_id == "test123"
    assert loaded.topic == "AI testing"
    assert len(loaded.evidence) == 1
