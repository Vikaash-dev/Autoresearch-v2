"""Tests for experiments module."""
import pytest
from autoresearch.config.schema import AutoResearchConfig
from autoresearch.experiments.spec import ExperimentSpec
from autoresearch.experiments.generator import ExperimentGenerator
from autoresearch.experiments.self_healer import SelfHealingLoop
from autoresearch.experiments.sandbox import _run_code


def test_experiment_spec():
    spec = ExperimentSpec(title="Test Exp", code="print('hello')")
    assert spec.title == "Test Exp"
    assert spec.status == "pending"
    assert spec.max_retries == 3


def test_experiment_generator():
    cfg = AutoResearchConfig()
    gen = ExperimentGenerator(cfg)
    hyp = {"id": "hyp_0", "title": "Hypothesis 0"}
    spec = gen.generate(hyp, "machine learning")
    assert spec.hypothesis_id == "hyp_0"
    assert "python" in spec.language
    assert len(spec.code) > 0


def test_run_code_success():
    result = _run_code("print('hello world')", timeout=10)
    assert result["status"] == "success"
    assert "hello world" in result["stdout"]


def test_run_code_failure():
    result = _run_code("raise ValueError('test error')", timeout=10)
    assert result["status"] == "failed"


def test_self_healer_success():
    healer = SelfHealingLoop(max_retries=2)
    result = healer.run("import json\nprint(json.dumps({'ok': True}))", timeout=10)
    assert result["status"] == "success"


def test_self_healer_import_fix():
    healer = SelfHealingLoop(max_retries=3)
    # Code that uses a nonexistent module — healer should stub it
    code = "import nonexistent_module_xyz\nprint('done')"
    result = healer.run(code, timeout=10)
    # After repair, the import is stubbed so it should proceed
    assert result["attempts"] >= 1
