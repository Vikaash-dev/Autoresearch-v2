"""
Tests for agents/hyperagent_client.py

Covers:
  - HyperAgentClient.from_config() construction
  - bash fallback pipeline (no hyperagent package required)
  - _plan_task decomposition
  - _navigate file discovery
  - _parse_metric stdout parsing
  - evaluate_hypothesis_on_repo with a tiny synthetic repo
  - HyperAgentResult data class
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from agents.hyperagent_client import (
    HyperAgentClient,
    HyperAgentResult,
    SubtaskResult,
)


# ────────────────────────────────────────────────────────────────────────── #
#  Fixtures                                                                   #
# ────────────────────────────────────────────────────────────────────────── #

@pytest.fixture()
def client() -> HyperAgentClient:
    return HyperAgentClient(mode="bash_fallback")


@pytest.fixture()
def tiny_repo(tmp_path: Path) -> Path:
    """A minimal synthetic repo with an evaluate.py that prints a metric."""
    (tmp_path / "evaluate.py").write_text(
        "print('METRIC: 0.88')\n"
    )
    (tmp_path / "README.md").write_text("# Tiny repo\nA test fixture.\n")
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_basic.py").write_text(
        "def test_ok(): assert True\n"
    )
    return tmp_path


# ────────────────────────────────────────────────────────────────────────── #
#  Construction                                                               #
# ────────────────────────────────────────────────────────────────────────── #

class TestConstruction:
    def test_defaults(self, tmp_path):
        c = HyperAgentClient(clone_dir=str(tmp_path))
        assert c._mode == "auto"
        assert c._language == "python"

    def test_from_config_empty(self, tmp_path):
        c = HyperAgentClient.from_config({})
        assert c._mode == "auto"
        assert c._nav_model == "claude-3-haiku-20240307"

    def test_from_config_overrides(self, tmp_path):
        cfg = {"hyperagent": {"mode": "bash", "language": "java", "exec_timeout": 60}}
        c = HyperAgentClient.from_config(cfg)
        assert c._mode == "bash"
        assert c._language == "java"
        assert c._exec_timeout == 60

    def test_check_available_returns_bool(self):
        # Force a fresh check
        HyperAgentClient._HYPERAGENT_AVAILABLE = None
        result = HyperAgentClient._check_available()
        assert isinstance(result, bool)


# ────────────────────────────────────────────────────────────────────────── #
#  _plan_task                                                                 #
# ────────────────────────────────────────────────────────────────────────── #

class TestPlanTask:
    def test_implement_produces_steps(self):
        steps = HyperAgentClient._plan_task("implement a new loss function")
        assert any("Implement" in s or "implement" in s.lower() for s in steps)

    def test_evaluate_produces_steps(self):
        steps = HyperAgentClient._plan_task("evaluate the model on benchmark")
        assert any("eval" in s.lower() or "metric" in s.lower() for s in steps)

    def test_empty_task_has_fallback(self):
        steps = HyperAgentClient._plan_task("")
        assert len(steps) >= 1

    def test_test_keyword_adds_test_step(self):
        steps = HyperAgentClient._plan_task("run tests on the codebase")
        assert any("test" in s.lower() for s in steps)


# ────────────────────────────────────────────────────────────────────────── #
#  _parse_metric                                                              #
# ────────────────────────────────────────────────────────────────────────── #

class TestParseMetric:
    def test_basic(self):
        assert HyperAgentClient._parse_metric("METRIC: 0.92") == pytest.approx(0.92)

    def test_last_line_wins(self):
        stdout = "METRIC: 0.5\nsome output\nMETRIC: 0.75\n"
        assert HyperAgentClient._parse_metric(stdout) == pytest.approx(0.75)

    def test_no_metric_returns_none(self):
        assert HyperAgentClient._parse_metric("no metric here") is None

    def test_malformed_metric_skipped(self):
        stdout = "METRIC: abc\nMETRIC: 0.66\n"
        assert HyperAgentClient._parse_metric(stdout) == pytest.approx(0.66)

    def test_integer_metric(self):
        assert HyperAgentClient._parse_metric("METRIC: 1") == pytest.approx(1.0)

    def test_empty_string(self):
        assert HyperAgentClient._parse_metric("") is None


# ────────────────────────────────────────────────────────────────────────── #
#  _navigate                                                                  #
# ────────────────────────────────────────────────────────────────────────── #

class TestNavigate:
    def test_returns_python_files(self, client, tiny_repo):
        files = client._navigate(tiny_repo, "evaluate model metric")
        assert any(f.suffix == ".py" for f in files)

    def test_returns_list(self, client, tiny_repo):
        files = client._navigate(tiny_repo, "anything")
        assert isinstance(files, list)


# ────────────────────────────────────────────────────────────────────────── #
#  Bash fallback full pipeline                                                #
# ────────────────────────────────────────────────────────────────────────── #

class TestBashFallback:
    def test_run_task_with_repo_path(self, client, tiny_repo):
        result = client.run_task(
            task="Run evaluation and report metric",
            repo_path=str(tiny_repo),
        )
        assert isinstance(result, HyperAgentResult)
        assert result.mode == "bash_fallback"

    def test_metric_parsed_from_stdout(self, client, tiny_repo):
        result = client.run_task(
            task="Evaluate the model",
            repo_path=str(tiny_repo),
        )
        assert result.metric == pytest.approx(0.88)

    def test_subtask_results_have_four_stages(self, client, tiny_repo):
        result = client.run_task(task="implement and evaluate", repo_path=str(tiny_repo))
        assert len(result.subtask_results) == 4
        steps = [r.step for r in result.subtask_results]
        assert steps == [1, 2, 3, 4]

    def test_evaluate_hypothesis_on_repo(self, client, tiny_repo):
        result = client.evaluate_hypothesis_on_repo(
            hypothesis="A simple baseline should achieve ~0.88 accuracy",
            repo_url=None,       # type: ignore[arg-type]
            repo_path=str(tiny_repo),  # type: ignore[call-arg]
        )
        assert result.metric == pytest.approx(0.88)

    def test_missing_both_url_and_path_raises(self, client):
        with pytest.raises(ValueError, match="repo_url or repo_path"):
            client.run_task(task="anything")

    def test_success_flag_set_on_metric(self, client, tiny_repo):
        result = client.run_task(task="evaluate", repo_path=str(tiny_repo))
        assert result.success is True


# ────────────────────────────────────────────────────────────────────────── #
#  HyperAgentResult data class                                                #
# ────────────────────────────────────────────────────────────────────────── #

class TestHyperAgentResult:
    def test_defaults(self):
        r = HyperAgentResult(task="t", repo_path="/tmp", mode="bash_fallback", success=True)
        assert r.metric is None
        assert r.final_patch == ""
        assert r.subtask_results == []

    def test_subtask_result_dataclass(self):
        sr = SubtaskResult(step=1, description="Planner", status="ok", output="[]")
        assert sr.patch == ""
        assert sr.step == 1
