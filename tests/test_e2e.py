"""End-to-end smoke test for AutoResearch v2 pipeline."""
import pytest
from autoresearch.config.schema import AutoResearchConfig
from autoresearch.core.runtime import AgentRuntime
from autoresearch.core.objective_graph import Objective
from autoresearch.agents.planner import ResearchPlannerAgent
from autoresearch.agents.literature_miner import LiteratureMinerAgent
from autoresearch.agents.hypothesis_generator import HypothesisGeneratorAgent
from autoresearch.agents.experiment_coder import ExperimentCoderAgent
from autoresearch.agents.executor import ExecutorAgent
from autoresearch.agents.reviewer import ReviewerAgent
from autoresearch.agents.meta_agent import MetaAgent
from autoresearch.agents.tom_agent import TomAgent


def test_e2e_smoke(tmp_path, monkeypatch):
    """Smoke test: full pipeline runs without crashing."""
    # Patch arXiv search to return offline results (no network needed)
    import autoresearch.literature.arxiv_connector as ax
    monkeypatch.setattr(ax.ArxivConnector, "search", lambda self, q: [
        {"source": "arxiv", "arxiv_id": "2401.0000" + str(i), "title": f"Test Paper {i}",
         "abstract": "Multi-agent autonomous research systems using transformers.",
         "published": "2024-01-01", "authors": ["Author A"], "url": f"https://arxiv.org/abs/2401.0000{i}"}
        for i in range(5)
    ])

    cfg = AutoResearchConfig(
        topic="multi-agent autonomous research",
        max_iterations=1,
        run_id="smoke_test",
    )
    cfg.checkpoint.checkpoint_dir = str(tmp_path / "checkpoints")
    cfg.experiment.timeout_seconds = 15
    cfg.tom.adversarial_rounds = 1

    agents = {
        "planner": ResearchPlannerAgent(cfg),
        "literature_miner": LiteratureMinerAgent(cfg),
        "hypothesis_generator": HypothesisGeneratorAgent(cfg),
        "experiment_coder": ExperimentCoderAgent(cfg),
        "executor": ExecutorAgent(cfg),
        "reviewer": ReviewerAgent(cfg),
        "meta_agent": MetaAgent(cfg),
        "tom_agent": TomAgent(cfg),
    }

    runtime = AgentRuntime(cfg, agents)

    runtime.add_objective(Objective(id="plan", name="Planning", priority=10), "planner")
    runtime.add_objective(Objective(id="mine", name="Literature Mining", dependencies=["plan"], priority=9), "literature_miner")
    runtime.add_objective(Objective(id="hypothesize", name="Hypothesis Generation", dependencies=["mine"], priority=8), "hypothesis_generator")
    runtime.add_objective(Objective(id="tom", name="ToM", dependencies=["hypothesize"], priority=7), "tom_agent")
    runtime.add_objective(Objective(id="code", name="Coding", dependencies=["tom"], priority=6), "experiment_coder")
    runtime.add_objective(Objective(id="execute", name="Execute", dependencies=["code"], priority=5), "executor")
    runtime.add_objective(Objective(id="review", name="Review", dependencies=["execute"], priority=4), "reviewer")
    runtime.add_objective(Objective(id="reflect", name="Reflect", dependencies=["review"], priority=3), "meta_agent")

    state = runtime.run()

    assert state.status == "completed"
    assert len(state.objectives_completed) == 8
    assert len(state.evidence) >= 5
    assert len(state.hypotheses) >= 1
    assert len(state.experiments) >= 1
    assert len(state.reviews) >= 1
    assert len(state.reflections) >= 1
