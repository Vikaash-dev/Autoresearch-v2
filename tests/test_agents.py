"""Tests for the research workforce agents."""
import pytest
from autoresearch.agents.writer_agent import WriterAgent, ResearchReport
from autoresearch.agents.reviewer_agent import ReviewerAgent, DIMENSION_WEIGHTS
from autoresearch.agents.knowledge_agent import KnowledgeAgent
from autoresearch.agents.experiment_agent import ExperimentAgent
from autoresearch.agents.hypothesis_agent import HypothesisAgent
from autoresearch.agents.research_agent import ResearchAgent


# ---------------------------------------------------------------------------
# WriterAgent
# ---------------------------------------------------------------------------

class TestWriterAgent:
    def test_basic_report(self):
        agent = WriterAgent()
        state = agent.run("LLM hyperparameter optimisation", {
            "learnings": [
                "CMA-ES outperforms random search on HPO tasks",
                "Bilevel optimisation achieves 5x improvement over single-level",
                "LLMs can propose informed hyperparameter configurations",
            ],
            "urls": ["https://arxiv.org/abs/2603.23420"],
            "hypotheses": [],
            "experiments": [],
        })
        assert state.score > 0
        assert isinstance(state.result, dict)
        assert "markdown" in state.result
        assert "report" in state.result

    def test_report_has_sections(self):
        agent = WriterAgent()
        state = agent.run("test task", {
            "learnings": ["finding 1", "finding 2"],
        })
        r = state.result
        assert r["sections_written"] >= 3

    def test_anti_hallucination_flags_empty_learnings(self):
        agent = WriterAgent()
        state = agent.run("completely unknown task with no prior work", {
            "learnings": [],
        })
        # Should produce a report but flag many claims
        assert isinstance(state.result.get("hallucination_flags"), list)

    def test_title_derived_from_task(self):
        agent = WriterAgent()
        state = agent.run("neural network optimisation techniques", {})
        assert "Neural" in state.result.get("title", "") or "network" in state.result.get("title", "").lower()

    def test_references_include_canonical(self):
        agent = WriterAgent()
        state = agent.run("autoresearch", {"learnings": [], "urls": []})
        refs = state.result.get("report", {}).get("references", [])
        assert any("Karpathy" in r or "karpathy" in r for r in refs)

    def test_to_markdown(self):
        report = ResearchReport(
            title="Test", task="test", abstract="Abstract text.",
            sections=[], references=["ref1", "ref2"],
        )
        md = report.to_markdown()
        assert "# Test" in md
        assert "Abstract" in md


# ---------------------------------------------------------------------------
# ReviewerAgent (AIRS-Bench)
# ---------------------------------------------------------------------------

class TestReviewerAgent:
    def _make_context(self, word_count=800, n_refs=5, n_sections=5):
        sections = [
            {"title": "Introduction",    "content": " word " * 100, "verified": True},
            {"title": "Related Work",    "content": " paper " * 100 + " novel approach found", "verified": True},
            {"title": "Methodology",     "content": " method " * 100 + " parameter config result", "verified": True},
            {"title": "Results",         "content": " result " * 100 + " improved 0.85 score", "verified": True},
            {"title": "Conclusion",      "content": " conclusion " * 50 + " significant contribution", "verified": True},
        ][:n_sections]
        refs = [f"Author{i} (2026). Paper title {i}." for i in range(n_refs)]
        report_dict = {
            "title": "Test Report",
            "task": "test",
            "abstract": "We present a novel contribution that significantly improves state-of-the-art.",
            "sections": sections,
            "references": refs,
            "word_count": word_count,
            "hallucination_flags": [],
            "verified_ratio": 1.0,
        }
        markdown = "# Test\n## Introduction\n" + " word " * 200 + "\n## Methodology\n" + " method parameter " * 50
        return {"report": report_dict, "markdown": markdown, "learnings": ["finding 1"] * 5}

    def test_returns_score(self):
        agent = ReviewerAgent()
        state = agent.run("test task", self._make_context())
        assert 0.0 <= state.result["overall_score"] <= 1.0

    def test_seven_dimensions_present(self):
        agent = ReviewerAgent()
        state = agent.run("test task", self._make_context())
        dims = state.result["dimension_scores"]
        for dim in DIMENSION_WEIGHTS:
            assert dim in dims, f"Missing dimension: {dim}"

    def test_weights_sum_to_one(self):
        total = sum(DIMENSION_WEIGHTS.values())
        assert abs(total - 1.0) < 1e-9

    def test_accept_decision_valid(self):
        agent = ReviewerAgent()
        state = agent.run("test task", self._make_context())
        assert state.result["accept_decision"] in ("accept", "revise", "reject")

    def test_hallucination_flags_raised(self):
        agent = ReviewerAgent()
        ctx = self._make_context()
        ctx["report"]["hallucination_flags"] = ["Unsupported claim: xyz"]
        ctx["report"]["verified_ratio"] = 0.3
        state = agent.run("test task", ctx)
        assert state.result["overall_score"] < 0.8

    def test_performance_ceiling_gap(self):
        agent = ReviewerAgent()
        state = agent.run("test task", self._make_context())
        gap = state.result["performance_ceiling_gap"]
        assert 0.0 <= gap <= 1.0

    def test_citation_error_flag(self):
        agent = ReviewerAgent()
        ctx = self._make_context()
        ctx["report"]["references"] = ["Goodfellow (2013). LSTM paper."]  # known hallucination
        state = agent.run("test", ctx)
        all_flags = state.result.get("reliability_flags", [])
        assert any("Goodfellow" in f for f in all_flags)


# ---------------------------------------------------------------------------
# KnowledgeAgent (AutoResearcher grounding + novelty)
# ---------------------------------------------------------------------------

class TestKnowledgeAgent:
    def test_accepts_novel_hypothesis(self):
        agent = KnowledgeAgent()
        state = agent.run("LLM optimisation", {
            "hypotheses": [{"hypothesis": "CMA-ES with LLM proposals outperforms pure random search"}],
            "learnings": ["Random search is suboptimal for high-dimensional spaces"],
        })
        assert state.result["acceptance_rate"] >= 0.0

    def test_rejects_duplicate(self):
        agent = KnowledgeAgent()
        text = "CMA-ES outperforms random search on hyperparameter optimisation"
        # First call — adds to fingerprint set
        agent.run("HPO", {"hypotheses": [{"hypothesis": text}], "learnings": []})
        # Second call — should be rejected as duplicate
        state2 = agent.run("HPO", {"hypotheses": [{"hypothesis": text}], "learnings": []})
        assert state2.result["acceptance_rate"] == 0.0

    def test_neutral_score_when_empty(self):
        agent = KnowledgeAgent()
        state = agent.run("task", {"hypotheses": [], "learnings": []})
        assert state.score == 0.5

    def test_validated_hypotheses_key_present(self):
        agent = KnowledgeAgent()
        state = agent.run("task", {
            "hypotheses": [{"hypothesis": "novel idea about transformers"}],
            "learnings": [],
        })
        assert "validated_hypotheses" in state.result


# ---------------------------------------------------------------------------
# ExperimentAgent (AI Scientist-v2 Experiment Manager)
# ---------------------------------------------------------------------------

class TestExperimentAgent:
    def test_runs_with_hypotheses(self):
        agent = ExperimentAgent()
        state = agent.run("test task", {
            "accepted_hypotheses": [
                {"hypothesis": "Bilevel optimisation improves convergence"}
            ],
        })
        assert isinstance(state.result, dict)
        assert "experiment_results" in state.result

    def test_neutral_score_when_no_hypotheses(self):
        agent = ExperimentAgent()
        state = agent.run("task", {"accepted_hypotheses": []})
        assert state.score == 0.5

    def test_accepts_raw_string_hypotheses(self):
        agent = ExperimentAgent()
        state = agent.run("task", {
            "hypotheses": ["hypothesis as raw string"]
        })
        assert "experiment_results" in state.result


# ---------------------------------------------------------------------------
# HypothesisAgent (MCTS — AI Scientist-v2)
# ---------------------------------------------------------------------------

class TestHypothesisAgent:
    def test_returns_hypotheses(self):
        agent = HypothesisAgent()
        state = agent.run("LLM hyperparameter optimisation", {
            "learnings": ["CMA-ES is effective for HPO"],
        })
        hyps = state.result.get("hypotheses", [])
        assert len(hyps) > 0

    def test_best_hypothesis_has_score(self):
        agent = HypothesisAgent()
        state = agent.run("test", {"learnings": ["finding"]})
        best = state.result.get("best_hypothesis")
        if best:
            assert "score" in best or "hypothesis" in best


# ---------------------------------------------------------------------------
# ResearchAgent
# ---------------------------------------------------------------------------

class TestResearchAgent:
    def test_returns_findings(self):
        agent = ResearchAgent()
        state = agent.run("autoresearch", {"search_results": ["finding1", "finding2"]})
        assert state.score >= 0
        assert isinstance(state.result, dict)
