"""Tests for LLM client integration and fallback behavior."""
from __future__ import annotations

from unittest.mock import patch

from autoresearch.config.schema import AutoResearchConfig
from autoresearch.llm.client import LLMClient
from autoresearch.agents.base import BaseAgent
from autoresearch.core.objective_graph import Objective
from autoresearch.core.state import RunState


class _DummyAgent(BaseAgent):
    name = "dummy"
    role = "dummy"

    def run(self, objective, state, config):
        return self._llm_prompt("hello world")


def test_llm_client_fallback_without_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    cfg = AutoResearchConfig()
    client = LLMClient(cfg.llm)
    out = client.prompt("test prompt")
    assert out.startswith("[LLM response to:")


def test_llm_client_openai_success(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    cfg = AutoResearchConfig()
    client = LLMClient(cfg.llm)

    class _Resp:
        def raise_for_status(self):
            return None

        def json(self):
            return {"choices": [{"message": {"content": "real answer"}}]}

    with patch("httpx.Client.post", return_value=_Resp()):
        out = client.prompt("test prompt")

    assert out == "real answer"


def test_base_agent_uses_llm_client(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    cfg = AutoResearchConfig(topic="x")
    agent = _DummyAgent(cfg)
    result = agent.run(Objective(id="o1", name="obj"), RunState(run_id="r1", topic="x"), cfg)
    assert result.startswith("[LLM response to:")
