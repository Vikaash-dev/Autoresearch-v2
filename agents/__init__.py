"""Research agents — all agent classes for Autoresearch-v2."""

from agents.base_agent import BaseAgent
from agents.hypothesis_agent import HypothesisAgent
from agents.experiment_agent import ExperimentAgent
from agents.literature_agent import LiteratureAgent
from agents.writer_agent import WriterAgent
from agents.reviewer_agent import ReviewerAgent
from agents.orchestrator import Orchestrator
from agents.hyperagent_client import HyperAgentClient, HyperAgentResult, SubtaskResult

__all__ = [
    "BaseAgent",
    "HypothesisAgent",
    "ExperimentAgent",
    "LiteratureAgent",
    "WriterAgent",
    "ReviewerAgent",
    "Orchestrator",
    "HyperAgentClient",
    "HyperAgentResult",
    "SubtaskResult",
]
