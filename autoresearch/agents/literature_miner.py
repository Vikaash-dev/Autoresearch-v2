"""Literature Miner Agent."""
from __future__ import annotations
from typing import Any, TYPE_CHECKING
from .base import BaseAgent

if TYPE_CHECKING:
    from ..core.objective_graph import Objective
    from ..core.state import RunState
    from ..config.schema import AutoResearchConfig


class LiteratureMinerAgent(BaseAgent):
    name = "literature_miner"
    role = "Literature Miner"

    def run(self, objective: "Objective", state: "RunState", config: "AutoResearchConfig") -> Any:
        from ..literature.arxiv_connector import ArxivConnector
        from ..literature.evidence_registry import EvidenceRegistry

        self.logger.info("Mining literature for: %s", state.topic)
        connector = ArxivConnector(max_results=config.literature.arxiv_max_results)
        papers = connector.search(state.topic)

        registry = EvidenceRegistry()
        evidence_items = []
        for paper in papers:
            item = registry.add_paper(paper)
            evidence_items.append(item)

        state.evidence.extend([e.model_dump() for e in evidence_items])
        self.logger.info("Found %d papers", len(papers))
        return {"papers_found": len(papers), "evidence_ids": [e.id for e in evidence_items]}
