from __future__ import annotations

from typing import Any

from autoresearch_v2.agents.base_agent import BaseAgent
from autoresearch_v2.dog.objective import Objective
from autoresearch_v2.retrieval.semantic import LocalSemanticRetriever, RetrievedPaper


class LiteratureAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__(agent_id="literature_agent")

    def _run(self, objective: Objective, context: dict[str, Any]) -> dict[str, Any]:
        topic = context.get("topic", objective.description)
        query = context.get("literature_query", topic)
        retriever = context.get("retriever")
        if retriever is None:
            retriever = LocalSemanticRetriever()
        top_k = int(context.get("top_k", 10))
        retrieved: list[RetrievedPaper] = retriever.search(str(query), top_k=max(1, top_k))
        citations = [f"{r.doc_id}::{r.title}" for r in retrieved]
        evidence = [
            {
                "evidence_id": r.evidence_id,
                "doc_id": r.doc_id,
                "title": r.title,
                "score": r.score,
                "source": r.source,
                "year": r.year,
                "url": r.url,
                "chunk": r.chunk,
            }
            for r in retrieved
        ]
        avg_score = round(sum(r.score for r in retrieved) / len(retrieved), 6) if retrieved else 0.0
        summary = (
            f"Retrieved {len(retrieved)} semantically ranked papers for '{topic}' "
            f"(avg_score={avg_score}) using local offline corpus."
        )
        return {
            "topic": topic,
            "query": str(query),
            "citations": citations,
            "evidence": evidence,
            "summary": summary,
        }
