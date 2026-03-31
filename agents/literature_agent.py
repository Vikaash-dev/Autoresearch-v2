"""
Literature Agent — searches arXiv, Semantic Scholar, and GitHub for relevant work.

Feeds into:
  - Community-ToM (what the field collectively knows)
  - HypothesisAgent (avoid reinventing, build on existing work)
  - WriterAgent (citations)

Based on:
  - AI-Scientist v2 Semantic Scholar integration
  - NousResearch/autonovel literature review loop
  - karpathy/autoresearch "reference nanochat parent repository" pattern
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any
from urllib.error import URLError
from urllib.parse import urlencode, quote
from urllib.request import urlopen

from agents.base_agent import BaseAgent
from core.blackboard import Blackboard
from tom.engine import TheoryOfMindEngine

logger = logging.getLogger(__name__)

_SEMANTIC_SCHOLAR_API = "https://api.semanticscholar.org/graph/v1/paper/search"
_ARXIV_API = "https://export.arxiv.org/api/query"


class LiteratureAgent(BaseAgent):
    """Searches for relevant papers and GitHub repos."""

    def __init__(
        self,
        blackboard: Blackboard,
        tom_engine: TheoryOfMindEngine,
        llm_fn: Any = None,
        s2_api_key: str | None = None,
    ) -> None:
        super().__init__("literature", blackboard, tom_engine, llm_fn)
        self._s2_key = s2_api_key

    def act(self, context: dict[str, Any]) -> dict[str, Any]:
        """
        Search for papers related to the query.

        Returns:
            {abstracts: [...], titles: [...], urls: [...]}
        """
        query = context.get("query", "")
        max_results = context.get("max_results", 10)

        results: dict[str, list] = {"abstracts": [], "titles": [], "urls": []}

        if not query:
            return results

        # Try Semantic Scholar first, fall back to arXiv
        s2_results = self._search_semantic_scholar(query, max_results)
        if s2_results:
            results.update(s2_results)
        else:
            arxiv_results = self._search_arxiv(query, max_results)
            if arxiv_results:
                results.update(arxiv_results)

        self.record_success("search", float(len(results["titles"])))
        self.update_own_belief(
            last_action=f"searched:{query[:40]}",
            current_uncertainty=0.4,
        )
        return results

    def summarize_for_hypothesis(self, papers: list[dict[str, Any]], topic: str) -> str:
        """
        Summarize retrieved papers as context for the HypothesisAgent.
        Identifies gaps and opportunities the agent should target.
        """
        if not papers or self._llm is None:
            return ""

        abstracts = "\n".join(
            f"[{i+1}] {p.get('title','?')}: {p.get('abstract','')[:200]}"
            for i, p in enumerate(papers[:10])
        )
        prompt = (
            f"Topic: {topic}\n\nRecent papers:\n{abstracts}\n\n"
            "Identify:\n"
            "1. Key open problems not yet solved\n"
            "2. Overcrowded directions to avoid\n"
            "3. Promising under-explored directions\n"
            "4. Contradictions or active debates in the literature\n"
            "Be specific and concise."
        )
        try:
            return self.call_llm(prompt)
        except Exception as exc:
            logger.warning("Literature summarization failed: %s", exc)
            return ""

    # ------------------------------------------------------------------ #
    #  Search backends                                                      #
    # ------------------------------------------------------------------ #

    def _search_semantic_scholar(
        self, query: str, max_results: int
    ) -> dict[str, list] | None:
        params = {
            "query": query,
            "limit": min(max_results, 100),
            "fields": "title,abstract,year,externalIds",
        }
        url = f"{_SEMANTIC_SCHOLAR_API}?{urlencode(params)}"
        headers: dict[str, str] = {}
        if self._s2_key:
            headers["x-api-key"] = self._s2_key

        try:
            import urllib.request
            req = urllib.request.Request(url, headers=headers)
            with urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode())
        except Exception as exc:
            logger.debug("Semantic Scholar search failed: %s", exc)
            return None

        papers = data.get("data", [])
        return {
            "abstracts": [p.get("abstract") or "" for p in papers],
            "titles": [p.get("title") or "" for p in papers],
            "urls": [
                f"https://www.semanticscholar.org/paper/{p.get('paperId','')}"
                for p in papers
            ],
        }

    def _search_arxiv(
        self, query: str, max_results: int
    ) -> dict[str, list] | None:
        params = {
            "search_query": f"all:{quote(query)}",
            "start": 0,
            "max_results": min(max_results, 50),
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }
        url = f"{_ARXIV_API}?{urlencode(params)}"
        try:
            with urlopen(url, timeout=15) as resp:
                content = resp.read().decode()
        except Exception as exc:
            logger.debug("arXiv search failed: %s", exc)
            return None

        # Simple XML parse (no lxml dependency)
        titles = self._extract_xml_tags(content, "title")
        abstracts = self._extract_xml_tags(content, "summary")
        ids = self._extract_xml_tags(content, "id")

        # Remove the feed title/summary
        titles = [t for t in titles if "arXiv" not in t][:max_results]
        abstracts = abstracts[:max_results]
        urls = [i for i in ids if "arxiv.org/abs" in i][:max_results]

        return {"abstracts": abstracts, "titles": titles, "urls": urls}

    @staticmethod
    def _extract_xml_tags(xml: str, tag: str) -> list[str]:
        import re
        return re.findall(rf"<{tag}[^>]*>(.*?)</{tag}>", xml, re.DOTALL)
