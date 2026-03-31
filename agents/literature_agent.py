"""
Literature Agent — searches for research papers and GitHub repos.

Search priority (waterfall — each tier is tried only if the previous fails):
  1. Tavily (primary)   — real-time web search with domain filters;
                          multi-key pool with automatic rotation/fallback
  2. Semantic Scholar   — structured paper metadata + abstracts
  3. arXiv API          — fallback for very recent preprints

Tavily is used for BOTH paper search (arxiv/s2/openreview domains) and
GitHub repository search (github.com domain).

Feeds into:
  - Community-ToM (what the field collectively knows)
  - HypothesisAgent (avoid reinventing, build on existing work)
  - WriterAgent (citations)
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any
from urllib.parse import urlencode, quote
from urllib.request import urlopen

from agents.base_agent import BaseAgent
from core.blackboard import Blackboard
from tom.engine import TheoryOfMindEngine
from tools.tavily_search import TavilyPool, TavilyPoolExhausted

logger = logging.getLogger(__name__)

_SEMANTIC_SCHOLAR_API = "https://api.semanticscholar.org/graph/v1/paper/search"
_ARXIV_API = "https://export.arxiv.org/api/query"


class LiteratureAgent(BaseAgent):
    """
    Searches for relevant papers and GitHub repos.

    Backend waterfall:
      Tavily (multi-key pool) → Semantic Scholar → arXiv
    """

    def __init__(
        self,
        blackboard: Blackboard,
        tom_engine: TheoryOfMindEngine,
        llm_fn: Any = None,
        tavily_keys: list[str] | None = None,
        s2_api_key: str | None = None,
    ) -> None:
        super().__init__("literature", blackboard, tom_engine, llm_fn)
        self._s2_key = s2_api_key
        self._tavily: TavilyPool | None = self._init_tavily(tavily_keys)

    # ------------------------------------------------------------------ #
    #  Public interface                                                     #
    # ------------------------------------------------------------------ #

    def act(self, context: dict[str, Any]) -> dict[str, Any]:
        """
        Search for papers and GitHub repos related to the query.

        Context keys:
            query (str)         — search query
            max_results (int)   — max papers to return  [default: 10]
            include_github (bool) — also search GitHub repos [default: True]
            year_from (int)     — filter to papers from this year onwards

        Returns:
            {
              abstracts: [...],
              titles:    [...],
              urls:      [...],
              github_repos: [{name, url, description}, ...],  # if include_github
              source: "tavily" | "semantic_scholar" | "arxiv" | "none",
            }
        """
        query = context.get("query", "")
        max_results = int(context.get("max_results", 10))
        include_github = context.get("include_github", True)
        year_from = context.get("year_from", None)

        results: dict[str, Any] = {
            "abstracts": [],
            "titles": [],
            "urls": [],
            "github_repos": [],
            "source": "none",
        }

        if not query:
            return results

        # ── Paper search ─────────────────────────────────────────────── #
        paper_results = (
            self._search_papers_tavily(query, max_results, year_from)
            or self._search_semantic_scholar(query, max_results)
            or self._search_arxiv(query, max_results)
        )
        if paper_results:
            results.update(paper_results)

        # ── GitHub repo search ────────────────────────────────────────── #
        if include_github and self._tavily is not None:
            gh_results = self._search_github_tavily(query, max_results=5)
            if gh_results:
                results["github_repos"] = gh_results

        count = len(results["titles"])
        self.record_success("search", float(count))
        self.update_own_belief(
            last_action=f"searched:{query[:40]} ({count} results, src={results['source']})",
            current_uncertainty=0.4 if count > 0 else 0.8,
        )
        logger.info(
            "LiteratureAgent: %d papers via %s, %d GitHub repos",
            count, results["source"], len(results["github_repos"]),
        )
        return results

    def search_github(self, query: str, max_results: int = 10) -> list[dict[str, Any]]:
        """
        Standalone GitHub repo search (called directly by Orchestrator).

        Returns list of {name, url, description, content_snippet}.
        """
        if self._tavily is None:
            logger.debug("Tavily not available — skipping GitHub search")
            return []
        return self._search_github_tavily(query, max_results) or []

    def extract_paper_content(self, urls: list[str]) -> list[dict[str, Any]]:
        """
        Extract full text from paper URLs (arXiv abstract pages, PDFs, etc.)
        using Tavily Extract.
        """
        if not urls or self._tavily is None:
            return []
        try:
            response = self._tavily.extract(urls)
            return response.get("results", [])
        except TavilyPoolExhausted:
            logger.warning("Tavily pool exhausted during extract")
            return []
        except Exception as exc:
            logger.warning("Tavily extract failed: %s", exc)
            return []

    def summarize_for_hypothesis(self, papers: list[dict[str, Any]], topic: str) -> str:
        """
        Summarize retrieved papers as context for the HypothesisAgent.
        Identifies gaps, overcrowded areas, and promising directions.
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
    #  Tavily backends (primary)                                            #
    # ------------------------------------------------------------------ #

    def _search_papers_tavily(
        self, query: str, max_results: int, year_from: int | None
    ) -> dict[str, Any] | None:
        if self._tavily is None:
            return None
        try:
            response = self._tavily.search_papers(
                query, max_results=max_results, year_from=year_from
            )
        except TavilyPoolExhausted as exc:
            logger.warning("Tavily pool exhausted for paper search: %s", exc)
            return None
        except Exception as exc:
            logger.debug("Tavily paper search failed: %s", exc)
            return None

        items = response.get("results", [])
        if not items:
            return None

        return {
            "abstracts": [r.get("content", "") for r in items],
            "titles": [r.get("title", "") for r in items],
            "urls": [r.get("url", "") for r in items],
            "source": "tavily",
        }

    def _search_github_tavily(
        self, query: str, max_results: int = 10
    ) -> list[dict[str, Any]] | None:
        if self._tavily is None:
            return None
        try:
            response = self._tavily.search_github(query, max_results=max_results)
        except TavilyPoolExhausted as exc:
            logger.warning("Tavily pool exhausted for GitHub search: %s", exc)
            return None
        except Exception as exc:
            logger.debug("Tavily GitHub search failed: %s", exc)
            return None

        items = response.get("results", [])
        repos = []
        for r in items:
            url = r.get("url", "")
            # Use proper URL parsing to verify the host is exactly github.com
            try:
                from urllib.parse import urlparse
                parsed = urlparse(url)
                host = parsed.netloc.lower().lstrip("www.")
                if host != "github.com":
                    continue
            except Exception:
                continue
            # Only keep actual repo URLs (not gists, issues, wiki pages, etc.)
            path_parts = [p for p in parsed.path.strip("/").split("/") if p]
            if len(path_parts) < 2:
                continue
            repos.append({
                "name": r.get("title", ""),
                "url": url,
                "description": r.get("content", "")[:200],
                "score": r.get("score", 0.0),
            })
        return repos or None

    # ------------------------------------------------------------------ #
    #  Semantic Scholar fallback                                            #
    # ------------------------------------------------------------------ #

    def _search_semantic_scholar(
        self, query: str, max_results: int
    ) -> dict[str, Any] | None:
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
        if not papers:
            return None
        return {
            "abstracts": [p.get("abstract") or "" for p in papers],
            "titles": [p.get("title") or "" for p in papers],
            "urls": [
                f"https://www.semanticscholar.org/paper/{p.get('paperId','')}"
                for p in papers
            ],
            "source": "semantic_scholar",
        }

    # ------------------------------------------------------------------ #
    #  arXiv fallback                                                       #
    # ------------------------------------------------------------------ #

    def _search_arxiv(
        self, query: str, max_results: int
    ) -> dict[str, Any] | None:
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

        titles = self._extract_xml_tags(content, "title")
        abstracts = self._extract_xml_tags(content, "summary")
        ids = self._extract_xml_tags(content, "id")

        titles = [t for t in titles if "arXiv" not in t][:max_results]
        abstracts = abstracts[:max_results]
        urls = [i for i in ids if "arxiv.org/abs" in i][:max_results]

        if not titles:
            return None
        return {"abstracts": abstracts, "titles": titles, "urls": urls, "source": "arxiv"}

    # ------------------------------------------------------------------ #
    #  Helpers                                                             #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _init_tavily(keys: list[str] | None) -> TavilyPool | None:
        """Try to build a TavilyPool; return None if no keys available."""
        try:
            return TavilyPool.from_env_or_list(keys)
        except TavilyPoolExhausted:
            logger.info(
                "No Tavily API keys configured — will fall back to Semantic Scholar / arXiv. "
                "Set TAVILY_API_KEY to enable Tavily search."
            )
            return None
        except Exception as exc:
            logger.warning("TavilyPool init failed: %s", exc)
            return None

    @staticmethod
    def _extract_xml_tags(xml: str, tag: str) -> list[str]:
        return re.findall(rf"<{tag}[^>]*>(.*?)</{tag}>", xml, re.DOTALL)
