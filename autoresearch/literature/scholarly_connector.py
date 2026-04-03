"""Pluggable scholarly source connector (Semantic Scholar / OpenAlex)."""
from __future__ import annotations
import logging
import urllib.request
import urllib.parse
import json
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


class ScholarlyConnector:
    """Backend-agnostic connector for scholarly APIs.

    Supported backends: 'semantic_scholar' (default), 'openalex'.
    """

    def __init__(self, backend: str = "semantic_scholar", api_key: Optional[str] = None) -> None:
        self.backend = backend
        self.api_key = api_key

    def search(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        if self.backend == "semantic_scholar":
            return self._semantic_scholar(query, limit)
        if self.backend == "openalex":
            return self._openalex(query, limit)
        logger.warning("Unknown scholarly backend: %s", self.backend)
        return []

    def _semantic_scholar(self, query: str, limit: int) -> List[Dict[str, Any]]:
        params = urllib.parse.urlencode({
            "query": query,
            "limit": limit,
            "fields": "title,abstract,year,authors,externalIds",
        })
        url = f"https://api.semanticscholar.org/graph/v1/paper/search?{params}"
        headers = {}
        if self.api_key:
            headers["x-api-key"] = self.api_key
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read())
            papers = []
            for p in data.get("data", []):
                papers.append({
                    "source": "semantic_scholar",
                    "paper_id": p.get("paperId"),
                    "title": p.get("title", ""),
                    "abstract": p.get("abstract", ""),
                    "year": p.get("year"),
                    "authors": [a.get("name") for a in p.get("authors", [])],
                    "doi": p.get("externalIds", {}).get("DOI"),
                })
            return papers
        except Exception as exc:
            logger.warning("Semantic Scholar query failed: %s", exc)
            return []

    def _openalex(self, query: str, limit: int) -> List[Dict[str, Any]]:
        params = urllib.parse.urlencode({"search": query, "per-page": limit})
        url = f"https://api.openalex.org/works?{params}"
        try:
            with urllib.request.urlopen(url, timeout=30) as resp:
                data = json.loads(resp.read())
            papers = []
            for work in data.get("results", []):
                papers.append({
                    "source": "openalex",
                    "openalex_id": work.get("id"),
                    "title": work.get("display_name", ""),
                    "abstract": work.get("abstract", ""),
                    "year": work.get("publication_year"),
                    "doi": work.get("doi"),
                    "authors": [
                        a.get("author", {}).get("display_name")
                        for a in work.get("authorships", [])
                    ],
                })
            return papers
        except Exception as exc:
            logger.warning("OpenAlex query failed: %s", exc)
            return []
