"""arXiv connector for literature mining."""
from __future__ import annotations
import logging
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

ARXIV_API = "https://export.arxiv.org/api/query"
NS = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}


class ArxivConnector:
    """Fetches papers from arXiv API."""

    def __init__(self, max_results: int = 20) -> None:
        self.max_results = max_results

    def search(self, query: str) -> List[Dict[str, Any]]:
        params = urllib.parse.urlencode({
            "search_query": f"all:{query}",
            "start": 0,
            "max_results": self.max_results,
            "sortBy": "relevance",
            "sortOrder": "descending",
        })
        url = f"{ARXIV_API}?{params}"
        logger.debug("arXiv query: %s", url)
        try:
            with urllib.request.urlopen(url, timeout=30) as resp:
                data = resp.read()
            return self._parse(data)
        except Exception as exc:
            logger.warning("arXiv query failed: %s", exc)
            return []

    def _parse(self, xml_data: bytes) -> List[Dict[str, Any]]:
        root = ET.fromstring(xml_data)
        papers = []
        for entry in root.findall("atom:entry", NS):
            def text(tag: str) -> str:
                el = entry.find(tag, NS)
                return el.text.strip() if el is not None and el.text else ""

            arxiv_id_url = text("atom:id")
            arxiv_id = arxiv_id_url.split("/abs/")[-1] if "/abs/" in arxiv_id_url else arxiv_id_url
            papers.append({
                "source": "arxiv",
                "arxiv_id": arxiv_id,
                "title": text("atom:title"),
                "abstract": text("atom:summary"),
                "published": text("atom:published"),
                "url": arxiv_id_url,
                "authors": [
                    a.find("atom:name", NS).text.strip()
                    for a in entry.findall("atom:author", NS)
                    if a.find("atom:name", NS) is not None
                ],
            })
        return papers
