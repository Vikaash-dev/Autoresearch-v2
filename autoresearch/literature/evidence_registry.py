"""Evidence registry schema and implementation."""
from __future__ import annotations
import uuid
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class EvidenceItem(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:12])
    source: str
    title: str
    abstract: str = ""
    authors: List[str] = Field(default_factory=list)
    year: Optional[str] = None
    url: Optional[str] = None
    doi: Optional[str] = None
    arxiv_id: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    claims: List[str] = Field(default_factory=list)
    verified: bool = False
    metadata: Dict[str, Any] = Field(default_factory=dict)


class EvidenceRegistry:
    """In-memory evidence registry with de-duplication."""

    def __init__(self) -> None:
        self._items: Dict[str, EvidenceItem] = {}

    def add_paper(self, paper: Dict[str, Any]) -> EvidenceItem:
        # De-duplicate by arxiv_id or title
        key = paper.get("arxiv_id") or paper.get("doi") or paper.get("title", "")
        if key in self._items:
            return self._items[key]
        item = EvidenceItem(
            source=paper.get("source", "unknown"),
            title=paper.get("title", ""),
            abstract=paper.get("abstract", ""),
            authors=paper.get("authors", []),
            year=str(paper.get("year", paper.get("published", ""))),
            url=paper.get("url"),
            doi=paper.get("doi"),
            arxiv_id=paper.get("arxiv_id"),
        )
        self._items[key] = item
        return item

    def get(self, item_id: str) -> Optional[EvidenceItem]:
        for item in self._items.values():
            if item.id == item_id:
                return item
        return None

    def all(self) -> List[EvidenceItem]:
        return list(self._items.values())

    def search(self, keyword: str) -> List[EvidenceItem]:
        kw = keyword.lower()
        return [
            item for item in self._items.values()
            if kw in item.title.lower() or kw in item.abstract.lower()
        ]

    def __len__(self) -> int:
        return len(self._items)
