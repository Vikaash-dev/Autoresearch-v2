from __future__ import annotations


class ArxivClient:
    def search(self, topic: str, limit: int = 5) -> list[str]:
        return [f"arXiv::{topic}::{i+1}" for i in range(max(0, limit))]

