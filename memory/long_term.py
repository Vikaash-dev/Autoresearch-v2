"""
Long-Term Memory — persistent, semantic, cross-run knowledge store.

Stores everything the system learns across runs:
  - Papers found (with embeddings for semantic search)
  - Hypotheses tried (with their outcomes)
  - Experimental findings (confirmed results)
  - Research summaries (synthesised knowledge per topic)

Two query modes:
  1. Keyword / structured — exact field matching (always available)
  2. Semantic — cosine similarity over embeddings (requires numpy; gracefully
     degrades to keyword search when numpy is unavailable)

Inspired by:
  - NousResearch/hermes-agent (FTS5 SQLite memory, procedural long-term store)
  - plastic-labs/honcho (Pareto-frontier cross-session user memory)
  - Voyager (vector-indexed skill library with recency weighting)
  - AI-Scientist v2 (cross-run experiment results cache)

Storage: one JSON-Lines file per record type under long_term_path/.
    memory/long_term/
        papers.jsonl
        hypotheses.jsonl
        findings.jsonl
        summaries.jsonl
        embeddings.npy     ← optional; created on first semantic search
        embeddings_idx.json ← maps row → record_id
"""

from __future__ import annotations

import json
import logging
import math
import time
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ======================================================================= #
#  Record types                                                             #
# ======================================================================= #

@dataclass
class PaperRecord:
    """A research paper found during literature search."""
    record_id: str = field(default_factory=lambda: str(uuid.uuid4())[:10])
    title: str = ""
    abstract: str = ""
    url: str = ""
    source: str = ""          # "tavily" | "semantic_scholar" | "arxiv"
    year: int | None = None
    relevance_score: float = 0.0
    run_id: str = ""
    topic: str = ""
    added_at: float = field(default_factory=time.time)


@dataclass
class HypothesisRecord:
    """A hypothesis generated and tested by the system."""
    record_id: str = field(default_factory=lambda: str(uuid.uuid4())[:10])
    hypothesis: str = ""
    topic: str = ""
    outcome: str = "untested"     # untested | supported | refuted | inconclusive
    best_metric: float | None = None
    num_experiments: int = 0
    run_id: str = ""
    node_id: str = ""             # links to ExperimentNode on Blackboard
    added_at: float = field(default_factory=time.time)


@dataclass
class FindingRecord:
    """A confirmed experimental finding."""
    record_id: str = field(default_factory=lambda: str(uuid.uuid4())[:10])
    finding: str = ""             # natural language description
    metric: float | None = None
    metric_name: str = ""
    hypothesis_id: str = ""       # links to HypothesisRecord
    run_id: str = ""
    topic: str = ""
    confidence: float = 0.5       # 0–1
    added_at: float = field(default_factory=time.time)


@dataclass
class SummaryRecord:
    """A synthesised research summary for a topic."""
    record_id: str = field(default_factory=lambda: str(uuid.uuid4())[:10])
    topic: str = ""
    summary: str = ""
    key_findings: list[str] = field(default_factory=list)
    open_questions: list[str] = field(default_factory=list)
    run_ids: list[str] = field(default_factory=list)
    added_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)


# ======================================================================= #
#  Long-term memory store                                                  #
# ======================================================================= #

_STORES = {
    "papers":      ("papers.jsonl",     PaperRecord),
    "hypotheses":  ("hypotheses.jsonl", HypothesisRecord),
    "findings":    ("findings.jsonl",   FindingRecord),
    "summaries":   ("summaries.jsonl",  SummaryRecord),
}


class LongTermMemory:
    """
    Persistent, searchable cross-run knowledge store.

    Usage:
        ltm = LongTermMemory(Path("memory/long_term"))

        # Store
        ltm.store_paper(PaperRecord(title="Attention Is All You Need", ...))
        ltm.store_hypothesis(HypothesisRecord(hypothesis="...", outcome="supported"))

        # Retrieve
        papers = ltm.search_papers("sparse attention efficiency", top_k=10)
        past   = ltm.recall_hypotheses(topic="transformers", outcome="supported")
        all_findings = ltm.all_findings()
    """

    def __init__(self, root: Path) -> None:
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    #  Store                                                               #
    # ------------------------------------------------------------------ #

    def store_paper(self, record: PaperRecord) -> None:
        self._append("papers", asdict(record))

    def store_hypothesis(self, record: HypothesisRecord) -> None:
        self._append("hypotheses", asdict(record))

    def store_finding(self, record: FindingRecord) -> None:
        self._append("findings", asdict(record))

    def store_summary(self, record: SummaryRecord) -> None:
        """
        Upsert: if a summary for this topic already exists, update it;
        otherwise append a new record.
        """
        existing = self._load_all("summaries")
        for i, row in enumerate(existing):
            if row.get("topic", "").lower() == record.topic.lower():
                # Update in-place
                record.record_id = row["record_id"]
                record.added_at = row.get("added_at", record.added_at)
                record.updated_at = time.time()
                existing[i] = asdict(record)
                self._write_all("summaries", existing)
                return
        # New summary
        self._append("summaries", asdict(record))

    def update_hypothesis_outcome(
        self,
        record_id: str,
        outcome: str,
        best_metric: float | None = None,
        num_experiments: int | None = None,
    ) -> bool:
        """Update the outcome of a stored hypothesis. Returns True if found."""
        rows = self._load_all("hypotheses")
        for row in rows:
            if row["record_id"] == record_id:
                row["outcome"] = outcome
                if best_metric is not None:
                    row["best_metric"] = best_metric
                if num_experiments is not None:
                    row["num_experiments"] = num_experiments
                self._write_all("hypotheses", rows)
                return True
        return False

    # ------------------------------------------------------------------ #
    #  Retrieve — structured / keyword                                     #
    # ------------------------------------------------------------------ #

    def search_papers(
        self,
        query: str,
        top_k: int = 10,
        min_relevance: float = 0.0,
        topic: str | None = None,
        source: str | None = None,
    ) -> list[PaperRecord]:
        """
        Search stored papers by keyword matching across title + abstract.
        Results sorted by (keyword_score × relevance_score), newest first.
        """
        q_tokens = set(query.lower().split())
        results: list[tuple[float, PaperRecord]] = []

        for row in self._load_all("papers"):
            record = PaperRecord(**row)
            if record.relevance_score < min_relevance:
                continue
            if topic and topic.lower() not in record.topic.lower():
                continue
            if source and record.source != source:
                continue

            text = f"{record.title} {record.abstract}".lower()
            kw_score = sum(1 for t in q_tokens if t in text) / max(len(q_tokens), 1)
            combined = kw_score * 0.5 + record.relevance_score * 0.5
            if combined > 0 or not query:
                results.append((combined, record))

        results.sort(key=lambda x: (x[0], x[1].added_at), reverse=True)
        return [r for _, r in results[:top_k]]

    def recall_hypotheses(
        self,
        topic: str | None = None,
        outcome: str | None = None,
        run_id: str | None = None,
    ) -> list[HypothesisRecord]:
        """Return previously tested hypotheses, optionally filtered."""
        results = []
        for row in self._load_all("hypotheses"):
            record = HypothesisRecord(**row)
            if topic and topic.lower() not in record.topic.lower():
                continue
            if outcome and record.outcome != outcome:
                continue
            if run_id and record.run_id != run_id:
                continue
            results.append(record)
        return sorted(results, key=lambda r: r.added_at, reverse=True)

    def all_findings(
        self,
        topic: str | None = None,
        min_confidence: float = 0.0,
    ) -> list[FindingRecord]:
        """Return all stored findings, optionally filtered."""
        results = []
        for row in self._load_all("findings"):
            record = FindingRecord(**row)
            if topic and topic.lower() not in record.topic.lower():
                continue
            if record.confidence < min_confidence:
                continue
            results.append(record)
        return sorted(results, key=lambda r: r.confidence, reverse=True)

    def get_summary(self, topic: str) -> SummaryRecord | None:
        """Return the summary for a topic, or None if not found."""
        for row in self._load_all("summaries"):
            if row.get("topic", "").lower() == topic.lower():
                return SummaryRecord(**row)
        return None

    def all_summaries(self) -> list[SummaryRecord]:
        return [SummaryRecord(**row) for row in self._load_all("summaries")]

    # ------------------------------------------------------------------ #
    #  Semantic search (cosine similarity, numpy optional)                 #
    # ------------------------------------------------------------------ #

    def semantic_search_papers(
        self,
        query: str,
        top_k: int = 10,
        embed_fn: Any | None = None,
    ) -> list[PaperRecord]:
        """
        Semantic search over stored papers using cosine similarity.

        embed_fn: callable(text: str) -> list[float]
        Falls back to keyword search if embed_fn is None or numpy unavailable.
        """
        if embed_fn is None:
            logger.debug("LongTermMemory: no embed_fn — falling back to keyword search")
            return self.search_papers(query, top_k=top_k)

        try:
            import numpy as np
        except ImportError:
            logger.debug("LongTermMemory: numpy not available — falling back to keyword search")
            return self.search_papers(query, top_k=top_k)

        papers = [PaperRecord(**row) for row in self._load_all("papers")]
        if not papers:
            return []

        # Embed query
        q_vec = np.array(embed_fn(query), dtype=float)
        q_norm = np.linalg.norm(q_vec)
        if q_norm == 0:
            return self.search_papers(query, top_k=top_k)
        q_unit = q_vec / q_norm

        scored: list[tuple[float, PaperRecord]] = []
        for paper in papers:
            text = f"{paper.title} {paper.abstract[:300]}"
            doc_vec = np.array(embed_fn(text), dtype=float)
            doc_norm = np.linalg.norm(doc_vec)
            if doc_norm == 0:
                continue
            similarity = float(np.dot(q_unit, doc_vec / doc_norm))
            scored.append((similarity, paper))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [p for _, p in scored[:top_k]]

    # ------------------------------------------------------------------ #
    #  Deduplication helpers                                               #
    # ------------------------------------------------------------------ #

    def paper_already_stored(self, url: str) -> bool:
        """Return True if a paper with this URL is already in memory."""
        return any(row.get("url") == url for row in self._load_all("papers"))

    def hypothesis_already_tested(self, hypothesis: str, topic: str) -> HypothesisRecord | None:
        """Return a matching hypothesis record if the same hypothesis was tried before."""
        h_lower = hypothesis.lower()
        for row in self._load_all("hypotheses"):
            stored = row.get("hypothesis", "").lower()
            if topic.lower() in row.get("topic", "").lower():
                # Jaccard similarity on tokens
                h_toks = set(h_lower.split())
                s_toks = set(stored.split())
                if h_toks and s_toks:
                    jaccard = len(h_toks & s_toks) / len(h_toks | s_toks)
                    if jaccard > 0.7:
                        return HypothesisRecord(**row)
        return None

    # ------------------------------------------------------------------ #
    #  Stats                                                               #
    # ------------------------------------------------------------------ #

    def stats(self) -> dict[str, Any]:
        return {
            name: sum(1 for _ in self._load_all(name))
            for name in _STORES
        }

    # ------------------------------------------------------------------ #
    #  Internal I/O                                                         #
    # ------------------------------------------------------------------ #

    def _path(self, store: str) -> Path:
        filename, _ = _STORES[store]
        return self._root / filename

    def _append(self, store: str, record: dict[str, Any]) -> None:
        path = self._path(store)
        with path.open("a") as f:
            f.write(json.dumps(record, default=str) + "\n")

    def _load_all(self, store: str) -> list[dict[str, Any]]:
        path = self._path(store)
        if not path.exists():
            return []
        rows = []
        for line in path.read_text().splitlines():
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        return rows

    def _write_all(self, store: str, rows: list[dict[str, Any]]) -> None:
        path = self._path(store)
        tmp = path.with_suffix(".tmp")
        with tmp.open("w") as f:
            for row in rows:
                f.write(json.dumps(row, default=str) + "\n")
        tmp.replace(path)

    def __repr__(self) -> str:
        s = self.stats()
        return (
            f"LongTermMemory(papers={s['papers']}, hypotheses={s['hypotheses']}, "
            f"findings={s['findings']}, summaries={s['summaries']})"
        )
