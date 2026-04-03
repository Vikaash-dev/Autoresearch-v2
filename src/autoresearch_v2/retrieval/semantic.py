from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
import re
from pathlib import Path


TOKEN_RE = re.compile(r"[a-z0-9]+")


@dataclass(slots=True)
class PaperDocument:
    doc_id: str
    title: str
    abstract: str
    source: str
    year: int
    url: str = ""


@dataclass(slots=True)
class RetrievedPaper:
    doc_id: str
    title: str
    source: str
    score: float
    chunk: str
    evidence_id: str
    year: int
    url: str = ""


class SemanticRetriever:
    def search(self, query: str, top_k: int = 5) -> list[RetrievedPaper]:
        raise NotImplementedError


DEFAULT_OFFLINE_PAPERS: list[PaperDocument] = [
    PaperDocument(
        doc_id="arxiv:2603.19461",
        title="Hyperagents",
        abstract=(
            "Hyperagents defines self-referential and self-improving programs. "
            "The framework emphasizes recursive metacognitive self-modification and adaptation over time."
        ),
        source="arXiv",
        year=2026,
        url="https://arxiv.org/abs/2603.19461",
    ),
    PaperDocument(
        doc_id="related:group-evolving-agents-2026",
        title="Group-Evolving Agents: Open-Ended Self-Improvement via Experience Sharing",
        abstract=(
            "A multi-agent system where agents improve by sharing experiences and reusable skills. "
            "Targets open-ended self-improvement in collaborative settings."
        ),
        source="community-preprint",
        year=2026,
    ),
    PaperDocument(
        doc_id="related:agentfactory-2026",
        title="AgentFactory: A Self-Evolving Framework Through Executable Subagent Accumulation and Reuse",
        abstract=(
            "AgentFactory accumulates and reuses executable subagents to continuously expand capabilities. "
            "Focuses on growth by composition and retention of effective agent behaviors."
        ),
        source="community-preprint",
        year=2026,
    ),
    PaperDocument(
        doc_id="related:sage-2026",
        title="SAGE: Multi-Agent Self-Evolution for LLM Reasoning",
        abstract=(
            "SAGE applies self-evolution to improve large language model reasoning through multi-agent refinement loops."
        ),
        source="community-preprint",
        year=2026,
    ),
    PaperDocument(
        doc_id="related:evoskill-2026",
        title="EvoSkill: Automated Skill Discovery for Multi-Agent Systems",
        abstract=(
            "EvoSkill studies automated skill discovery and selection in multi-agent workflows."
        ),
        source="community-preprint",
        year=2026,
    ),
    PaperDocument(
        doc_id="related:semag-2026",
        title="SEMAG: Self-Evolutionary Multi-Agent Code Generation",
        abstract=(
            "SEMAG applies self-evolutionary multi-agent methods to code generation and iterative repair."
        ),
        source="community-preprint",
        year=2026,
    ),
    PaperDocument(
        doc_id="arxiv:2409.16299",
        title="HyperAgent: Generalist Software Engineering Agents to Solve Coding Tasks at Scale",
        abstract=(
            "HyperAgent is a software-engineering automation framework for coding tasks at scale. "
            "It is distinct from Hyperagents (2026) and does not define recursive metacognitive self-modification."
        ),
        source="arXiv",
        year=2024,
        url="https://arxiv.org/abs/2409.16299",
    ),
]


def _tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def _sparse_vector(tokens: list[str]) -> dict[str, float]:
    counts: dict[str, float] = {}
    for token in tokens:
        counts[token] = counts.get(token, 0.0) + 1.0
    if not counts:
        return counts
    norm = math.sqrt(sum(v * v for v in counts.values()))
    if norm == 0:
        return counts
    return {k: v / norm for k, v in counts.items()}


def _cosine_sparse(a: dict[str, float], b: dict[str, float]) -> float:
    if not a or not b:
        return 0.0
    if len(a) > len(b):
        a, b = b, a
    return sum(v * b.get(k, 0.0) for k, v in a.items())


class LocalCorpusIngestor:
    def load_jsonl(self, path: str | Path) -> list[PaperDocument]:
        rows: list[PaperDocument] = []
        with Path(path).open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                raw = json.loads(line)
                rows.append(PaperDocument(**raw))
        return rows

    def default_corpus(self) -> list[PaperDocument]:
        return list(DEFAULT_OFFLINE_PAPERS)

    def chunk_documents(
        self,
        documents: list[PaperDocument],
        chunk_size_words: int = 48,
        overlap_words: int = 8,
    ) -> list[dict[str, str | int]]:
        size = max(8, chunk_size_words)
        overlap = max(0, min(overlap_words, size // 2))
        step = max(1, size - overlap)
        chunks: list[dict[str, str | int]] = []
        for doc in documents:
            words = doc.abstract.split()
            if not words:
                continue
            idx = 0
            chunk_idx = 0
            while idx < len(words):
                chunk_words = words[idx : idx + size]
                text = " ".join(chunk_words).strip()
                if text:
                    chunks.append(
                        {
                            "evidence_id": f"{doc.doc_id}#chunk-{chunk_idx}",
                            "doc_id": doc.doc_id,
                            "title": doc.title,
                            "source": doc.source,
                            "year": doc.year,
                            "url": doc.url,
                            "text": text,
                        }
                    )
                idx += step
                chunk_idx += 1
        return chunks


class LocalSemanticRetriever(SemanticRetriever):
    def __init__(
        self,
        index_path: str | Path | None = None,
        documents: list[PaperDocument] | None = None,
    ) -> None:
        self._ingestor = LocalCorpusIngestor()
        self._index_path = Path(index_path) if index_path else None
        self._documents: list[PaperDocument] = []
        self._chunks: list[dict[str, str | int]] = []
        self._vectors: list[dict[str, float]] = []
        if self._index_path and self._index_path.exists():
            self._load_index(self._index_path)
        else:
            source_docs = documents if documents is not None else self._ingestor.default_corpus()
            self._build_index(source_docs)
            if self._index_path:
                self._persist_index(self._index_path)

    def _build_index(self, documents: list[PaperDocument]) -> None:
        self._documents = list(documents)
        self._chunks = self._ingestor.chunk_documents(self._documents)
        self._vectors = [_sparse_vector(_tokenize(str(c["text"]))) for c in self._chunks]

    def _persist_index(self, index_path: Path) -> None:
        index_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "documents": [asdict(doc) for doc in self._documents],
            "chunks": self._chunks,
        }
        index_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _load_index(self, index_path: Path) -> None:
        raw = json.loads(index_path.read_text(encoding="utf-8"))
        self._documents = [PaperDocument(**row) for row in raw.get("documents", [])]
        self._chunks = list(raw.get("chunks", []))
        self._vectors = [_sparse_vector(_tokenize(str(c.get("text", "")))) for c in self._chunks]

    def search(self, query: str, top_k: int = 5) -> list[RetrievedPaper]:
        qv = _sparse_vector(_tokenize(query))
        scored: list[tuple[float, int]] = []
        for i, vec in enumerate(self._vectors):
            score = _cosine_sparse(qv, vec)
            if score > 0:
                scored.append((score, i))
        scored.sort(key=lambda item: (-item[0], str(self._chunks[item[1]].get("doc_id", ""))))
        out: list[RetrievedPaper] = []
        for score, idx in scored[: max(1, top_k)]:
            chunk = self._chunks[idx]
            out.append(
                RetrievedPaper(
                    doc_id=str(chunk.get("doc_id", "")),
                    title=str(chunk.get("title", "")),
                    source=str(chunk.get("source", "")),
                    year=int(chunk.get("year", 0)),
                    score=round(float(score), 6),
                    chunk=str(chunk.get("text", "")),
                    evidence_id=str(chunk.get("evidence_id", "")),
                    url=str(chunk.get("url", "")),
                )
            )
        return out

