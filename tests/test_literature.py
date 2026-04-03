"""Tests for literature module."""
import pytest
from autoresearch.literature.evidence_registry import EvidenceRegistry, EvidenceItem
from autoresearch.literature.citation_grounding import CitationGrounding


def test_evidence_registry_add():
    registry = EvidenceRegistry()
    paper = {
        "source": "arxiv",
        "arxiv_id": "2401.00001",
        "title": "Test Paper on AI",
        "abstract": "This paper discusses artificial intelligence methods.",
        "authors": ["Alice", "Bob"],
        "published": "2024-01",
    }
    item = registry.add_paper(paper)
    assert item.title == "Test Paper on AI"
    assert len(registry) == 1


def test_evidence_registry_dedup():
    registry = EvidenceRegistry()
    paper = {"source": "arxiv", "arxiv_id": "2401.00001", "title": "Test", "abstract": ""}
    registry.add_paper(paper)
    registry.add_paper(paper)
    assert len(registry) == 1


def test_evidence_registry_search():
    registry = EvidenceRegistry()
    registry.add_paper({"source": "arxiv", "arxiv_id": "1", "title": "Deep Learning Survey", "abstract": "Neural networks"})
    registry.add_paper({"source": "arxiv", "arxiv_id": "2", "title": "Quantum Computing", "abstract": "Qubits and gates"})
    results = registry.search("deep learning")
    assert len(results) == 1
    assert "Deep Learning" in results[0].title


def test_citation_grounding():
    items = [
        EvidenceItem(source="arxiv", title="Transformers in NLP", abstract="Attention mechanisms for language tasks"),
    ]
    grounder = CitationGrounding(items)
    result = grounder.verify_claim("attention mechanism for NLP")
    assert isinstance(result["verified"], bool)
    assert "supporting_evidence" in result
