"""Tests for Theory-of-Mind module."""
import pytest
from autoresearch.tom.profiles import ReviewerPersonaRegistry
from autoresearch.tom.intent import CollaboratorIntentProfile
from autoresearch.tom.adversarial import AdversarialChallengeGenerator
from autoresearch.config.schema import AutoResearchConfig


def test_reviewer_persona_registry():
    registry = ReviewerPersonaRegistry()
    persona = registry.get("methodological_purist")
    assert persona is not None
    assert persona.label == "Methodological Purist"
    assert len(persona.rejection_triggers) > 0


def test_adversarial_challenge_generation():
    registry = ReviewerPersonaRegistry()
    generator = AdversarialChallengeGenerator(registry)
    persona = registry.get("empirical_skeptic")
    hyp = {"id": "hyp_1", "title": "Attention Improves Reasoning"}
    challenge = generator.generate(hyp, persona)
    assert challenge["persona"] == "empirical_skeptic"
    assert len(challenge["challenges"]) > 0


def test_minority_report():
    registry = ReviewerPersonaRegistry()
    generator = AdversarialChallengeGenerator(registry)
    hyp = {"id": "hyp_1", "title": "Test Hypothesis"}
    report = generator.generate_minority_report(hyp)
    assert "minority_reports" in report


def test_collaborator_intent_alignment():
    intent = CollaboratorIntentProfile(novelty_weight=0.8, rigor_weight=0.9)
    score = intent.alignment_score({"novelty_score": 0.7, "rigor_score": 0.8})
    assert 0.0 <= score <= 2.0


def test_intent_from_config():
    cfg = AutoResearchConfig(stop_criteria={"min_confidence": 0.8})
    intent = CollaboratorIntentProfile.from_config(cfg)
    assert intent.rigor_weight == 0.8
