"""Tests for config module."""
import pytest
from autoresearch.config.schema import AutoResearchConfig, load_config


def test_default_config():
    cfg = AutoResearchConfig()
    assert cfg.max_iterations == 3
    assert cfg.llm.provider == "openai"
    assert cfg.tom.enabled is True


def test_load_config_defaults():
    cfg = load_config()
    assert isinstance(cfg, AutoResearchConfig)
    assert cfg.max_iterations >= 1


def test_config_override():
    cfg = AutoResearchConfig(topic="quantum ML", max_iterations=5)
    assert cfg.topic == "quantum ML"
    assert cfg.max_iterations == 5
