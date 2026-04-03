"""Configuration schema for AutoResearch v2."""
from __future__ import annotations
import yaml
from pathlib import Path
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any


class LLMConfig(BaseModel):
    provider: str = "openai"
    model: str = "gpt-4o"
    api_key_env: str = "OPENAI_API_KEY"
    temperature: float = 0.7
    max_tokens: int = 4096
    timeout: int = 120


class LiteratureConfig(BaseModel):
    arxiv_max_results: int = 20
    scholarly_backend: str = "semantic_scholar"
    semantic_scholar_api_key_env: str = "SEMANTIC_SCHOLAR_API_KEY"
    cache_dir: str = ".cache/literature"


class ExperimentConfig(BaseModel):
    sandbox: str = "subprocess"  # "subprocess" | "docker"
    docker_image: str = "python:3.11-slim"
    max_retries: int = 3
    timeout_seconds: int = 300
    artifacts_dir: str = "runs/{run_id}/artifacts"


class TomConfig(BaseModel):
    enabled: bool = True
    reviewer_personas: list[str] = ["methodological_purist", "empirical_skeptic", "novelty_maximizer"]
    adversarial_rounds: int = 2


class ReflectionConfig(BaseModel):
    enabled: bool = True
    max_policy_updates_per_run: int = 5
    safe_modification_only: bool = True


class CheckpointConfig(BaseModel):
    enabled: bool = True
    checkpoint_dir: str = "runs/{run_id}/checkpoints"
    save_every_n_steps: int = 1


class AutoResearchConfig(BaseModel):
    run_id: Optional[str] = None
    topic: str = ""
    max_iterations: int = 3
    stop_criteria: Dict[str, Any] = Field(default_factory=lambda: {"min_confidence": 0.7})
    llm: LLMConfig = Field(default_factory=LLMConfig)
    literature: LiteratureConfig = Field(default_factory=LiteratureConfig)
    experiment: ExperimentConfig = Field(default_factory=ExperimentConfig)
    tom: TomConfig = Field(default_factory=TomConfig)
    reflection: ReflectionConfig = Field(default_factory=ReflectionConfig)
    checkpoint: CheckpointConfig = Field(default_factory=CheckpointConfig)
    log_level: str = "INFO"


def load_config(path: Optional[str | Path] = None) -> AutoResearchConfig:
    """Load config from YAML/TOML file, falling back to defaults."""
    if path is None:
        defaults_path = Path(__file__).parent / "defaults.yaml"
        if defaults_path.exists():
            with open(defaults_path) as f:
                data = yaml.safe_load(f) or {}
            return AutoResearchConfig(**data)
        return AutoResearchConfig()
    path = Path(path)
    with open(path) as f:
        if path.suffix in (".yaml", ".yml"):
            data = yaml.safe_load(f) or {}
        else:
            import toml
            data = toml.load(f)
    return AutoResearchConfig(**data)
