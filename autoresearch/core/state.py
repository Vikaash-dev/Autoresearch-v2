"""Persistent run state for AutoResearch v2."""
from __future__ import annotations
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class RunState(BaseModel):
    run_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    topic: str = ""
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    status: str = "initialized"  # initialized | running | paused | completed | failed
    iteration: int = 0
    objectives_completed: List[str] = Field(default_factory=list)
    evidence: List[Dict[str, Any]] = Field(default_factory=list)
    hypotheses: List[Dict[str, Any]] = Field(default_factory=list)
    experiments: List[Dict[str, Any]] = Field(default_factory=list)
    reviews: List[Dict[str, Any]] = Field(default_factory=list)
    reflections: List[Dict[str, Any]] = Field(default_factory=list)
    policy_updates: List[Dict[str, Any]] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def touch(self) -> None:
        self.updated_at = datetime.now(timezone.utc).isoformat()

    def save(self, directory: Path) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "state.json"
        self.touch()
        path.write_text(self.model_dump_json(indent=2))
        return path

    @classmethod
    def load(cls, directory: Path) -> "RunState":
        path = directory / "state.json"
        return cls.model_validate_json(path.read_text())
