"""Experiment specification schema."""
from __future__ import annotations
import uuid
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ExperimentSpec(BaseModel):
    id: str = Field(default_factory=lambda: "exp_" + str(uuid.uuid4())[:8])
    hypothesis_id: str = ""
    title: str = ""
    description: str = ""
    code: str = ""
    language: str = "python"
    dependencies: List[str] = Field(default_factory=list)
    inputs: Dict[str, Any] = Field(default_factory=dict)
    expected_outputs: List[str] = Field(default_factory=list)
    timeout_seconds: int = 300
    max_retries: int = 3
    status: str = "pending"  # pending | running | success | failed
    result: Optional[Dict[str, Any]] = None
    artifacts: List[str] = Field(default_factory=list)
    metrics: Dict[str, Any] = Field(default_factory=dict)
    provenance: Dict[str, Any] = Field(default_factory=dict)
