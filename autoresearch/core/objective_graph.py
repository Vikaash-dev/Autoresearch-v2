"""Directed objective graph for dynamic research workflow."""
from __future__ import annotations
import logging
from enum import Enum
from typing import Callable, Dict, List, Optional, Any
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ObjectiveStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class Objective(BaseModel):
    id: str
    name: str
    description: str = ""
    dependencies: List[str] = Field(default_factory=list)
    status: ObjectiveStatus = ObjectiveStatus.PENDING
    result: Optional[Any] = None
    error: Optional[str] = None
    priority: int = 0

    model_config = {"arbitrary_types_allowed": True}


class ObjectiveGraph:
    """Directed acyclic graph of research objectives."""

    def __init__(self) -> None:
        self._objectives: Dict[str, Objective] = {}
        self._handlers: Dict[str, Callable] = {}

    def add(self, objective: Objective, handler: Callable) -> None:
        self._objectives[objective.id] = objective
        self._handlers[objective.id] = handler

    def get_ready(self) -> List[Objective]:
        """Return objectives whose dependencies are all completed."""
        ready = []
        for obj in self._objectives.values():
            if obj.status != ObjectiveStatus.PENDING:
                continue
            deps_done = all(
                self._objectives[d].status == ObjectiveStatus.COMPLETED
                for d in obj.dependencies
                if d in self._objectives
            )
            if deps_done:
                ready.append(obj)
        return sorted(ready, key=lambda o: -o.priority)

    def mark_running(self, obj_id: str) -> None:
        self._objectives[obj_id].status = ObjectiveStatus.RUNNING

    def mark_completed(self, obj_id: str, result: Any = None) -> None:
        self._objectives[obj_id].status = ObjectiveStatus.COMPLETED
        self._objectives[obj_id].result = result

    def mark_failed(self, obj_id: str, error: str) -> None:
        self._objectives[obj_id].status = ObjectiveStatus.FAILED
        self._objectives[obj_id].error = error

    def is_done(self) -> bool:
        return all(
            o.status in (ObjectiveStatus.COMPLETED, ObjectiveStatus.FAILED, ObjectiveStatus.SKIPPED)
            for o in self._objectives.values()
        )

    def get_handler(self, obj_id: str) -> Optional[Callable]:
        return self._handlers.get(obj_id)

    def summary(self) -> Dict[str, str]:
        return {obj.id: obj.status.value for obj in self._objectives.values()}
