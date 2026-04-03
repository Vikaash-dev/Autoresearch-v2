"""Checkpoint management for resumable runs."""
from __future__ import annotations
import logging
from pathlib import Path
from typing import Optional
from .state import RunState

logger = logging.getLogger(__name__)


class CheckpointManager:
    def __init__(self, base_dir: str = "runs", run_id: Optional[str] = None, save_every: int = 1):
        self.run_id = run_id or "default"
        self.save_every = save_every
        self.checkpoint_dir = Path(base_dir) / self.run_id / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._step = 0

    def save(self, state: RunState) -> Optional[Path]:
        self._step += 1
        if self._step % self.save_every == 0:
            path = state.save(self.checkpoint_dir)
            logger.debug("Checkpoint saved: %s", path)
            return path
        return None

    def load_latest(self) -> Optional[RunState]:
        state_file = self.checkpoint_dir / "state.json"
        if state_file.exists():
            logger.info("Resuming from checkpoint: %s", state_file)
            return RunState.load(self.checkpoint_dir)
        return None

    def exists(self) -> bool:
        return (self.checkpoint_dir / "state.json").exists()
