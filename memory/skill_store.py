"""
Persistent, versioned skill store for Autoresearch-v2.

Skills are text artifacts (system prompts, code snippets, tool descriptions)
that agents use and that the GEPA evolution engine improves over time.

Inspired by:
  - NousResearch/hermes-agent skill system (procedural memory)
  - Voyager's vector-indexed executable skill library
  - hermes-agent-self-evolution Phase 1 (SKILL.md files)

Storage layout (all under skill_store_path):
    skills/
      <name>/
        current.json      ← active version
        history/
          <timestamp>.json  ← previous versions

Each skill record:
    {
      "name":        str,
      "artifact":    str,          # the skill text/code itself
      "artifact_type": str,        # "prompt" | "code" | "tool_desc" | "config"
      "metric":      float,        # last evaluated performance
      "version":     int,
      "created_at":  float,
      "updated_at":  float,
      "tags":        [str],
      "description": str,
    }
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class SkillStore:
    """
    Persistent, versioned store for agent skills.

    Usage:
        store = SkillStore(Path("memory/skills"))
        store.save("hypothesis_system_prompt", artifact=PROMPT_TEXT, artifact_type="prompt")
        skill = store.load("hypothesis_system_prompt")
        best = store.get_best(artifact_type="prompt")
    """

    def __init__(self, root: Path) -> None:
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    #  Write                                                               #
    # ------------------------------------------------------------------ #

    def save(
        self,
        name: str,
        artifact: str,
        artifact_type: str = "prompt",
        metric: float = 0.0,
        description: str = "",
        tags: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Save or update a skill. Existing versions are archived to history/.

        Returns the saved skill record.
        """
        skill_dir = self._root / name
        skill_dir.mkdir(parents=True, exist_ok=True)
        history_dir = skill_dir / "history"
        history_dir.mkdir(exist_ok=True)

        current_path = skill_dir / "current.json"
        now = time.time()

        # Archive current version before overwriting
        if current_path.exists():
            existing = json.loads(current_path.read_text())
            ts = int(existing.get("updated_at", now))
            archive_path = history_dir / f"{ts}.json"
            archive_path.write_text(json.dumps(existing, indent=2))

        # Determine version number
        history_files = list(history_dir.glob("*.json"))
        version = len(history_files) + 1

        record: dict[str, Any] = {
            "name": name,
            "artifact": artifact,
            "artifact_type": artifact_type,
            "metric": metric,
            "version": version,
            "created_at": json.loads(current_path.read_text()).get("created_at", now)
            if current_path.exists()
            else now,
            "updated_at": now,
            "tags": tags or [],
            "description": description,
        }

        tmp = current_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(record, indent=2))
        tmp.replace(current_path)

        logger.debug("SkillStore: saved skill %r v%d (metric=%.4f)", name, version, metric)
        return record

    # ------------------------------------------------------------------ #
    #  Read                                                                #
    # ------------------------------------------------------------------ #

    def load(self, name: str) -> dict[str, Any] | None:
        """Load the current version of a skill. Returns None if not found."""
        path = self._root / name / "current.json"
        if not path.exists():
            return None
        return json.loads(path.read_text())

    def load_artifact(self, name: str) -> str | None:
        """Convenience method: load just the artifact text of a skill."""
        record = self.load(name)
        return record["artifact"] if record else None

    def load_version(self, name: str, version: int) -> dict[str, Any] | None:
        """Load a specific historical version of a skill."""
        history_dir = self._root / name / "history"
        if not history_dir.exists():
            return None
        files = sorted(history_dir.glob("*.json"))
        # version 1 = oldest archive, counting from 1
        idx = version - 1
        if idx < 0 or idx >= len(files):
            return None
        return json.loads(files[idx].read_text())

    def list_all(self, artifact_type: str | None = None) -> list[dict[str, Any]]:
        """List all current skills, optionally filtered by artifact_type."""
        skills = []
        for skill_dir in sorted(self._root.iterdir()):
            current = skill_dir / "current.json"
            if current.exists():
                record = json.loads(current.read_text())
                if artifact_type is None or record.get("artifact_type") == artifact_type:
                    skills.append(record)
        return skills

    def get_best(
        self, artifact_type: str | None = None, tags: list[str] | None = None
    ) -> dict[str, Any] | None:
        """Return the skill with the highest metric, optionally filtered."""
        candidates = self.list_all(artifact_type=artifact_type)
        if tags:
            candidates = [s for s in candidates if any(t in s.get("tags", []) for t in tags)]
        if not candidates:
            return None
        return max(candidates, key=lambda s: s.get("metric", 0.0))

    def search(self, query: str, artifact_type: str | None = None) -> list[dict[str, Any]]:
        """
        Simple keyword search across skill names, descriptions, and artifact text.
        Returns skills sorted by relevance (number of keyword matches).
        """
        q_tokens = set(query.lower().split())
        results = []
        for skill in self.list_all(artifact_type=artifact_type):
            text = " ".join([
                skill.get("name", ""),
                skill.get("description", ""),
                skill.get("artifact", "")[:500],
            ]).lower()
            score = sum(1 for t in q_tokens if t in text)
            if score > 0:
                results.append((score, skill))
        return [s for _, s in sorted(results, key=lambda x: x[0], reverse=True)]

    # ------------------------------------------------------------------ #
    #  Delete / maintenance                                                 #
    # ------------------------------------------------------------------ #

    def delete(self, name: str) -> bool:
        """Delete a skill and all its history. Returns True if it existed."""
        import shutil
        skill_dir = self._root / name
        if skill_dir.exists():
            shutil.rmtree(skill_dir)
            logger.info("SkillStore: deleted skill %r", name)
            return True
        return False

    def history(self, name: str) -> list[dict[str, Any]]:
        """Return all historical versions of a skill, oldest first."""
        history_dir = self._root / name / "history"
        if not history_dir.exists():
            return []
        return [
            json.loads(f.read_text())
            for f in sorted(history_dir.glob("*.json"))
        ]

    def rollback(self, name: str) -> dict[str, Any] | None:
        """
        Roll back a skill to the previous version.
        Returns the restored record, or None if no history exists.
        """
        hist = self.history(name)
        if not hist:
            logger.warning("SkillStore: no history for %r — cannot rollback", name)
            return None
        previous = hist[-1]
        return self.save(
            name=previous["name"],
            artifact=previous["artifact"],
            artifact_type=previous.get("artifact_type", "prompt"),
            metric=previous.get("metric", 0.0),
            description=previous.get("description", ""),
            tags=previous.get("tags", []),
        )

    def __len__(self) -> int:
        return sum(1 for d in self._root.iterdir() if (d / "current.json").exists())

    def __repr__(self) -> str:
        return f"SkillStore(root={self._root!r}, skills={len(self)})"
