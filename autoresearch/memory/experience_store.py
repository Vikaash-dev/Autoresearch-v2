"""
Persistent Experience Store — SQLite-backed memory for the research system.

Inspired by:
- Bilevel Autoresearch (2026): multi-batch persistent experience, failure recycling
- Hermes Agent (NousResearch): skills auto-created from experience, FTS5 session search
- DGM-Hyperagents (Meta): performance tracking that emerges without being explicitly programmed

The store tracks every research round, agent run, and skill learned.
The MetaAgent queries this to decide what to try next.
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS rounds (
    round_id    TEXT PRIMARY KEY,
    task        TEXT NOT NULL,
    timestamp   REAL NOT NULL,
    score       REAL NOT NULL DEFAULT 0.0,
    status      TEXT NOT NULL DEFAULT 'completed',
    metadata    TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS agent_states (
    state_id    TEXT PRIMARY KEY,
    round_id    TEXT NOT NULL,
    agent_name  TEXT NOT NULL,
    agent_id    TEXT NOT NULL,
    round_num   INTEGER NOT NULL,
    score       REAL NOT NULL DEFAULT 0.0,
    status      TEXT NOT NULL,
    elapsed     REAL NOT NULL DEFAULT 0.0,
    parameters  TEXT NOT NULL DEFAULT '{}',
    result      TEXT NOT NULL DEFAULT '{}',
    timestamp   REAL NOT NULL,
    FOREIGN KEY (round_id) REFERENCES rounds(round_id)
);

CREATE TABLE IF NOT EXISTS skills (
    skill_id    TEXT PRIMARY KEY,
    name        TEXT NOT NULL UNIQUE,
    description TEXT NOT NULL,
    code        TEXT NOT NULL DEFAULT '',
    use_count   INTEGER NOT NULL DEFAULT 0,
    avg_score   REAL NOT NULL DEFAULT 0.0,
    created_at  REAL NOT NULL,
    updated_at  REAL NOT NULL,
    metadata    TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS learnings (
    learning_id TEXT PRIMARY KEY,
    round_id    TEXT NOT NULL,
    content     TEXT NOT NULL,
    source_url  TEXT NOT NULL DEFAULT '',
    timestamp   REAL NOT NULL,
    FOREIGN KEY (round_id) REFERENCES rounds(round_id)
);

CREATE INDEX IF NOT EXISTS idx_agent_states_round ON agent_states(round_id);
CREATE INDEX IF NOT EXISTS idx_agent_states_name  ON agent_states(agent_name);
CREATE INDEX IF NOT EXISTS idx_learnings_round    ON learnings(round_id);
"""


@dataclass
class RoundRecord:
    round_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    task: str = ""
    timestamp: float = field(default_factory=time.time)
    score: float = 0.0
    status: str = "completed"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SkillRecord:
    skill_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    code: str = ""
    use_count: int = 0
    avg_score: float = 0.0
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


class ExperienceStore:
    """
    SQLite-backed persistent experience store.

    Provides:
    - Round tracking (every full research cycle)
    - Agent state history (bounded, queryable)
    - Skill registry (Hermes-style: skills grow from experience)
    - Failure recycling (Bilevel: failed rounds feed the outer loop)
    - Cross-session search (FTS-style via LIKE queries on learnings)
    """

    def __init__(self, db_path: str = "autoresearch_memory.db") -> None:
        self.db_path = db_path
        # For in-memory databases, keep a single persistent connection
        # (each new sqlite3.connect(":memory:") creates an independent DB)
        self._mem_conn: Optional[sqlite3.Connection] = None
        if db_path == ":memory:":
            self._mem_conn = sqlite3.connect(":memory:", check_same_thread=False)
            self._mem_conn.row_factory = sqlite3.Row
        self._init_db()

    # ------------------------------------------------------------------
    # DB setup
    # ------------------------------------------------------------------

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(_SCHEMA)
        logger.debug("ExperienceStore initialised at %s", self.db_path)

    @contextmanager
    def _connect(self) -> Generator[sqlite3.Connection, None, None]:
        if self._mem_conn is not None:
            # In-memory: reuse the single persistent connection
            try:
                yield self._mem_conn
                self._mem_conn.commit()
            except Exception:
                self._mem_conn.rollback()
                raise
        else:
            conn = sqlite3.connect(str(self.db_path))
            conn.row_factory = sqlite3.Row
            try:
                yield conn
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            finally:
                conn.close()

    # ------------------------------------------------------------------
    # Round lifecycle
    # ------------------------------------------------------------------

    def start_round(self, task: str, metadata: Optional[Dict] = None) -> RoundRecord:
        rec = RoundRecord(task=task, status="running", metadata=metadata or {})
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO rounds VALUES (?,?,?,?,?,?)",
                (rec.round_id, rec.task, rec.timestamp, rec.score,
                 rec.status, json.dumps(rec.metadata)),
            )
        return rec

    def finish_round(self, round_id: str, score: float, status: str = "completed") -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE rounds SET score=?, status=? WHERE round_id=?",
                (score, status, round_id),
            )

    def get_round(self, round_id: str) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM rounds WHERE round_id=?", (round_id,)
            ).fetchone()
        return dict(row) if row else None

    def recent_rounds(self, n: int = 10) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM rounds ORDER BY timestamp DESC LIMIT ?", (n,)
            ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Agent state persistence
    # ------------------------------------------------------------------

    def save_agent_state(self, round_id: str, state: Any) -> None:
        """Persist an AgentState from base_agent.py."""
        # Safely convert AgentStatus enum to string
        raw_status = getattr(state, "status", None)
        if raw_status is None:
            status_str = "unknown"
        elif hasattr(raw_status, "name"):
            status_str = raw_status.name
        else:
            status_str = str(raw_status)

        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO agent_states VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (
                    str(uuid.uuid4()),
                    round_id,
                    getattr(state, "agent_class", type(state).__name__),
                    getattr(state, "agent_id", ""),
                    getattr(state, "round_number", 0),
                    float(getattr(state, "score", 0.0)),
                    status_str,
                    float(getattr(state, "elapsed_seconds", 0.0)),
                    json.dumps(getattr(state, "parameters", {}), default=str),
                    json.dumps(getattr(state, "result", {}), default=str)[:4096],
                    time.time(),
                ),
            )

    def agent_history(
        self, agent_name: str, n: int = 50
    ) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM agent_states WHERE agent_name=? "
                "ORDER BY timestamp DESC LIMIT ?",
                (agent_name, n),
            ).fetchall()
        return [dict(r) for r in rows]

    def best_agent_parameters(self, agent_name: str) -> Dict[str, Any]:
        """Return the parameter set that produced the highest score for this agent."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT parameters FROM agent_states WHERE agent_name=? "
                "ORDER BY score DESC LIMIT 1",
                (agent_name,),
            ).fetchone()
        if row:
            try:
                return json.loads(row["parameters"])
            except (json.JSONDecodeError, KeyError):
                pass
        return {}

    # ------------------------------------------------------------------
    # Failure recycling (Bilevel Autoresearch)
    # ------------------------------------------------------------------

    def recyclable_failures(self, min_score: float = 0.2) -> List[Dict[str, Any]]:
        """
        Return failed agent states whose score is above min_score.
        These are 'near-misses' that carry useful signal for the outer loop.
        """
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM agent_states WHERE status='FAILED' AND score>=? "
                "ORDER BY score DESC LIMIT 20",
                (min_score,),
            ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Skills (Hermes-style: grow from experience)
    # ------------------------------------------------------------------

    def save_skill(self, skill: SkillRecord) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO skills VALUES (?,?,?,?,?,?,?,?,?)",
                (skill.skill_id, skill.name, skill.description, skill.code,
                 skill.use_count, skill.avg_score, skill.created_at,
                 skill.updated_at, json.dumps(skill.metadata)),
            )

    def get_skill(self, name: str) -> Optional[SkillRecord]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM skills WHERE name=?", (name,)
            ).fetchone()
        if not row:
            return None
        return SkillRecord(
            skill_id=row["skill_id"],
            name=row["name"],
            description=row["description"],
            code=row["code"],
            use_count=row["use_count"],
            avg_score=row["avg_score"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            metadata=json.loads(row["metadata"]),
        )

    def update_skill_score(self, name: str, new_score: float) -> None:
        """Running average update — skill improves with use (Hermes pattern)."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT use_count, avg_score FROM skills WHERE name=?", (name,)
            ).fetchone()
            if row:
                n = row["use_count"] + 1
                avg = (row["avg_score"] * row["use_count"] + new_score) / n
                conn.execute(
                    "UPDATE skills SET use_count=?, avg_score=?, updated_at=? WHERE name=?",
                    (n, avg, time.time(), name),
                )

    def list_skills(self, min_score: float = 0.0) -> List[SkillRecord]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM skills WHERE avg_score>=? ORDER BY avg_score DESC",
                (min_score,),
            ).fetchall()
        return [
            SkillRecord(
                skill_id=r["skill_id"], name=r["name"],
                description=r["description"], code=r["code"],
                use_count=r["use_count"], avg_score=r["avg_score"],
                created_at=r["created_at"], updated_at=r["updated_at"],
                metadata=json.loads(r["metadata"]),
            )
            for r in rows
        ]

    # ------------------------------------------------------------------
    # Learnings persistence + search
    # ------------------------------------------------------------------

    def save_learnings(self, round_id: str, learnings: List[str], urls: Optional[List[str]] = None) -> None:
        urls = urls or []
        with self._connect() as conn:
            for i, learning in enumerate(learnings):
                conn.execute(
                    "INSERT INTO learnings VALUES (?,?,?,?,?)",
                    (str(uuid.uuid4()), round_id, learning,
                     urls[i] if i < len(urls) else "", time.time()),
                )

    def search_learnings(self, query: str, n: int = 20) -> List[str]:
        """Full-text search over accumulated learnings (FTS5-style via LIKE)."""
        terms = re.findall(r"\w{4,}", query.lower())
        if not terms:
            return []
        conditions = " OR ".join("LOWER(content) LIKE ?" for _ in terms)
        params = [f"%{t}%" for t in terms] + [n]
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT content FROM learnings WHERE {conditions} "  # noqa: S608
                f"ORDER BY timestamp DESC LIMIT ?",
                params,
            ).fetchall()
        return [r["content"] for r in rows]

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> Dict[str, Any]:
        with self._connect() as conn:
            total_rounds = conn.execute("SELECT COUNT(*) FROM rounds").fetchone()[0]
            avg_score = conn.execute("SELECT AVG(score) FROM rounds").fetchone()[0] or 0.0
            total_skills = conn.execute("SELECT COUNT(*) FROM skills").fetchone()[0]
            total_learnings = conn.execute("SELECT COUNT(*) FROM learnings").fetchone()[0]
        return {
            "total_rounds": total_rounds,
            "average_score": round(avg_score, 3),
            "total_skills": total_skills,
            "total_learnings": total_learnings,
            "db_path": self.db_path,
        }


def AgentStatusStr(state: Any) -> str:
    """Extract status string safely from an AgentState."""
    status = getattr(state, "status", None)
    if status is None:
        return "unknown"
    if hasattr(status, "name"):
        return status.name
    return str(status)
