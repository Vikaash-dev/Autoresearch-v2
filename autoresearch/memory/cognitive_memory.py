"""
CognitiveMemory — neuroscience-grounded memory for autoresearch agents.

Implements the four mechanisms introduced in autoresearch-engram
(github.com/tonitangpotato/autoresearch-engram) and formalised in the
neuroscience literature:

  ACT-R Activation  (Anderson, 1993)
      Memories with higher recency, frequency, and success float to the top.
      activation = importance * log(1 + frequency)
                   * exp(-decay_rate * age_hours)
                   + recency_bonus

  Hebbian Learning  (Hebb, 1949)
      Changes that co-occur in successful experiments are linked.
      Every time two memories appear in the same successful round their
      link strength increases (bounded at 10).  When recalling, linked
      memories receive an additional activation boost.

  Ebbinghaus Forgetting  (Ebbinghaus, 1885)
      Failed experiments decay faster than successful ones.
      decay_rate by category:
          episodic  discard/crash  →  0.08  (fast, ~12 h half-life)
          episodic  keep           →  0.01  (slow, ~70 h half-life)
          semantic                 →  0.002 (very slow, 14 d half-life)
          procedural               →  0.0005 (nearly permanent)

  Memory Consolidation  (Born & Diekelmann, 2010)
      After every N experiments raw episodic memories are synthesised
      into higher-level semantic patterns that resist decay.

Three memory types
------------------
  episodic    Specific experiment results — "increased LR to 0.04, improved
              by 0.004, status=keep".  Decays at normal rate.
  semantic    Cross-experiment patterns — "Architecture changes are 3×
              more effective than optimizer tweaks".  Decays slowly.
  procedural  Research heuristics — "Always check VRAM before doubling
              model size".  Nearly permanent.

Main interface
--------------
  cm = CognitiveMemory(db_path)

  # Before each experiment
  context = cm.recall("LR tuning, transformer architecture")

  # After each experiment
  mem_id = cm.store_memory(
      content="KEEP: increased LR to 0.04, val_bpb 0.993→0.989 (+0.004)",
      memory_type="episodic",
      importance=0.8,
      status="keep",
      tags=["lr", "optimizer"],
      round_id="abc123",
  )

  # Every 10 experiments
  pattern = cm.reflect()

  # At any time
  cm.hebbian_reinforce(["mem_id_1", "mem_id_2"])  # link two memories
"""

from __future__ import annotations

import json
import logging
import math
import re
import sqlite3
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, Generator, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Decay rates (Ebbinghaus forgetting by category)
# ---------------------------------------------------------------------------
_DECAY = {
    ("episodic",   "discard"): 0.08,
    ("episodic",   "crash"):   0.08,
    ("episodic",   "keep"):    0.01,
    ("semantic",   "keep"):    0.002,
    ("semantic",   ""):        0.002,
    ("procedural", "keep"):    0.0005,
    ("procedural", ""):        0.0005,
}
_DECAY_DEFAULT = 0.05  # fallback


def _decay_rate(memory_type: str, status: str) -> float:
    return _DECAY.get((memory_type, status), _DECAY_DEFAULT)


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------
_COGNITIVE_SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
    memory_id     TEXT PRIMARY KEY,
    content       TEXT NOT NULL,
    memory_type   TEXT NOT NULL DEFAULT 'episodic',
    importance    REAL NOT NULL DEFAULT 0.5,
    frequency     INTEGER NOT NULL DEFAULT 1,
    last_accessed REAL NOT NULL,
    created_at    REAL NOT NULL,
    status        TEXT NOT NULL DEFAULT 'keep',
    tags          TEXT NOT NULL DEFAULT '[]',
    round_id      TEXT NOT NULL DEFAULT '',
    metadata      TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS hebbian_links (
    link_id    TEXT PRIMARY KEY,
    memory_a   TEXT NOT NULL,
    memory_b   TEXT NOT NULL,
    strength   REAL NOT NULL DEFAULT 1.0,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_memories_type     ON memories(memory_type);
CREATE INDEX IF NOT EXISTS idx_memories_round    ON memories(round_id);
CREATE INDEX IF NOT EXISTS idx_memories_accessed ON memories(last_accessed);
CREATE INDEX IF NOT EXISTS idx_hebbian_a         ON hebbian_links(memory_a);
CREATE INDEX IF NOT EXISTS idx_hebbian_b         ON hebbian_links(memory_b);
"""


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class MemoryRecord:
    memory_id:     str   = field(default_factory=lambda: str(uuid.uuid4()))
    content:       str   = ""
    memory_type:   str   = "episodic"   # episodic | semantic | procedural
    importance:    float = 0.5
    frequency:     int   = 1
    last_accessed: float = field(default_factory=time.time)
    created_at:    float = field(default_factory=time.time)
    status:        str   = "keep"       # keep | discard | crash
    tags:          List[str] = field(default_factory=list)
    round_id:      str   = ""
    metadata:      Dict[str, Any] = field(default_factory=dict)

    # computed at retrieval
    activation: float = field(default=0.0, compare=False)

    def activation_score(self, now: Optional[float] = None) -> float:
        """
        ACT-R-style activation:
          base  = importance * log(1 + frequency)
          decay = exp(-rate * age_hours)
          recency = 1/(1 + age_hours) * 0.2   (small proximity bonus)
        """
        now = now or time.time()
        age_hours = max(0.0, (now - self.created_at) / 3600.0)
        rate = _decay_rate(self.memory_type, self.status)
        base = self.importance * math.log(1.0 + self.frequency)
        decay = math.exp(-rate * age_hours)
        recency = 0.2 / (1.0 + age_hours)
        return base * decay + recency


@dataclass
class RecallContext:
    """Structured memory context returned to the agent before an experiment."""
    what_worked:  List[str] = field(default_factory=list)
    what_failed:  List[str] = field(default_factory=list)
    patterns:     List[str] = field(default_factory=list)
    avoid_tags:   List[str] = field(default_factory=list)
    suggestions:  List[str] = field(default_factory=list)
    total_memories: int = 0
    hebbian_links:  int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "what_worked":    self.what_worked,
            "what_failed":    self.what_failed,
            "patterns":       self.patterns,
            "avoid_tags":     self.avoid_tags,
            "suggestions":    self.suggestions,
            "total_memories": self.total_memories,
            "hebbian_links":  self.hebbian_links,
        }

    def to_prompt_text(self) -> str:
        """Format as readable text to inject into agent prompts."""
        lines: List[str] = ["🧠 Memory recall:"]
        if self.what_worked:
            lines.append("  💡 What worked:")
            lines.extend(f"     • {w}" for w in self.what_worked[:5])
        if self.what_failed:
            lines.append("  ⚠️  What failed:")
            lines.extend(f"     • {f}" for f in self.what_failed[:3])
        if self.patterns:
            lines.append("  🧬 Patterns:")
            lines.extend(f"     • {p}" for p in self.patterns[:3])
        if self.avoid_tags:
            lines.append(f"  🚫 Avoid: {', '.join(self.avoid_tags[:10])}")
        if self.suggestions:
            lines.append("  💭 Suggestions:")
            lines.extend(f"     • {s}" for s in self.suggestions[:3])
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# CognitiveMemory
# ---------------------------------------------------------------------------

class CognitiveMemory:
    """
    Neuroscience-grounded memory layer for the SERA-X research system.

    Wraps the same SQLite database used by ExperienceStore, adding two new
    tables (memories, hebbian_links) without touching existing tables.

    Parameters
    ----------
    db_path : str
        Path to the SQLite file (or ":memory:" for ephemeral use in tests).
        Use the *same* path as ExperienceStore so both share one file.
    reflect_every : int
        Run the consolidation/reflection step after this many stored memories
        (default: 10, matching the Engram recommendation).
    """

    def __init__(
        self,
        db_path: str = "autoresearch_memory.db",
        reflect_every: int = 10,
    ) -> None:
        self.db_path = db_path
        self.reflect_every = reflect_every
        self._store_count = 0          # total memories stored this session
        self._mem_conn: Optional[sqlite3.Connection] = None
        if db_path == ":memory:":
            self._mem_conn = sqlite3.connect(":memory:", check_same_thread=False)
            self._mem_conn.row_factory = sqlite3.Row
        self._init_db()

    # ------------------------------------------------------------------
    # DB helpers
    # ------------------------------------------------------------------

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(_COGNITIVE_SCHEMA)
        logger.debug("CognitiveMemory initialised at %s", self.db_path)

    @contextmanager
    def _connect(self) -> Generator[sqlite3.Connection, None, None]:
        if self._mem_conn is not None:
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
    # STORE
    # ------------------------------------------------------------------

    def store_memory(
        self,
        content: str,
        memory_type: str = "episodic",
        importance: float = 0.5,
        status: str = "keep",
        tags: Optional[List[str]] = None,
        round_id: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Store a new memory.  Returns the memory_id.

        Importance guidelines (from engram README):
          episodic keep  →  0.8
          episodic discard → 0.4
          episodic crash → 0.6
          semantic pattern → 0.9
          procedural lesson → 0.95
        """
        if tags is None:
            tags = _auto_tags(content)
        mem = MemoryRecord(
            content=content,
            memory_type=memory_type,
            importance=importance,
            status=status,
            tags=tags,
            round_id=round_id,
            metadata=metadata or {},
        )
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO memories VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (
                    mem.memory_id, mem.content, mem.memory_type,
                    mem.importance, mem.frequency, mem.last_accessed,
                    mem.created_at, mem.status,
                    json.dumps(mem.tags), mem.round_id,
                    json.dumps(mem.metadata),
                ),
            )
        self._store_count += 1
        logger.debug("Stored memory %s (type=%s, status=%s)", mem.memory_id[:8], memory_type, status)

        # Auto-reflect every N memories (Memory Consolidation)
        if self._store_count % self.reflect_every == 0:
            self.reflect()

        return mem.memory_id

    def store_experiment(
        self,
        round_id: str,
        task: str,
        score: float,
        status: str,
        learnings: List[str],
        report_summary: str = "",
        tags: Optional[List[str]] = None,
    ) -> List[str]:
        """
        High-level helper: store all key memories from one orchestrator round.

        Stores:
          1. One episodic memory for the round outcome.
          2. One episodic memory per learning (if score > 0.3).
        Returns list of memory_ids so they can be Hebbian-reinforced.
        """
        is_success = score >= 0.5
        ep_importance = 0.8 if is_success else 0.4
        ep_status = "keep" if is_success else "discard"

        round_content = (
            f"{'KEEP' if is_success else 'DISCARD'}: Round on task '{task[:80]}'. "
            f"Score={score:.3f}. "
            f"{report_summary[:200]}"
        )
        ids = [
            self.store_memory(
                content=round_content,
                memory_type="episodic",
                importance=ep_importance,
                status=ep_status,
                tags=tags or _auto_tags(task),
                round_id=round_id,
            )
        ]

        # Store top learnings as episodic memories
        if is_success:
            for learning in learnings[:5]:
                mid = self.store_memory(
                    content=f"LEARNING: {learning[:300]}",
                    memory_type="episodic",
                    importance=min(0.9, ep_importance + 0.1),
                    status="keep",
                    tags=_auto_tags(learning),
                    round_id=round_id,
                )
                ids.append(mid)

        # Link co-occurring memories (Hebbian)
        if len(ids) >= 2:
            self.hebbian_reinforce(ids)

        return ids

    # ------------------------------------------------------------------
    # RECALL (ACT-R + Hebbian)
    # ------------------------------------------------------------------

    def recall(
        self,
        query: str,
        limit: int = 10,
        now: Optional[float] = None,
    ) -> RecallContext:
        """
        Recall relevant memories, ranked by ACT-R activation.

        Steps:
          1. Keyword filter on content AND tags (fixes the tag-only match case).
          2. Always include all semantic/procedural memories (slow-decay patterns).
          3. Compute activation score for each candidate.
          4. Boost scores of memories linked via Hebbian chains.
          5. Separate into what_worked / what_failed / patterns.
          6. Collect avoid_tags from failed memories.
          7. Synthesise suggestions from high-activation semantic memories.
        """
        now = now or time.time()
        terms = _query_terms(query)

        with self._connect() as conn:
            if terms:
                # Search both content and tags columns using qmark placeholders
                conds = " OR ".join(
                    "(LOWER(content) LIKE ? OR LOWER(tags) LIKE ?)"
                    for _ in terms
                )
                params: List[Any] = []
                for t in terms:
                    params.extend([f"%{t}%", f"%{t}%"])
                rows = conn.execute(
                    f"SELECT * FROM memories WHERE {conds} "  # noqa: S608
                    "ORDER BY last_accessed DESC LIMIT 200",
                    params,
                ).fetchall()
                # Always pull semantic/procedural memories (patterns apply to all tasks)
                extra = conn.execute(
                    "SELECT * FROM memories WHERE memory_type IN ('semantic','procedural') "
                    "ORDER BY last_accessed DESC LIMIT 50"
                ).fetchall()
                seen = {r["memory_id"] for r in rows}
                rows = list(rows) + [r for r in extra if r["memory_id"] not in seen]
            else:
                rows = conn.execute(
                    "SELECT * FROM memories ORDER BY last_accessed DESC LIMIT 200"
                ).fetchall()

            total = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
            link_count = conn.execute("SELECT COUNT(*) FROM hebbian_links").fetchone()[0]

        # Score candidates
        candidates: List[MemoryRecord] = []
        for r in rows:
            mem = _row_to_record(r)
            mem.activation = mem.activation_score(now)
            candidates.append(mem)

        # Hebbian boost: find the top-5 candidates, boost their linked memories
        top5_ids = [m.memory_id for m in sorted(candidates, key=lambda x: -x.activation)[:5]]
        hebbian_boost = self._hebbian_boost(top5_ids)
        for mem in candidates:
            if mem.memory_id in hebbian_boost:
                mem.activation += hebbian_boost[mem.memory_id] * 0.1

        # Sort and take top-N
        ranked = sorted(candidates, key=lambda x: -x.activation)[:limit]

        # Update access frequency (Hebbian reinforcement of accessed memories)
        self._update_access_frequency([m.memory_id for m in ranked])

        # Partition
        worked   = [m for m in ranked if m.status == "keep"    and m.memory_type == "episodic"]
        failed   = [m for m in ranked if m.status in ("discard", "crash")]
        patterns = [m for m in ranked if m.memory_type in ("semantic", "procedural")]

        # Collect avoid_tags from failed memories
        avoid: List[str] = []
        for m in failed:
            avoid.extend(_auto_tags(m.content))
        avoid_unique = list(dict.fromkeys(avoid))[:15]

        # Suggestions: top semantic pattern that is NOT yet in working memories
        suggestions = [m.content[:150] for m in patterns if m.status != "discard"][:3]

        ctx = RecallContext(
            what_worked  =[m.content[:200] for m in worked[:5]],
            what_failed  =[m.content[:200] for m in failed[:3]],
            patterns     =[m.content[:200] for m in patterns[:3]],
            avoid_tags   =avoid_unique,
            suggestions  =suggestions,
            total_memories=total,
            hebbian_links =link_count,
        )
        logger.debug(
            "Recall: %d worked, %d failed, %d patterns from %d total",
            len(ctx.what_worked), len(ctx.what_failed), len(ctx.patterns), total,
        )
        return ctx

    # ------------------------------------------------------------------
    # REFLECT (Memory Consolidation)
    # ------------------------------------------------------------------

    def reflect(self, n_recent: int = 30) -> Optional[str]:
        """
        Consolidation step (Born & Diekelmann, 2010).

        Scans the most recent episodic memories, finds recurring successful
        and failure patterns, and stores them as semantic memories that
        decay very slowly.

        Returns the content of the synthetic pattern memory (or None if
        there are not enough memories to synthesise from).
        """
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM memories WHERE memory_type='episodic' "
                "ORDER BY created_at DESC LIMIT ?",
                (n_recent,),
            ).fetchall()

        if len(rows) < 3:
            logger.debug("Reflect: not enough episodic memories (%d < 3)", len(rows))
            return None

        records = [_row_to_record(r) for r in rows]
        kept    = [r for r in records if r.status == "keep"]
        failed  = [r for r in records if r.status in ("discard", "crash")]

        # Collect tag frequencies
        kept_tags:   Dict[str, int] = {}
        failed_tags: Dict[str, int] = {}
        for m in kept:
            for t in m.tags:
                kept_tags[t] = kept_tags.get(t, 0) + 1
        for m in failed:
            for t in m.tags:
                failed_tags[t] = failed_tags.get(t, 0) + 1

        # Top tags
        top_worked = sorted(kept_tags, key=lambda k: -kept_tags[k])[:5]
        top_failed = sorted(failed_tags, key=lambda k: -failed_tags[k])[:5]

        n = len(records)
        keep_rate = len(kept) / n if n else 0.0

        pattern_content = (
            f"REFLECTION after {n} episodic memories: "
            f"keep_rate={keep_rate:.0%}. "
            f"Most effective themes: {', '.join(top_worked) or 'none'}. "
            f"Most avoided themes: {', '.join(top_failed) or 'none'}. "
            f"kept={len(kept)}, failed={len(failed)}."
        )

        mid = self.store_memory(
            content=pattern_content,
            memory_type="semantic",
            importance=0.9,
            status="keep",
            tags=top_worked + top_failed,
        )
        logger.info("Reflect: consolidated %d episodic → semantic memory %s", n, mid[:8])
        return pattern_content

    # ------------------------------------------------------------------
    # HEBBIAN reinforcement
    # ------------------------------------------------------------------

    def hebbian_reinforce(self, memory_ids: List[str], strength_delta: float = 0.5) -> None:
        """
        Reinforce Hebbian links between all pairs in memory_ids.

        Called automatically by store_experiment when multiple memories are
        stored in the same round.  Can also be called manually.
        """
        now = time.time()
        for i, a in enumerate(memory_ids):
            for b in memory_ids[i + 1:]:
                if a == b:
                    continue
                self._upsert_link(a, b, strength_delta, now)

    def _upsert_link(self, a: str, b: str, delta: float, now: float) -> None:
        # Canonical order so (a,b) == (b,a)
        lo, hi = (a, b) if a < b else (b, a)
        with self._connect() as conn:
            existing = conn.execute(
                "SELECT link_id, strength FROM hebbian_links WHERE memory_a=? AND memory_b=?",
                (lo, hi),
            ).fetchone()
            if existing:
                new_strength = min(10.0, existing["strength"] + delta)
                conn.execute(
                    "UPDATE hebbian_links SET strength=?, updated_at=? WHERE link_id=?",
                    (new_strength, now, existing["link_id"]),
                )
            else:
                conn.execute(
                    "INSERT INTO hebbian_links VALUES (?,?,?,?,?,?)",
                    (str(uuid.uuid4()), lo, hi, 1.0 + delta, now, now),
                )

    def _hebbian_boost(self, memory_ids: List[str]) -> Dict[str, float]:
        """Return {memory_id: total_linked_strength} for memories linked to the given ids."""
        if not memory_ids:
            return {}
        placeholders = ",".join("?" * len(memory_ids))
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT memory_a, memory_b, strength FROM hebbian_links "  # noqa: S608
                f"WHERE memory_a IN ({placeholders}) OR memory_b IN ({placeholders})",
                memory_ids + memory_ids,
            ).fetchall()
        boost: Dict[str, float] = {}
        for r in rows:
            a, b, s = r["memory_a"], r["memory_b"], r["strength"]
            if a in memory_ids and b not in memory_ids:
                boost[b] = boost.get(b, 0.0) + s
            elif b in memory_ids and a not in memory_ids:
                boost[a] = boost.get(a, 0.0) + s
        return boost

    def _update_access_frequency(self, memory_ids: List[str]) -> None:
        """Increment frequency and update last_accessed for recalled memories."""
        now = time.time()
        if not memory_ids:
            return
        placeholders = ",".join("?" * len(memory_ids))
        with self._connect() as conn:
            conn.execute(
                f"UPDATE memories SET frequency = frequency + 1, last_accessed = ? "  # noqa: S608
                f"WHERE memory_id IN ({placeholders})",
                [now] + memory_ids,
            )

    # ------------------------------------------------------------------
    # Summary + introspection
    # ------------------------------------------------------------------

    def summary(self) -> Dict[str, Any]:
        with self._connect() as conn:
            total = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
            by_type = conn.execute(
                "SELECT memory_type, COUNT(*) as n FROM memories GROUP BY memory_type"
            ).fetchall()
            by_status = conn.execute(
                "SELECT status, COUNT(*) as n FROM memories GROUP BY status"
            ).fetchall()
            links = conn.execute("SELECT COUNT(*) FROM hebbian_links").fetchone()[0]
            avg_freq = conn.execute("SELECT AVG(frequency) FROM memories").fetchone()[0] or 0.0
        return {
            "total_memories": total,
            "by_type":    {r["memory_type"]: r["n"] for r in by_type},
            "by_status":  {r["status"]: r["n"] for r in by_status},
            "hebbian_links": links,
            "avg_frequency": round(avg_freq, 2),
            "store_count_this_session": self._store_count,
            "db_path": self.db_path,
        }

    def list_memories(
        self,
        memory_type: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 50,
    ) -> List[MemoryRecord]:
        """List memories, optionally filtered by type or status."""
        clauses: List[str] = []
        params: List[Any] = []
        if memory_type:
            clauses.append("memory_type=?")
            params.append(memory_type)
        if status:
            clauses.append("status=?")
            params.append(status)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(limit)
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT * FROM memories {where} "  # noqa: S608
                "ORDER BY last_accessed DESC LIMIT ?",
                params,
            ).fetchall()
        return [_row_to_record(r) for r in rows]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _auto_tags(text: str, max_tags: int = 10) -> List[str]:
    """Extract meaningful keyword tags from free text."""
    stop = {
        "the", "and", "for", "that", "this", "with", "from", "are", "was",
        "were", "has", "have", "had", "not", "its", "also", "more", "been",
        "keep", "discard", "crash", "pattern", "reflection",
        "after", "before", "when", "then", "into", "onto",
    }
    words = re.findall(r"\b[a-z]{4,}\b", text.lower())
    seen: Dict[str, int] = {}
    for w in words:
        if w not in stop:
            seen[w] = seen.get(w, 0) + 1
    return sorted(seen, key=lambda k: -seen[k])[:max_tags]


def _query_terms(query: str) -> List[str]:
    """Extract search terms from a query string."""
    stop = {"what", "when", "where", "which", "about", "some", "more", "with"}
    terms = re.findall(r"\b[a-z]{4,}\b", query.lower())
    return [t for t in terms if t not in stop][:8]


def _row_to_record(row: sqlite3.Row) -> MemoryRecord:
    try:
        tags = json.loads(row["tags"])
    except (json.JSONDecodeError, TypeError):
        tags = []
    try:
        meta = json.loads(row["metadata"])
    except (json.JSONDecodeError, TypeError):
        meta = {}
    return MemoryRecord(
        memory_id    =row["memory_id"],
        content      =row["content"],
        memory_type  =row["memory_type"],
        importance   =row["importance"],
        frequency    =row["frequency"],
        last_accessed=row["last_accessed"],
        created_at   =row["created_at"],
        status       =row["status"],
        tags         =tags,
        round_id     =row["round_id"],
        metadata     =meta,
    )
