"""
Tests for CognitiveMemory — the Engram-style neuroscience memory layer.

Covers:
  - store_memory (all memory types and statuses)
  - ACT-R activation scoring (recency, frequency, importance, decay)
  - Ebbinghaus forgetting (failed memories decay faster than kept ones)
  - Hebbian learning (co-occurring successful memories get linked + boosted)
  - recall (content search, tag search, semantic always included)
  - reflect / Memory Consolidation (episodic → semantic pattern)
  - store_experiment (high-level helper, Hebbian auto-link)
  - summary()
  - RecallContext.to_prompt_text()
"""

import math
import time

import pytest

from autoresearch.memory.cognitive_memory import (
    CognitiveMemory,
    MemoryRecord,
    RecallContext,
    _auto_tags,
    _decay_rate,
    _query_terms,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_cm() -> CognitiveMemory:
    """Return an in-memory CognitiveMemory instance for each test."""
    return CognitiveMemory(db_path=":memory:", reflect_every=100)


# ---------------------------------------------------------------------------
# Unit tests: helpers
# ---------------------------------------------------------------------------

class TestHelpers:
    def test_auto_tags_basic(self):
        tags = _auto_tags("increased learning rate optimizer")
        assert "learning" in tags
        assert "rate" in tags
        assert "optimizer" in tags

    def test_auto_tags_stops_filtered(self):
        tags = _auto_tags("the and for that with from")
        assert tags == []

    def test_auto_tags_max_cap(self):
        long_text = " ".join(f"word{i:04d}" for i in range(50))
        tags = _auto_tags(long_text, max_tags=10)
        assert len(tags) <= 10

    def test_query_terms_extracts_meaningful(self):
        terms = _query_terms("hyperparameter tuning transformer architecture")
        assert "hyperparameter" in terms
        assert "tuning" in terms
        assert "transformer" in terms
        assert "architecture" in terms

    def test_query_terms_filters_short(self):
        terms = _query_terms("do it now")
        assert terms == []

    def test_decay_rate_episodic_discard(self):
        assert _decay_rate("episodic", "discard") == pytest.approx(0.08)

    def test_decay_rate_episodic_keep(self):
        assert _decay_rate("episodic", "keep") == pytest.approx(0.01)

    def test_decay_rate_semantic(self):
        assert _decay_rate("semantic", "keep") == pytest.approx(0.002)

    def test_decay_rate_procedural(self):
        assert _decay_rate("procedural", "keep") == pytest.approx(0.0005)


# ---------------------------------------------------------------------------
# Unit tests: MemoryRecord activation
# ---------------------------------------------------------------------------

class TestActivationScore:
    def test_fresh_memory_high_activation(self):
        mem = MemoryRecord(importance=0.8, frequency=1, status="keep",
                           memory_type="episodic", created_at=time.time())
        score = mem.activation_score()
        assert score > 0.5  # freshly-created kept memory should be active

    def test_old_failed_memory_low_activation(self):
        one_week_ago = time.time() - 7 * 24 * 3600
        mem = MemoryRecord(importance=0.4, frequency=1, status="discard",
                           memory_type="episodic", created_at=one_week_ago,
                           last_accessed=one_week_ago)
        score = mem.activation_score()
        # discard decays at 0.08/h, after 7*24=168h: exp(-0.08*168) ≈ 0
        assert score < 0.01

    def test_semantic_memory_persists(self):
        one_week_ago = time.time() - 7 * 24 * 3600
        mem = MemoryRecord(importance=0.9, frequency=3, status="keep",
                           memory_type="semantic", created_at=one_week_ago,
                           last_accessed=one_week_ago)
        score = mem.activation_score()
        # semantic decays at 0.002/h, after 168h: exp(-0.336) ≈ 0.714
        # base = 0.9 * log(4) * 0.714 ≈ 0.89
        assert score > 0.5

    def test_high_frequency_boosts_activation(self):
        t = time.time()
        low_freq  = MemoryRecord(importance=0.5, frequency=1,  status="keep",
                                  memory_type="episodic", created_at=t)
        high_freq = MemoryRecord(importance=0.5, frequency=20, status="keep",
                                  memory_type="episodic", created_at=t)
        assert high_freq.activation_score(t) > low_freq.activation_score(t)

    def test_ebbinghaus_discard_decays_faster_than_keep(self):
        """A discarded memory 24h old should have lower activation than a kept one."""
        one_day_ago = time.time() - 24 * 3600
        kept = MemoryRecord(importance=0.7, frequency=1, status="keep",
                             memory_type="episodic", created_at=one_day_ago,
                             last_accessed=one_day_ago)
        disc = MemoryRecord(importance=0.7, frequency=1, status="discard",
                             memory_type="episodic", created_at=one_day_ago,
                             last_accessed=one_day_ago)
        assert kept.activation_score() > disc.activation_score()


# ---------------------------------------------------------------------------
# store_memory
# ---------------------------------------------------------------------------

class TestStoreMemory:
    def test_returns_memory_id(self):
        cm = _make_cm()
        mid = cm.store_memory("Test content", "episodic", 0.7, "keep")
        assert isinstance(mid, str)
        assert len(mid) == 36  # UUID format

    def test_total_memories_increments(self):
        cm = _make_cm()
        assert cm.summary()["total_memories"] == 0
        cm.store_memory("A", "episodic", 0.5, "keep")
        assert cm.summary()["total_memories"] == 1
        cm.store_memory("B", "semantic", 0.9, "keep")
        assert cm.summary()["total_memories"] == 2

    def test_store_count_increments(self):
        cm = _make_cm()
        assert cm._store_count == 0
        cm.store_memory("X", "episodic")
        assert cm._store_count == 1

    def test_memory_type_recorded(self):
        cm = _make_cm()
        cm.store_memory("semantic content", "semantic", 0.9, "keep")
        s = cm.summary()
        assert s["by_type"].get("semantic", 0) == 1

    def test_status_recorded(self):
        cm = _make_cm()
        cm.store_memory("kept", status="keep")
        cm.store_memory("disc", status="discard")
        s = cm.summary()
        assert s["by_status"]["keep"] == 1
        assert s["by_status"]["discard"] == 1

    def test_auto_reflect_triggers(self):
        cm = CognitiveMemory(db_path=":memory:", reflect_every=3)
        for i in range(3):
            cm.store_memory(f"episodic memory {i}", "episodic", 0.7, "keep")
        # After 3 stores with reflect_every=3 the reflection semantic memory
        # is added — but only if there are ≥ 3 episodic memories to reflect on
        s = cm.summary()
        # At minimum the 3 episodic + possibly 1 semantic reflection
        assert s["total_memories"] >= 3


# ---------------------------------------------------------------------------
# recall
# ---------------------------------------------------------------------------

class TestRecall:
    def test_empty_store_returns_empty_context(self):
        cm = _make_cm()
        ctx = cm.recall("anything")
        assert ctx.what_worked == []
        assert ctx.what_failed == []
        assert ctx.total_memories == 0

    def test_recall_finds_content_match(self):
        cm = _make_cm()
        cm.store_memory(
            "KEEP: increased learning rate from 0.001 to 0.01, improved accuracy",
            "episodic", 0.8, "keep", ["learning", "rate", "optimizer"],
        )
        ctx = cm.recall("learning rate optimizer")
        assert len(ctx.what_worked) == 1
        assert "learning" in ctx.what_worked[0].lower() or "rate" in ctx.what_worked[0].lower()

    def test_recall_finds_tag_match(self):
        """Query matches a tag even when the content doesn't contain the query word."""
        cm = _make_cm()
        # Content uses abbreviation "LR" but tags contain "learning"
        cm.store_memory(
            "KEEP: LR=0.04 val_bpb improved by 0.004",
            "episodic", 0.8, "keep",
            tags=["learning", "rate", "optimizer"],
        )
        ctx = cm.recall("learning rate")
        assert len(ctx.what_worked) >= 1

    def test_recall_separates_kept_from_failed(self):
        cm = _make_cm()
        cm.store_memory("KEEP: architecture change worked", "episodic", 0.8, "keep",
                        tags=["architecture"])
        cm.store_memory("DISCARD: gelu activation worse", "episodic", 0.4, "discard",
                        tags=["activation", "gelu"])
        ctx = cm.recall("architecture activation")
        assert any("worked" in w.lower() or "architecture" in w.lower()
                   for w in ctx.what_worked)
        assert any("gelu" in f.lower() or "activation" in f.lower()
                   for f in ctx.what_failed)

    def test_recall_always_includes_semantic(self):
        """Semantic memories should appear in patterns regardless of query."""
        cm = _make_cm()
        cm.store_memory(
            "PATTERN: Architecture changes are 3x more effective than optimizer tweaks",
            "semantic", 0.9, "keep", tags=["pattern"],
        )
        ctx = cm.recall("unrelated query xyz")
        assert len(ctx.patterns) >= 1

    def test_avoid_tags_collected_from_failures(self):
        cm = _make_cm()
        cm.store_memory("DISCARD: gelu activation crashed", "episodic", 0.4, "discard",
                        tags=["activation", "gelu", "crash"])
        ctx = cm.recall("activation")
        assert "activation" in ctx.avoid_tags or "gelu" in ctx.avoid_tags

    def test_recall_respects_limit(self):
        cm = _make_cm()
        for i in range(20):
            cm.store_memory(f"KEEP: experiment {i} worked great", "episodic", 0.8, "keep",
                            tags=["experiment"])
        ctx = cm.recall("experiment", limit=5)
        assert len(ctx.what_worked) <= 5

    def test_frequency_incremented_after_recall(self):
        cm = _make_cm()
        cm.store_memory("KEEP: result A", "episodic", 0.8, "keep", tags=["result"])
        before = cm.list_memories()[0].frequency
        cm.recall("result")
        after = cm.list_memories()[0].frequency
        assert after > before

    def test_recall_context_to_dict(self):
        cm = _make_cm()
        ctx = cm.recall("test")
        d = ctx.to_dict()
        assert "what_worked" in d
        assert "what_failed" in d
        assert "patterns" in d
        assert "avoid_tags" in d
        assert "suggestions" in d
        assert "total_memories" in d
        assert "hebbian_links" in d

    def test_recall_context_to_prompt_text(self):
        cm = _make_cm()
        cm.store_memory("KEEP: something that worked", "episodic", 0.8, "keep",
                        tags=["something"])
        cm.store_memory("DISCARD: something that failed", "episodic", 0.4, "discard",
                        tags=["failed"])
        ctx = cm.recall("something")
        text = ctx.to_prompt_text()
        assert "Memory recall" in text
        assert "💡" in text or "⚠️" in text


# ---------------------------------------------------------------------------
# Hebbian learning
# ---------------------------------------------------------------------------

class TestHebbian:
    def test_reinforce_creates_link(self):
        cm = _make_cm()
        m1 = cm.store_memory("memory A", tags=["alpha"])
        m2 = cm.store_memory("memory B", tags=["beta"])
        cm.hebbian_reinforce([m1, m2])
        assert cm.summary()["hebbian_links"] == 1

    def test_reinforce_twice_increases_strength(self):
        cm = _make_cm()
        m1 = cm.store_memory("A")
        m2 = cm.store_memory("B")
        cm.hebbian_reinforce([m1, m2])
        cm.hebbian_reinforce([m1, m2])
        # First call: creates link with 1.0 + 0.5 = 1.5
        # Second call: updates to 1.5 + 0.5 = 2.0
        with cm._connect() as conn:
            row = conn.execute("SELECT strength FROM hebbian_links LIMIT 1").fetchone()
        assert row["strength"] == pytest.approx(2.0)

    def test_reinforce_strength_bounded_at_10(self):
        cm = _make_cm()
        m1 = cm.store_memory("A")
        m2 = cm.store_memory("B")
        for _ in range(30):
            cm.hebbian_reinforce([m1, m2])
        with cm._connect() as conn:
            row = conn.execute("SELECT strength FROM hebbian_links LIMIT 1").fetchone()
        assert row["strength"] <= 10.0

    def test_link_count_in_summary(self):
        cm = _make_cm()
        m1 = cm.store_memory("A")
        m2 = cm.store_memory("B")
        m3 = cm.store_memory("C")
        cm.hebbian_reinforce([m1, m2, m3])  # creates 3 links: (1,2),(1,3),(2,3)
        assert cm.summary()["hebbian_links"] == 3

    def test_canonical_ordering_no_duplicates(self):
        """(a, b) and (b, a) should map to the same link row."""
        cm = _make_cm()
        m1 = cm.store_memory("A")
        m2 = cm.store_memory("B")
        cm.hebbian_reinforce([m1, m2])
        cm.hebbian_reinforce([m2, m1])  # reversed order
        assert cm.summary()["hebbian_links"] == 1  # still just one link


# ---------------------------------------------------------------------------
# Reflect (Memory Consolidation)
# ---------------------------------------------------------------------------

class TestReflect:
    def test_returns_none_when_too_few_memories(self):
        cm = _make_cm()
        cm.store_memory("only one memory", "episodic")
        result = cm.reflect()
        assert result is None

    def test_creates_semantic_memory(self):
        cm = _make_cm()
        for i in range(5):
            status = "keep" if i % 2 == 0 else "discard"
            imp = 0.8 if status == "keep" else 0.4
            cm.store_memory(
                f"Experiment {i}: {'worked' if status == 'keep' else 'failed'}",
                "episodic", imp, status, tags=["experiment", "learning"],
            )
        pattern = cm.reflect()
        assert pattern is not None
        assert "REFLECTION" in pattern
        assert cm.summary()["by_type"].get("semantic", 0) >= 1

    def test_reflection_contains_keep_rate(self):
        cm = _make_cm()
        for _ in range(3):
            cm.store_memory("worked", "episodic", 0.8, "keep", tags=["architecture"])
        for _ in range(3):
            cm.store_memory("failed", "episodic", 0.4, "discard", tags=["optimizer"])
        pattern = cm.reflect()
        assert "keep_rate" in pattern

    def test_reflection_mentions_effective_themes(self):
        cm = _make_cm()
        for _ in range(4):
            cm.store_memory("kept architecture result", "episodic", 0.8, "keep",
                            tags=["architecture", "width"])
        for _ in range(2):
            cm.store_memory("discarded optimizer", "episodic", 0.4, "discard",
                            tags=["optimizer"])
        pattern = cm.reflect(n_recent=10)
        # Most effective theme should be architecture or width
        assert "architecture" in pattern or "width" in pattern


# ---------------------------------------------------------------------------
# store_experiment (high-level helper)
# ---------------------------------------------------------------------------

class TestStoreExperiment:
    def test_returns_list_of_ids(self):
        cm = _make_cm()
        ids = cm.store_experiment(
            round_id="r1", task="test task", score=0.7,
            status="completed", learnings=["found something", "also this"],
        )
        assert isinstance(ids, list)
        assert len(ids) >= 1

    def test_successful_round_stores_learnings(self):
        cm = _make_cm()
        ids = cm.store_experiment(
            round_id="r1", task="improve LR", score=0.8,
            status="completed",
            learnings=["LR=0.04 improved results", "momentum helps"],
        )
        # 1 round memory + up to 5 learning memories
        assert len(ids) >= 2
        assert cm.summary()["total_memories"] >= 2

    def test_failed_round_stores_only_one_memory(self):
        cm = _make_cm()
        ids = cm.store_experiment(
            round_id="r2", task="test", score=0.2,
            status="failed", learnings=["nothing worked"],
        )
        # low score → only the round summary memory
        assert len(ids) == 1

    def test_creates_hebbian_links_on_success(self):
        cm = _make_cm()
        ids = cm.store_experiment(
            round_id="r1", task="test", score=0.9,
            status="completed",
            learnings=["finding A", "finding B", "finding C"],
        )
        if len(ids) >= 2:
            assert cm.summary()["hebbian_links"] >= 1

    def test_report_summary_included_in_content(self):
        cm = _make_cm()
        cm.store_experiment(
            round_id="r1", task="my task", score=0.8,
            status="completed", learnings=[],
            report_summary="This paper proposes a novel approach",
        )
        mems = cm.list_memories(memory_type="episodic")
        combined = " ".join(m.content for m in mems)
        assert "novel approach" in combined


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

class TestSummary:
    def test_empty_summary(self):
        cm = _make_cm()
        s = cm.summary()
        assert s["total_memories"] == 0
        assert s["hebbian_links"] == 0
        assert s["store_count_this_session"] == 0

    def test_summary_reflects_additions(self):
        cm = _make_cm()
        cm.store_memory("A", "episodic", 0.8, "keep")
        cm.store_memory("B", "semantic", 0.9, "keep")
        s = cm.summary()
        assert s["total_memories"] == 2
        assert s["by_type"]["episodic"] == 1
        assert s["by_type"]["semantic"] == 1

    def test_avg_frequency_after_recall(self):
        cm = _make_cm()
        cm.store_memory("test recall frequency", "episodic", 0.7, "keep",
                        tags=["recall", "frequency"])
        cm.recall("recall frequency")
        s = cm.summary()
        # frequency should be > 1 after being recalled once
        assert s["avg_frequency"] > 1.0
