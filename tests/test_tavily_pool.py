"""
Tests for TavilyPool multi-key rotation and fallback logic.
No real API calls — all Tavily clients are mocked.
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest

from tools.tavily_search import (
    KeyState,
    TavilyPool,
    TavilyPoolExhausted,
    _KeyRecord,
    _EXHAUSTED_RESET_HOURS,
    _RETRY_AFTER_DEFAULT,
)


# ── Helpers ───────────────────────────────────────────────────────────── #

def _pool(*keys: str) -> TavilyPool:
    return TavilyPool(list(keys))


def _make_error(status: int) -> Exception:
    """Fake an HTTP error with a given status code."""
    err = Exception(f"{status} Error")
    resp = MagicMock()
    resp.status_code = status
    err.response = resp
    return err


# ── Construction ──────────────────────────────────────────────────────── #

class TestConstruction:
    def test_requires_at_least_one_key(self):
        with pytest.raises(ValueError):
            TavilyPool([])

    def test_strips_whitespace_from_keys(self):
        pool = _pool("  tvly-abc  ", " tvly-def ")
        assert pool._records[0].api_key == "tvly-abc"
        assert pool._records[1].api_key == "tvly-def"

    def test_ignores_blank_keys(self):
        pool = TavilyPool(["tvly-a", "", "   ", "tvly-b"])
        assert len(pool._records) == 2

    def test_from_list(self):
        pool = TavilyPool.from_list(["tvly-x", "tvly-y"])
        assert len(pool._records) == 2


# ── from_env ──────────────────────────────────────────────────────────── #

class TestFromEnv:
    def test_reads_primary_key(self, monkeypatch):
        monkeypatch.setenv("TAVILY_API_KEY", "tvly-primary")
        pool = TavilyPool.from_env()
        assert pool._records[0].api_key == "tvly-primary"

    def test_reads_numbered_keys(self, monkeypatch):
        monkeypatch.setenv("TAVILY_API_KEY", "tvly-1")
        monkeypatch.setenv("TAVILY_API_KEY_2", "tvly-2")
        monkeypatch.setenv("TAVILY_API_KEY_3", "tvly-3")
        pool = TavilyPool.from_env()
        keys = [r.api_key for r in pool._records]
        assert keys == ["tvly-1", "tvly-2", "tvly-3"]

    def test_raises_when_no_env_keys(self, monkeypatch):
        for i in range(1, 21):
            name = "TAVILY_API_KEY" if i == 1 else f"TAVILY_API_KEY_{i}"
            monkeypatch.delenv(name, raising=False)
        with pytest.raises(TavilyPoolExhausted):
            TavilyPool.from_env()

    def test_from_env_or_list_falls_back_to_list(self, monkeypatch):
        for i in range(1, 21):
            name = "TAVILY_API_KEY" if i == 1 else f"TAVILY_API_KEY_{i}"
            monkeypatch.delenv(name, raising=False)
        pool = TavilyPool.from_env_or_list(["tvly-config-key"])
        assert pool._records[0].api_key == "tvly-config-key"

    def test_from_env_or_list_merges_extra_config_keys(self, monkeypatch):
        monkeypatch.setenv("TAVILY_API_KEY", "tvly-env")
        pool = TavilyPool.from_env_or_list(["tvly-env", "tvly-extra"])
        # tvly-env deduped, tvly-extra added
        keys = [r.api_key for r in pool._records]
        assert "tvly-env" in keys
        assert "tvly-extra" in keys


# ── Key-state transitions ─────────────────────────────────────────────── #

class TestKeyStateTransitions:
    def test_401_marks_invalid(self):
        pool = _pool("tvly-a")
        pool._handle_error(pool._records[0], _make_error(401))
        assert pool._records[0].state == KeyState.INVALID

    def test_403_marks_invalid(self):
        pool = _pool("tvly-a")
        pool._handle_error(pool._records[0], _make_error(403))
        assert pool._records[0].state == KeyState.INVALID

    def test_402_marks_exhausted_with_24h_window(self):
        pool = _pool("tvly-a")
        before = time.time()
        pool._handle_error(pool._records[0], _make_error(402))
        assert pool._records[0].state == KeyState.EXHAUSTED
        expected_min = before + _EXHAUSTED_RESET_HOURS * 3600 - 5
        assert pool._records[0].retry_after >= expected_min

    def test_429_marks_rate_limited_with_backoff(self):
        pool = _pool("tvly-a")
        pool._handle_error(pool._records[0], _make_error(429))
        assert pool._records[0].state == KeyState.RATE_LIMITED
        assert pool._records[0].retry_after > time.time()

    def test_429_exponential_backoff(self):
        pool = _pool("tvly-a")
        record = pool._records[0]
        backoffs = []
        for _ in range(5):
            before = time.time()
            pool._handle_error(record, _make_error(429))
            backoffs.append(record.retry_after - before)
            # Manually reset to ACTIVE so next handle_error works
            record.state = KeyState.ACTIVE

        # Each backoff should be >= the previous (capped at 600s)
        for i in range(1, len(backoffs)):
            assert backoffs[i] >= backoffs[i - 1] or backoffs[i] >= 590

    def test_429_backoff_capped_at_600s(self):
        pool = _pool("tvly-a")
        record = pool._records[0]
        record.fail_count = 100   # simulate many failures
        before = time.time()
        pool._handle_error(record, _make_error(429))
        assert record.retry_after - before <= 605   # 600s cap + small tolerance

    def test_unknown_error_becomes_rate_limited_after_3_failures(self):
        pool = _pool("tvly-a")
        record = pool._records[0]
        for _ in range(3):
            pool._handle_error(record, Exception("unknown error"))
        assert record.state == KeyState.RATE_LIMITED

    def test_unknown_error_stays_active_for_first_2_failures(self):
        pool = _pool("tvly-a")
        record = pool._records[0]
        for _ in range(2):
            pool._handle_error(record, Exception("unknown"))
        assert record.state == KeyState.ACTIVE


# ── Key picking ───────────────────────────────────────────────────────── #

class TestPickKey:
    def test_picks_active_key(self):
        pool = _pool("tvly-a", "tvly-b")
        key = pool._pick_key()
        assert key is not None
        assert key.state == KeyState.ACTIVE

    def test_skips_invalid_key(self):
        pool = _pool("tvly-a", "tvly-b")
        pool._records[0].state = KeyState.INVALID
        key = pool._pick_key()
        assert key.api_key == "tvly-b"

    def test_returns_none_when_all_invalid(self):
        pool = _pool("tvly-a", "tvly-b")
        for r in pool._records:
            r.state = KeyState.INVALID
        assert pool._pick_key() is None

    def test_reactivates_rate_limited_after_backoff(self):
        pool = _pool("tvly-a")
        pool._records[0].state = KeyState.RATE_LIMITED
        pool._records[0].retry_after = time.time() - 1  # already elapsed
        key = pool._pick_key()
        assert key is not None
        assert key.state == KeyState.ACTIVE

    def test_does_not_reactivate_rate_limited_before_backoff(self):
        pool = _pool("tvly-a")
        pool._records[0].state = KeyState.RATE_LIMITED
        pool._records[0].retry_after = time.time() + 9999
        assert pool._pick_key() is None

    def test_reactivates_exhausted_after_24h(self):
        pool = _pool("tvly-a")
        pool._records[0].state = KeyState.EXHAUSTED
        pool._records[0].retry_after = time.time() - 1
        key = pool._pick_key()
        assert key is not None

    def test_active_key_count(self):
        pool = _pool("tvly-a", "tvly-b", "tvly-c")
        pool._records[0].state = KeyState.INVALID
        pool._records[1].state = KeyState.RATE_LIMITED
        pool._records[1].retry_after = time.time() + 9999
        assert pool.active_key_count() == 1   # only tvly-c is usable right now


# ── Rotation on failure ───────────────────────────────────────────────── #

class TestRotation:
    def _mock_client_factory(self, behaviors: list):
        """
        behaviors: list of callables or exceptions to raise in sequence.
        Each element corresponds to one TavilyClient instantiation.
        """
        call_iter = iter(behaviors)

        def factory(api_key):
            behavior = next(call_iter)
            mock = MagicMock()
            if isinstance(behavior, Exception):
                mock.search.side_effect = behavior
            else:
                mock.search.return_value = behavior
            return mock

        return factory

    def test_succeeds_on_first_key(self):
        pool = _pool("tvly-a")
        expected = {"results": [{"title": "Paper 1", "url": "http://x", "content": "abstract"}]}
        with patch.object(pool, "_make_client", return_value=MagicMock(search=MagicMock(return_value=expected))):
            result = pool.search("test query")
        assert result == expected

    def test_rotates_to_second_key_on_429(self):
        pool = _pool("tvly-a", "tvly-b")
        call_count = {"n": 0}
        expected = {"results": []}

        def factory(api_key):
            call_count["n"] += 1
            mock = MagicMock()
            if api_key == "tvly-a":
                mock.search.side_effect = _make_error(429)
            else:
                mock.search.return_value = expected
            return mock

        with patch.object(pool, "_make_client", side_effect=factory):
            result = pool.search("test")

        assert result == expected
        assert call_count["n"] == 2  # tried both keys

    def test_raises_pool_exhausted_when_all_keys_invalid(self):
        pool = _pool("tvly-a", "tvly-b")

        def factory(api_key):
            mock = MagicMock()
            mock.search.side_effect = _make_error(401)
            return mock

        with patch.object(pool, "_make_client", side_effect=factory):
            with pytest.raises(TavilyPoolExhausted):
                pool.search("test")

        assert all(r.state == KeyState.INVALID for r in pool._records)

    def test_rotates_through_3_keys(self):
        pool = _pool("tvly-a", "tvly-b", "tvly-c")
        expected = {"results": [{"title": "ok"}]}
        call_order = []

        def factory(api_key):
            call_order.append(api_key)
            mock = MagicMock()
            if api_key in ("tvly-a", "tvly-b"):
                mock.search.side_effect = _make_error(429)
            else:
                mock.search.return_value = expected
            return mock

        with patch.object(pool, "_make_client", side_effect=factory):
            result = pool.search("test")

        assert result == expected
        assert call_order == ["tvly-a", "tvly-b", "tvly-c"]


# ── Status / diagnostics ──────────────────────────────────────────────── #

class TestStatus:
    def test_status_masks_key_values(self):
        pool = _pool("tvly-secret-key-abc123", "short")
        statuses = pool.status()
        for s in statuses:
            assert "secret" not in s["key_preview"]
            assert s["key_preview"].endswith("…") or s["key_preview"] == "***"

    def test_status_shows_correct_states(self):
        pool = _pool("tvly-a", "tvly-b", "tvly-c")
        pool._records[0].state = KeyState.INVALID
        pool._records[1].state = KeyState.EXHAUSTED
        statuses = pool.status()
        states = {s["index"]: s["state"] for s in statuses}
        assert states[0] == "INVALID"
        assert states[1] == "EXHAUSTED"
        assert states[2] == "ACTIVE"

    def test_reset_all_reactivates_non_invalid(self):
        pool = _pool("tvly-a", "tvly-b", "tvly-c")
        pool._records[0].state = KeyState.EXHAUSTED
        pool._records[1].state = KeyState.RATE_LIMITED
        pool._records[2].state = KeyState.INVALID
        pool.reset_all()
        assert pool._records[0].state == KeyState.ACTIVE
        assert pool._records[1].state == KeyState.ACTIVE
        assert pool._records[2].state == KeyState.INVALID  # INVALID not reset


# ── HTTP status parsing ───────────────────────────────────────────────── #

class TestExtractHttpStatus:
    def test_from_response_attribute(self):
        err = Exception("error")
        resp = MagicMock()
        resp.status_code = 429
        err.response = resp
        assert TavilyPool._extract_http_status(err) == 429

    def test_from_status_code_attribute(self):
        err = Exception("error")
        err.status_code = 402
        assert TavilyPool._extract_http_status(err) == 402

    def test_from_string_representation(self):
        err = Exception("429 Too Many Requests")
        assert TavilyPool._extract_http_status(err) == 429

    def test_returns_none_for_unknown(self):
        err = Exception("connection refused")
        assert TavilyPool._extract_http_status(err) is None
