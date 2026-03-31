"""
TavilyPool — multi-account Tavily client with automatic key rotation and fallback.

Features:
  - Loads API keys from environment variables AND config file
  - Automatically rotates to the next key when one is exhausted, rate-limited, or invalid
  - Per-key state machine: ACTIVE → RATE_LIMITED (temporary) → EXHAUSTED (semi-permanent) → INVALID (permanent)
  - Exponential back-off on rate limits before marking a key as exhausted
  - Supports all Tavily operations: search, extract, research
  - Specialised helpers: search_papers (arxiv.org filter) and search_github (github.com filter)
  - Thread-safe via a single lock on the key-state map

Environment variables read (in priority order):
    TAVILY_API_KEY            — primary key
    TAVILY_API_KEY_2          — second key
    TAVILY_API_KEY_3          — third key
    ...
    TAVILY_API_KEY_N          — Nth key (up to 20)

Config alternative (config/config.yaml → tavily.api_keys list):
    tavily:
      api_keys:
        - tvly-key1
        - tvly-key2

Usage:
    pool = TavilyPool.from_env()
    results = pool.search("transformer attention efficiency arxiv 2026")
    papers  = pool.search_papers("self-evolving agents", max_results=15)
    repos   = pool.search_github("GEPA genetic pareto prompt evolution")
    content = pool.extract(["https://arxiv.org/abs/2507.19457"])
"""

from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

logger = logging.getLogger(__name__)

# ── HTTP status codes that drive key-state transitions ───────────────── #
_STATUS_INVALID = {401, 403}          # bad key → INVALID (never retry)
_STATUS_EXHAUSTED = {402, 429}        # credits gone or hard quota → EXHAUSTED
_RETRY_AFTER_DEFAULT = 60             # seconds to wait before retrying a rate-limited key
_EXHAUSTED_RESET_HOURS = 24           # hours before an EXHAUSTED key is retried


class KeyState(Enum):
    ACTIVE = auto()
    RATE_LIMITED = auto()   # temporary — retry after back-off
    EXHAUSTED = auto()      # credits gone — retry after 24 h
    INVALID = auto()        # bad key — never retry


@dataclass
class _KeyRecord:
    api_key: str
    state: KeyState = KeyState.ACTIVE
    fail_count: int = 0
    retry_after: float = 0.0        # epoch time after which we may retry
    requests_made: int = 0
    last_error: str = ""


class TavilyPool:
    """
    Thread-safe pool of Tavily API keys with automatic rotation and fallback.

    Rotation order: always try ACTIVE keys first.
    RATE_LIMITED keys are retried after their back-off window.
    EXHAUSTED keys are retried after 24 hours.
    INVALID keys are never retried.

    Raises:
        TavilyPoolExhausted — when ALL keys are either invalid or exhausted
                              and there are no more fallbacks.
    """

    def __init__(self, api_keys: list[str]) -> None:
        if not api_keys:
            raise ValueError("TavilyPool requires at least one API key")
        self._records: list[_KeyRecord] = [_KeyRecord(k.strip()) for k in api_keys if k.strip()]
        self._lock = threading.Lock()
        logger.info("TavilyPool initialised with %d key(s)", len(self._records))

    # ------------------------------------------------------------------ #
    #  Factory constructors                                                #
    # ------------------------------------------------------------------ #

    @classmethod
    def from_env(cls) -> "TavilyPool":
        """
        Build pool from environment variables.

        Reads TAVILY_API_KEY, TAVILY_API_KEY_2 … TAVILY_API_KEY_20 in order.
        """
        keys: list[str] = []
        primary = os.environ.get("TAVILY_API_KEY", "")
        if primary:
            keys.append(primary)
        for i in range(2, 21):
            k = os.environ.get(f"TAVILY_API_KEY_{i}", "")
            if k:
                keys.append(k)
        if not keys:
            raise TavilyPoolExhausted(
                "No Tavily API keys found. Set TAVILY_API_KEY (and optionally "
                "TAVILY_API_KEY_2, TAVILY_API_KEY_3 …) in your environment."
            )
        return cls(keys)

    @classmethod
    def from_list(cls, keys: list[str]) -> "TavilyPool":
        """Build pool from an explicit list of keys."""
        return cls(keys)

    @classmethod
    def from_env_or_list(cls, keys: list[str] | None = None) -> "TavilyPool":
        """
        Try env vars first; fall back to a provided list; raise if neither works.
        Useful when keys come from config.yaml.
        """
        try:
            pool = cls.from_env()
            # Merge in any extra keys from config that aren't already in the pool
            if keys:
                existing = {r.api_key for r in pool._records}
                extras = [k for k in keys if k.strip() and k.strip() not in existing]
                pool._records.extend(_KeyRecord(k.strip()) for k in extras)
                if extras:
                    logger.info("TavilyPool: added %d extra key(s) from config", len(extras))
            return pool
        except TavilyPoolExhausted:
            if keys:
                return cls.from_list(keys)
            raise

    # ------------------------------------------------------------------ #
    #  Public search API                                                    #
    # ------------------------------------------------------------------ #

    def search(
        self,
        query: str,
        max_results: int = 10,
        search_depth: str = "advanced",
        include_domains: list[str] | None = None,
        exclude_domains: list[str] | None = None,
        include_answer: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        General web search via Tavily.

        Returns the raw Tavily response dict:
          {query, answer, results: [{title, url, content, score}, ...]}
        """
        call_kwargs: dict[str, Any] = {
            "max_results": max_results,
            "search_depth": search_depth,
            "include_answer": include_answer,
        }
        if include_domains:
            call_kwargs["include_domains"] = include_domains
        if exclude_domains:
            call_kwargs["exclude_domains"] = exclude_domains
        call_kwargs.update(kwargs)

        return self._call_with_rotation("search", query, **call_kwargs)

    def search_papers(
        self,
        query: str,
        max_results: int = 15,
        year_from: int | None = None,
    ) -> dict[str, Any]:
        """
        Search for research papers on arXiv and Semantic Scholar via Tavily.

        Filters to arxiv.org, semanticscholar.org, paperswithcode.com.
        Automatically appends 'site:arxiv.org' style boosting to the query.
        """
        # Boost recency if year requested
        boosted_query = query
        if year_from:
            boosted_query = f"{query} after:{year_from}"

        return self.search(
            query=boosted_query,
            max_results=max_results,
            search_depth="advanced",
            include_domains=[
                "arxiv.org",
                "semanticscholar.org",
                "paperswithcode.com",
                "openreview.net",
                "proceedings.mlr.press",
                "proceedings.neurips.cc",
            ],
            include_answer=False,
        )

    def search_github(
        self,
        query: str,
        max_results: int = 10,
    ) -> dict[str, Any]:
        """
        Search GitHub repositories and README files via Tavily.

        Filters to github.com only.
        """
        return self.search(
            query=query,
            max_results=max_results,
            search_depth="advanced",
            include_domains=["github.com"],
            include_answer=False,
        )

    def extract(
        self,
        urls: list[str],
        include_images: bool = False,
    ) -> dict[str, Any]:
        """
        Extract raw content from specific URLs (paper PDFs, GitHub READMEs, etc.).
        Up to 20 URLs per call.
        """
        return self._call_with_rotation("extract", urls=urls[:20], include_images=include_images)

    def qna_search(self, query: str) -> str:
        """Get a concise answer to a factual question."""
        result = self._call_with_rotation("qna_search", query)
        return str(result)

    # ------------------------------------------------------------------ #
    #  Key-rotation engine                                                  #
    # ------------------------------------------------------------------ #

    def _call_with_rotation(self, method: str, *args: Any, **kwargs: Any) -> Any:
        """
        Execute a Tavily API call, rotating through keys on failure.

        Tries every usable key in order. On exhaustion of all keys, raises
        TavilyPoolExhausted.
        """
        last_exc: Exception | None = None

        for attempt in range(len(self._records) * 2):   # allow each key up to 2 shots
            record = self._pick_key()
            if record is None:
                raise TavilyPoolExhausted(
                    f"All {len(self._records)} Tavily key(s) are exhausted or invalid. "
                    "Add more keys via TAVILY_API_KEY_2, TAVILY_API_KEY_3 …"
                )

            try:
                client = self._make_client(record.api_key)
                fn = getattr(client, method)
                result = fn(*args, **kwargs)
                with self._lock:
                    record.requests_made += 1
                    record.fail_count = 0   # reset on success
                logger.debug(
                    "Tavily %s succeeded (key index %d, request #%d)",
                    method, self._records.index(record), record.requests_made,
                )
                return result

            except Exception as exc:
                last_exc = exc
                self._handle_error(record, exc)
                logger.info(
                    "Tavily key %d failed [%s]: %s — rotating",
                    self._records.index(record), type(exc).__name__, str(exc)[:80],
                )

        raise TavilyPoolExhausted(
            f"All Tavily keys failed after rotation. Last error: {last_exc}"
        ) from last_exc

    def _pick_key(self) -> _KeyRecord | None:
        """
        Pick the best available key (ACTIVE preferred, then RATE_LIMITED if back-off elapsed).
        Returns None if no usable key exists.
        """
        now = time.time()
        with self._lock:
            # 1. Prefer ACTIVE keys
            active = [r for r in self._records if r.state == KeyState.ACTIVE]
            if active:
                return active[0]

            # 2. RATE_LIMITED keys whose back-off has expired → re-activate
            for r in self._records:
                if r.state == KeyState.RATE_LIMITED and now >= r.retry_after:
                    logger.info(
                        "TavilyPool: reactivating rate-limited key index %d",
                        self._records.index(r),
                    )
                    r.state = KeyState.ACTIVE
                    return r

            # 3. EXHAUSTED keys whose 24-h window has expired → re-activate
            for r in self._records:
                if r.state == KeyState.EXHAUSTED and now >= r.retry_after:
                    logger.info(
                        "TavilyPool: retrying exhausted key index %d after 24h",
                        self._records.index(r),
                    )
                    r.state = KeyState.ACTIVE
                    return r

            return None   # all INVALID or windows not yet elapsed

    def _handle_error(self, record: _KeyRecord, exc: Exception) -> None:
        """Update key state based on the exception raised by Tavily."""
        with self._lock:
            record.fail_count += 1
            record.last_error = str(exc)[:200]

            status = self._extract_http_status(exc)

            if status in _STATUS_INVALID:
                record.state = KeyState.INVALID
                logger.warning(
                    "TavilyPool: key index %d marked INVALID (HTTP %s)",
                    self._records.index(record), status,
                )

            elif status in _STATUS_EXHAUSTED:
                if status == 429:
                    # Rate-limited: exponential back-off, cap at 10 min
                    backoff = min(
                        _RETRY_AFTER_DEFAULT * (2 ** min(record.fail_count - 1, 4)),
                        600,
                    )
                    record.state = KeyState.RATE_LIMITED
                    record.retry_after = time.time() + backoff
                    logger.info(
                        "TavilyPool: key index %d RATE_LIMITED, retry in %.0fs",
                        self._records.index(record), backoff,
                    )
                else:
                    # 402: credits exhausted — wait 24h before retry
                    record.state = KeyState.EXHAUSTED
                    record.retry_after = time.time() + _EXHAUSTED_RESET_HOURS * 3600
                    logger.warning(
                        "TavilyPool: key index %d EXHAUSTED (HTTP 402), retry in 24h",
                        self._records.index(record),
                    )

            else:
                # Unknown / transient error — keep ACTIVE but bump fail count
                if record.fail_count >= 3:
                    # After 3 consecutive unknown failures, treat as rate-limited
                    record.state = KeyState.RATE_LIMITED
                    record.retry_after = time.time() + _RETRY_AFTER_DEFAULT
                    logger.info(
                        "TavilyPool: key index %d moved to RATE_LIMITED after %d failures",
                        self._records.index(record), record.fail_count,
                    )

    @staticmethod
    def _extract_http_status(exc: Exception) -> int | None:
        """Try to extract an HTTP status code from a Tavily/requests exception."""
        # tavily-python raises requests.HTTPError with a response attribute
        resp = getattr(exc, "response", None)
        if resp is not None:
            status = getattr(resp, "status_code", None)
            if isinstance(status, int):
                return status
        # Some wrappers store it directly
        for attr in ("status_code", "code", "status"):
            val = getattr(exc, attr, None)
            if isinstance(val, int):
                return val
        # Try to parse from the string representation e.g. "429 Too Many Requests"
        import re
        match = re.search(r"\b(4\d{2}|5\d{2})\b", str(exc))
        if match:
            return int(match.group(1))
        return None

    @staticmethod
    def _make_client(api_key: str) -> Any:
        """Lazily import and construct a TavilyClient."""
        try:
            from tavily import TavilyClient
        except ImportError as exc:
            raise ImportError(
                "tavily-python is not installed. Run: pip install tavily-python"
            ) from exc
        return TavilyClient(api_key=api_key)

    # ------------------------------------------------------------------ #
    #  Status / diagnostics                                                 #
    # ------------------------------------------------------------------ #

    def status(self) -> list[dict[str, Any]]:
        """Return a human-readable status snapshot of all keys (masks key values)."""
        now = time.time()
        with self._lock:
            return [
                {
                    "index": i,
                    "key_preview": f"{r.api_key[:8]}…" if len(r.api_key) > 8 else "***",
                    "state": r.state.name,
                    "requests_made": r.requests_made,
                    "fail_count": r.fail_count,
                    "retry_in_secs": max(0, round(r.retry_after - now)) if r.retry_after else 0,
                    "last_error": r.last_error[:60] if r.last_error else "",
                }
                for i, r in enumerate(self._records)
            ]

    def active_key_count(self) -> int:
        """Return the number of currently usable keys."""
        now = time.time()
        with self._lock:
            return sum(
                1 for r in self._records
                if r.state == KeyState.ACTIVE
                or (r.state in (KeyState.RATE_LIMITED, KeyState.EXHAUSTED) and now >= r.retry_after)
            )

    def reset_all(self) -> None:
        """Reset all keys to ACTIVE state (for testing / manual recovery)."""
        with self._lock:
            for r in self._records:
                if r.state != KeyState.INVALID:
                    r.state = KeyState.ACTIVE
                    r.retry_after = 0.0
                    r.fail_count = 0


class TavilyPoolExhausted(RuntimeError):
    """Raised when all keys in the pool are unavailable."""
