"""
GitHub Explorer — task-aware repo discovery and "starting point" integration.

This module is the implementation of the core architectural principle:

    "Surpass current auto paper generation by first utilizing existing work —
     find the best GitHub repo closest to your goal, clone it, understand its
     architecture, and extend it."

Flow
----
1. **Discover**: given a research task / hypothesis, query GitHub via Tavily
   (and optionally the GitHub API) for repos that implement related work.
2. **Rank**: score candidates by recency, stars, language match, test
   coverage signal, and semantic relevance to the hypothesis.
3. **Analyse**: for top candidates, read README + key source files to
   extract: purpose, architecture, entry points, evaluation scripts.
4. **Integrate**: register the best repo as the ``starting_point`` on the
   Blackboard so ExperimentAgent clones it and extends it rather than
   starting from scratch.

References
----------
- karpathy/autoresearch "start from existing work" design note (March 2026)
- hermes-agent ToolDiscovery pipeline pattern
- mini-SWE-agent bash-only navigator pattern
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ────────────────────────────────────────────────────────────────────────── #
#  Data model                                                                 #
# ────────────────────────────────────────────────────────────────────────── #

@dataclass
class RepoCandidate:
    """A discovered GitHub repository that may serve as a starting point."""

    name: str                         # "owner/repo"
    url: str
    description: str = ""
    stars: int = 0
    last_updated: str = ""            # ISO-8601 date
    language: str = ""
    has_tests: bool = False
    has_eval_script: bool = False
    readme_summary: str = ""
    entry_points: list[str] = field(default_factory=list)
    eval_scripts: list[str] = field(default_factory=list)
    relevance_score: float = 0.0      # 0–1, computed by rank()
    overall_score: float = 0.0        # composite rank score

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "url": self.url,
            "description": self.description,
            "stars": self.stars,
            "last_updated": self.last_updated,
            "language": self.language,
            "has_tests": self.has_tests,
            "has_eval_script": self.has_eval_script,
            "readme_summary": self.readme_summary[:400],
            "entry_points": self.entry_points,
            "eval_scripts": self.eval_scripts,
            "relevance_score": self.relevance_score,
            "overall_score": self.overall_score,
        }


# ────────────────────────────────────────────────────────────────────────── #
#  GitHubExplorer                                                             #
# ────────────────────────────────────────────────────────────────────────── #

class GitHubExplorer:
    """
    Discovers, ranks, and analyses GitHub repos for a research task.

    Configuration (``tools.github_explorer`` in config.yaml):

    =================  ==================================================
    Key                Meaning
    =================  ==================================================
    top_k              number of repos to deeply analyse (default: 3)
    min_stars          minimum star count to consider (default: 50)
    prefer_language    preferred language (default: "python")
    clone_dir          local directory for shallow clones
    github_token_env   env var holding a GitHub PAT (for higher rate limit)
    =================  ==================================================
    """

    def __init__(
        self,
        *,
        top_k: int = 3,
        min_stars: int = 50,
        prefer_language: str = "python",
        clone_dir: str = "workspaces/github_explorer",
        github_token_env: str = "GITHUB_TOKEN",
        tavily_pool: Any = None,           # tools.tavily_search.TavilyPool
    ) -> None:
        self._top_k = top_k
        self._min_stars = min_stars
        self._prefer_language = prefer_language.lower()
        self._clone_dir = Path(clone_dir)
        self._clone_dir.mkdir(parents=True, exist_ok=True)
        self._github_token_env = github_token_env
        self._tavily = tavily_pool

    @classmethod
    def from_config(cls, cfg: dict[str, Any], tavily_pool: Any = None) -> "GitHubExplorer":
        ge = cfg.get("tools", {}).get("github_explorer", {})
        return cls(
            top_k=ge.get("top_k", 3),
            min_stars=ge.get("min_stars", 50),
            prefer_language=ge.get("prefer_language", "python"),
            clone_dir=ge.get("clone_dir", "workspaces/github_explorer"),
            github_token_env=ge.get("github_token_env", "GITHUB_TOKEN"),
            tavily_pool=tavily_pool,
        )

    # ------------------------------------------------------------------ #
    #  Main interface                                                      #
    # ------------------------------------------------------------------ #

    def find_starting_points(
        self,
        task: str,
        hypothesis: str = "",
        keywords: list[str] | None = None,
        year_from: int = 2023,
    ) -> list[RepoCandidate]:
        """
        Find and rank the best GitHub repos as starting points for *task*.

        Returns a list of :class:`RepoCandidate` objects sorted by
        ``overall_score`` descending.  The top result is typically the best
        starting point for :class:`~agents.experiment_agent.ExperimentAgent`.
        """
        query_text = self._build_query(task, hypothesis, keywords)
        logger.info("GitHubExplorer: searching for '%s'", query_text[:80])

        raw_results = self._search(query_text, year_from)
        if not raw_results:
            logger.warning("GitHubExplorer: no results for query '%s'", query_text[:80])
            return []

        candidates = self._parse_results(raw_results)
        candidates = [c for c in candidates if c.stars >= self._min_stars]

        # Rank and deeply analyse top-k
        self._compute_scores(candidates, task, hypothesis)
        candidates.sort(key=lambda c: c.overall_score, reverse=True)

        for c in candidates[: self._top_k]:
            self._analyse_repo(c)

        logger.info(
            "GitHubExplorer: top candidates: %s",
            [c.name for c in candidates[: self._top_k]],
        )
        return candidates[: self._top_k]

    def best_starting_point(
        self, task: str, hypothesis: str = ""
    ) -> RepoCandidate | None:
        """Return the single best repo or None if nothing suitable found."""
        results = self.find_starting_points(task, hypothesis)
        return results[0] if results else None

    # ------------------------------------------------------------------ #
    #  Search backends                                                    #
    # ------------------------------------------------------------------ #

    def _build_query(
        self, task: str, hypothesis: str, keywords: list[str] | None
    ) -> str:
        parts = [task[:120]]
        if hypothesis:
            parts.append(hypothesis[:80])
        if keywords:
            parts.extend(keywords[:3])
        parts.append(f"language:{self._prefer_language}")
        parts.append("site:github.com")
        return " ".join(parts)

    def _search(self, query: str, year_from: int) -> list[dict[str, Any]]:
        """Search GitHub repos via Tavily, fall back to GitHub API."""
        results: list[dict[str, Any]] = []

        if self._tavily is not None:
            try:
                tavily_results = self._tavily.search_github(
                    query, max_results=20, year_from=year_from
                )
                results.extend(tavily_results)
                logger.debug("GitHubExplorer: Tavily returned %d results", len(tavily_results))
            except Exception as exc:
                logger.warning("GitHubExplorer: Tavily search failed (%s)", exc)

        if not results:
            results = self._github_api_search(query)

        return results

    def _github_api_search(self, query: str) -> list[dict[str, Any]]:
        """
        Direct GitHub search API call (no external library required).
        Returns a list of raw result dicts with keys: name, url, description,
        stars, last_updated, language.
        """
        import os
        import urllib.request
        import urllib.parse
        import urllib.error

        # Strip site: modifier for GitHub API
        api_query = re.sub(r"site:\S+", "", query).strip()
        encoded = urllib.parse.quote(
            api_query + " language:" + self._prefer_language
        )
        url = (
            f"https://api.github.com/search/repositories"
            f"?q={encoded}&sort=stars&order=desc&per_page=20"
        )

        token = os.environ.get(self._github_token_env, "")
        headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "Autoresearch-v2",
        }
        if token:
            headers["Authorization"] = f"Bearer {token}"

        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode())
            items = data.get("items", [])
            return [
                {
                    "name": item["full_name"],
                    "url": item["html_url"],
                    "description": item.get("description") or "",
                    "stars": item.get("stargazers_count", 0),
                    "last_updated": item.get("updated_at", ""),
                    "language": (item.get("language") or "").lower(),
                }
                for item in items
            ]
        except Exception as exc:
            logger.warning("GitHubExplorer: GitHub API search failed (%s)", exc)
            return []

    # ------------------------------------------------------------------ #
    #  Parsing                                                            #
    # ------------------------------------------------------------------ #

    def _parse_results(self, raw: list[dict[str, Any]]) -> list[RepoCandidate]:
        """Convert raw search results to RepoCandidate objects."""
        seen: set[str] = set()
        candidates: list[RepoCandidate] = []
        for r in raw:
            name = r.get("name", "")
            url = r.get("url", r.get("href", ""))
            if not name or name in seen:
                # Attempt to derive name from url
                if url and "github.com/" in url:
                    parts = url.split("github.com/")[-1].strip("/").split("/")[:2]
                    if len(parts) == 2:
                        name = "/".join(parts)
                    else:
                        continue
                else:
                    continue
            seen.add(name)
            if not url:
                url = f"https://github.com/{name}"
            candidates.append(RepoCandidate(
                name=name,
                url=url,
                description=r.get("description", r.get("content", ""))[:300],
                stars=int(r.get("stars", r.get("stargazers_count", 0))),
                last_updated=r.get("last_updated", r.get("updated_at", "")),
                language=(r.get("language") or "").lower(),
            ))
        return candidates

    # ------------------------------------------------------------------ #
    #  Scoring                                                            #
    # ------------------------------------------------------------------ #

    def _compute_scores(
        self,
        candidates: list[RepoCandidate],
        task: str,
        hypothesis: str,
    ) -> None:
        """
        Score each candidate on four axes and combine into ``overall_score``.

        Axes (weights):
          - Relevance  (0.40) — keyword overlap with task + hypothesis
          - Recency    (0.25) — repos updated in last 12 months score higher
          - Stars      (0.20) — log-scaled star count
          - Language   (0.15) — bonus for preferred language
        """
        task_words = set(re.findall(r"\w+", (task + " " + hypothesis).lower()))
        max_stars = max((c.stars for c in candidates), default=1)

        for c in candidates:
            # Relevance
            desc_words = set(re.findall(r"\w+", (c.description + " " + c.name).lower()))
            overlap = len(task_words & desc_words) / max(len(task_words), 1)
            relevance = min(1.0, overlap * 3)

            # Recency
            recency = 0.5
            if c.last_updated:
                try:
                    import datetime
                    updated = datetime.datetime.fromisoformat(
                        c.last_updated.replace("Z", "+00:00")
                    )
                    now = datetime.datetime.now(datetime.timezone.utc)
                    age_days = (now - updated).days
                    recency = max(0.0, 1.0 - age_days / 730)  # decay over 2 years
                except ValueError:
                    pass

            # Stars (log-scale)
            import math
            stars_score = math.log1p(c.stars) / math.log1p(max_stars) if max_stars > 0 else 0

            # Language bonus
            lang_score = 1.0 if c.language == self._prefer_language else 0.3

            c.relevance_score = relevance
            c.overall_score = (
                0.40 * relevance
                + 0.25 * recency
                + 0.20 * stars_score
                + 0.15 * lang_score
            )

    # ------------------------------------------------------------------ #
    #  Deep analysis of a single repo                                     #
    # ------------------------------------------------------------------ #

    def _analyse_repo(self, candidate: RepoCandidate) -> None:
        """
        Shallow-clone the repo and extract structure metadata:
          - README summary (first 400 chars)
          - Entry points (main.py, train.py, run.py, …)
          - Evaluation scripts (evaluate.py, eval.py, benchmark.py, …)
          - Has tests (tests/ or test_*.py)
        """
        repo_dir = self._clone_dir / candidate.name.replace("/", "__")
        if not repo_dir.exists():
            try:
                subprocess.run(
                    ["git", "clone", "--depth=1", candidate.url, str(repo_dir)],
                    check=True, capture_output=True, timeout=60,
                )
            except Exception as exc:
                logger.warning(
                    "GitHubExplorer: could not clone %s (%s)", candidate.name, exc
                )
                return

        # README
        for readme_name in ("README.md", "README.rst", "README.txt", "README"):
            readme = repo_dir / readme_name
            if readme.exists():
                try:
                    candidate.readme_summary = readme.read_text(errors="replace")[:400]
                except OSError:
                    pass
                break

        # Entry points
        for name in ("main.py", "train.py", "run.py", "launch.py", "app.py"):
            if (repo_dir / name).exists():
                candidate.entry_points.append(name)

        # Evaluation scripts
        for name in ("evaluate.py", "eval.py", "run_eval.py", "benchmark.py", "score.py"):
            if (repo_dir / name).exists():
                candidate.eval_scripts.append(name)
        # Also check src/ subdirectory
        src = repo_dir / "src"
        if src.is_dir():
            for name in ("evaluate.py", "eval.py", "benchmark.py"):
                if (src / name).exists():
                    candidate.eval_scripts.append(f"src/{name}")

        candidate.has_eval_script = bool(candidate.eval_scripts)

        # Test presence
        candidate.has_tests = (
            (repo_dir / "tests").is_dir()
            or any(repo_dir.glob("test_*.py"))
            or any(repo_dir.glob("*_test.py"))
        )

        logger.debug(
            "GitHubExplorer: analysed %s — entry_points=%s eval_scripts=%s tests=%s",
            candidate.name,
            candidate.entry_points,
            candidate.eval_scripts,
            candidate.has_tests,
        )

    # ------------------------------------------------------------------ #
    #  Blackboard integration helper                                      #
    # ------------------------------------------------------------------ #

    @staticmethod
    def format_for_blackboard(candidates: list[RepoCandidate]) -> dict[str, Any]:
        """
        Return a dict suitable for storing on the Blackboard as
        ``state.starting_repos``.
        """
        return {
            "repos": [c.to_dict() for c in candidates],
            "best": candidates[0].to_dict() if candidates else None,
        }
