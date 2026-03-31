"""
Tests for tools/github_explorer.py

Covers:
  - GitHubExplorer.from_config() construction
  - _build_query() keyword assembly
  - _parse_results() normalisation of raw results
  - _compute_scores() ranking logic
  - _analyse_repo() on a local fixture repo
  - best_starting_point() convenience method
  - RepoCandidate.to_dict() serialisation
  - format_for_blackboard() output shape
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.github_explorer import GitHubExplorer, RepoCandidate


# ────────────────────────────────────────────────────────────────────────── #
#  Fixtures                                                                   #
# ────────────────────────────────────────────────────────────────────────── #

@pytest.fixture()
def explorer(tmp_path: Path) -> GitHubExplorer:
    return GitHubExplorer(
        top_k=2,
        min_stars=0,
        prefer_language="python",
        clone_dir=str(tmp_path / "repos"),
    )


@pytest.fixture()
def raw_results() -> list[dict]:
    return [
        {
            "name": "owner/fast-attention",
            "url": "https://github.com/owner/fast-attention",
            "description": "Efficient attention mechanisms for transformers",
            "stars": 1500,
            "last_updated": "2025-11-01T00:00:00Z",
            "language": "python",
        },
        {
            "name": "org/slow-rnn",
            "url": "https://github.com/org/slow-rnn",
            "description": "Old RNN baseline",
            "stars": 30,
            "last_updated": "2022-01-01T00:00:00Z",
            "language": "python",
        },
        {
            "name": "ml/attention-bench",
            "url": "https://github.com/ml/attention-bench",
            "description": "Benchmarking attention variants for long context",
            "stars": 800,
            "last_updated": "2026-01-15T00:00:00Z",
            "language": "python",
        },
    ]


@pytest.fixture()
def local_repo(tmp_path: Path) -> Path:
    """A local directory that mimics a cloned GitHub repo."""
    repo = tmp_path / "myrepo"
    repo.mkdir()
    (repo / "README.md").write_text("# MyRepo\nFast attention implementation.\n")
    (repo / "evaluate.py").write_text("print('METRIC: 0.9')\n")
    (repo / "train.py").write_text("# training script\n")
    tests = repo / "tests"
    tests.mkdir()
    (tests / "test_model.py").write_text("def test_pass(): assert True\n")
    return repo


# ────────────────────────────────────────────────────────────────────────── #
#  Construction                                                               #
# ────────────────────────────────────────────────────────────────────────── #

class TestConstruction:
    def test_defaults(self, tmp_path):
        e = GitHubExplorer(clone_dir=str(tmp_path))
        assert e._top_k == 3
        assert e._prefer_language == "python"

    def test_from_config_empty(self, tmp_path):
        e = GitHubExplorer.from_config({})
        assert e._top_k == 3

    def test_from_config_overrides(self, tmp_path):
        cfg = {
            "tools": {
                "github_explorer": {
                    "top_k": 5,
                    "min_stars": 100,
                    "prefer_language": "java",
                    "clone_dir": str(tmp_path),
                }
            }
        }
        e = GitHubExplorer.from_config(cfg)
        assert e._top_k == 5
        assert e._min_stars == 100
        assert e._prefer_language == "java"


# ────────────────────────────────────────────────────────────────────────── #
#  _build_query                                                               #
# ────────────────────────────────────────────────────────────────────────── #

class TestBuildQuery:
    def test_includes_task(self, explorer):
        q = explorer._build_query("efficient attention transformers", "", None)
        assert "efficient attention transformers" in q

    def test_includes_hypothesis(self, explorer):
        q = explorer._build_query("task", "test hypothesis", None)
        assert "hypothesis" in q

    def test_includes_keywords(self, explorer):
        q = explorer._build_query("task", "", ["sparse", "linear"])
        assert "sparse" in q

    def test_includes_site_github(self, explorer):
        q = explorer._build_query("task", "", None)
        assert "github.com" in q


# ────────────────────────────────────────────────────────────────────────── #
#  _parse_results                                                             #
# ────────────────────────────────────────────────────────────────────────── #

class TestParseResults:
    def test_converts_all_results(self, explorer, raw_results):
        candidates = explorer._parse_results(raw_results)
        assert len(candidates) == 3

    def test_stars_mapped(self, explorer, raw_results):
        candidates = explorer._parse_results(raw_results)
        names_to_stars = {c.name: c.stars for c in candidates}
        assert names_to_stars["owner/fast-attention"] == 1500

    def test_deduplication(self, explorer):
        dupes = [
            {"name": "a/b", "url": "https://github.com/a/b", "stars": 100, "language": "python"},
            {"name": "a/b", "url": "https://github.com/a/b", "stars": 100, "language": "python"},
        ]
        candidates = explorer._parse_results(dupes)
        assert len(candidates) == 1

    def test_url_derived_from_name_if_missing(self, explorer):
        raw = [{"name": "owner/repo", "stars": 50, "language": "python"}]
        candidates = explorer._parse_results(raw)
        assert "github.com" in candidates[0].url


# ────────────────────────────────────────────────────────────────────────── #
#  _compute_scores                                                            #
# ────────────────────────────────────────────────────────────────────────── #

class TestComputeScores:
    def test_recent_high_star_scores_highest(self, explorer, raw_results):
        candidates = explorer._parse_results(raw_results)
        explorer._compute_scores(
            candidates,
            task="efficient attention mechanisms for long context transformers",
            hypothesis="",
        )
        sorted_c = sorted(candidates, key=lambda c: c.overall_score, reverse=True)
        # fast-attention (1500 stars, recent, relevant) should beat slow-rnn
        names = [c.name for c in sorted_c]
        assert names.index("owner/fast-attention") < names.index("org/slow-rnn")

    def test_all_scores_in_range(self, explorer, raw_results):
        candidates = explorer._parse_results(raw_results)
        explorer._compute_scores(candidates, task="attention", hypothesis="")
        for c in candidates:
            assert 0.0 <= c.overall_score <= 1.0

    def test_preferred_language_boosts_score(self, explorer):
        python_repo = RepoCandidate(
            name="a/py", url="", stars=100,
            last_updated="2025-06-01T00:00:00Z", language="python"
        )
        java_repo = RepoCandidate(
            name="b/java", url="", stars=100,
            last_updated="2025-06-01T00:00:00Z", language="java"
        )
        explorer._compute_scores([python_repo, java_repo], task="train model", hypothesis="")
        assert python_repo.overall_score > java_repo.overall_score


# ────────────────────────────────────────────────────────────────────────── #
#  _analyse_repo (local fixture, no git)                                      #
# ────────────────────────────────────────────────────────────────────────── #

class TestAnalyseRepo:
    def test_detects_readme(self, explorer, local_repo):
        # Point clone_dir at local_repo's parent so no git clone is needed
        c = RepoCandidate(name="myrepo", url="")
        # Manually place the repo in clone_dir so _analyse_repo skips clone
        dest = explorer._clone_dir / "myrepo"
        import shutil
        shutil.copytree(str(local_repo), str(dest))
        explorer._analyse_repo(c)
        assert "MyRepo" in c.readme_summary

    def test_detects_eval_script(self, explorer, local_repo):
        c = RepoCandidate(name="myrepo2", url="")
        dest = explorer._clone_dir / "myrepo2"
        import shutil
        shutil.copytree(str(local_repo), str(dest))
        explorer._analyse_repo(c)
        assert "evaluate.py" in c.eval_scripts
        assert c.has_eval_script is True

    def test_detects_entry_points(self, explorer, local_repo):
        c = RepoCandidate(name="myrepo3", url="")
        dest = explorer._clone_dir / "myrepo3"
        import shutil
        shutil.copytree(str(local_repo), str(dest))
        explorer._analyse_repo(c)
        assert "train.py" in c.entry_points

    def test_detects_tests(self, explorer, local_repo):
        c = RepoCandidate(name="myrepo4", url="")
        dest = explorer._clone_dir / "myrepo4"
        import shutil
        shutil.copytree(str(local_repo), str(dest))
        explorer._analyse_repo(c)
        assert c.has_tests is True


# ────────────────────────────────────────────────────────────────────────── #
#  RepoCandidate                                                              #
# ────────────────────────────────────────────────────────────────────────── #

class TestRepoCandidate:
    def test_to_dict_keys(self):
        c = RepoCandidate(name="a/b", url="https://github.com/a/b", stars=500)
        d = c.to_dict()
        assert "name" in d
        assert "url" in d
        assert "stars" in d
        assert "overall_score" in d

    def test_readme_summary_truncated(self):
        c = RepoCandidate(name="a/b", url="", readme_summary="x" * 500)
        d = c.to_dict()
        assert len(d["readme_summary"]) <= 400


# ────────────────────────────────────────────────────────────────────────── #
#  format_for_blackboard                                                      #
# ────────────────────────────────────────────────────────────────────────── #

class TestFormatForBlackboard:
    def test_shape_with_candidates(self):
        candidates = [
            RepoCandidate(name="a/b", url="u1", stars=100, overall_score=0.9),
            RepoCandidate(name="c/d", url="u2", stars=50, overall_score=0.7),
        ]
        result = GitHubExplorer.format_for_blackboard(candidates)
        assert "repos" in result
        assert "best" in result
        assert result["best"]["name"] == "a/b"

    def test_shape_with_empty_list(self):
        result = GitHubExplorer.format_for_blackboard([])
        assert result["repos"] == []
        assert result["best"] is None

    def test_json_serialisable(self):
        candidates = [RepoCandidate(name="a/b", url="u", stars=10)]
        result = GitHubExplorer.format_for_blackboard(candidates)
        # Should not raise
        json.dumps(result)
