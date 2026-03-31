"""
HyperAgent Client — wraps FSoft-AI4Code/HyperAgent 4-sub-agent system.

HyperAgent (FSoft-AI4Code/HyperAgent) is a generalist software-engineering agent
system with four specialised sub-agents that mirror the way a human developer
works on a large codebase:

    Planner      — decomposes a high-level SE task into ordered, actionable steps
    Navigator    — semantically searches the codebase (Zoekt + universal-ctags)
    Code Editor  — generates a minimal, correct patch for each step
    Executor     — runs the patched code in a Jupyter kernel and reports results

Reference:
    FSoft-AI4Code/HyperAgent · https://github.com/FSoft-AI4Code/HyperAgent
    SWE-Bench verified: 31.4 % resolved rate (Python)
    RepoExec-Python: 53.3 % Pass@5

In Autoresearch-v2, HyperAgentClient is the primary engine used by
ExperimentAgent for *repo-level* hypothesis evaluation:

    "Given an existing GitHub repo, modify it to implement hypothesis H and
     report the evaluation metric."

If the `hyperagent` package is not installed (it requires Zoekt + Go), the
client falls back to a self-contained bash-only executor inspired by
mini-SWE-agent (https://github.com/SWE-agent/mini-swe-agent).
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────────────── #
#  Data classes                                                               #
# ────────────────────────────────────────────────────────────────────────── #

@dataclass
class SubtaskResult:
    """Result from one HyperAgent sub-agent step."""

    step: int
    description: str
    status: str          # "ok" | "failed" | "skipped"
    output: str
    patch: str = ""


@dataclass
class HyperAgentResult:
    """Aggregated result from a complete HyperAgent run."""

    task: str
    repo_path: str
    mode: str            # "patch" | "predict" | "bash_fallback"
    success: bool
    subtask_results: list[SubtaskResult] = field(default_factory=list)
    final_patch: str = ""
    execution_output: str = ""
    metric: float | None = None
    error: str = ""


# ────────────────────────────────────────────────────────────────────────── #
#  HyperAgent client                                                          #
# ────────────────────────────────────────────────────────────────────────── #

class HyperAgentClient:
    """
    Unified client for FSoft-AI4Code/HyperAgent with bash fallback.

    Instantiation::

        client = HyperAgentClient.from_config(config_dict)

    Running a task::

        result = client.run_task(
            task="Implement and evaluate hypothesis: ...",
            repo_url="https://github.com/user/repo",
        )
        print(result.metric)

    Config keys (config/config.yaml → ``hyperagent`` section):

    ==================  ================================================
    Key                 Meaning
    ==================  ================================================
    mode                "auto" | "patch" | "predict" | "bash"
    language            "python" | "java"  (default: "python")
    clone_dir           local directory for cloned repos
    nav_model           Navigator sub-agent LLM
    edit_model          Code Editor sub-agent LLM
    plan_model          Planner sub-agent LLM
    exec_timeout        seconds before killing Executor subprocess
    api_key_env         name of env-var holding the LLM API key
    ==================  ================================================
    """

    _HYPERAGENT_AVAILABLE: bool | None = None  # cached import check

    def __init__(
        self,
        *,
        mode: str = "auto",
        language: str = "python",
        clone_dir: str = "workspaces/repos",
        nav_model: str = "claude-3-haiku-20240307",
        edit_model: str = "claude-3-5-sonnet-20240620",
        plan_model: str = "gpt-4o",
        exec_timeout: int = 300,
        api_key_env: str = "ANTHROPIC_API_KEY",
    ) -> None:
        self._mode = mode
        self._language = language
        self._clone_dir = Path(clone_dir)
        self._clone_dir.mkdir(parents=True, exist_ok=True)
        self._nav_model = nav_model
        self._edit_model = edit_model
        self._plan_model = plan_model
        self._exec_timeout = exec_timeout
        self._api_key_env = api_key_env

    # ------------------------------------------------------------------ #
    #  Construction                                                        #
    # ------------------------------------------------------------------ #

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "HyperAgentClient":
        """Build from the ``hyperagent`` sub-section of config.yaml."""
        ha = cfg.get("hyperagent", {})
        return cls(
            mode=ha.get("mode", "auto"),
            language=ha.get("language", "python"),
            clone_dir=ha.get("clone_dir", "workspaces/repos"),
            nav_model=ha.get("nav_model", "claude-3-haiku-20240307"),
            edit_model=ha.get("edit_model", "claude-3-5-sonnet-20240620"),
            plan_model=ha.get("plan_model", "gpt-4o"),
            exec_timeout=ha.get("exec_timeout", 300),
            api_key_env=ha.get("api_key_env", "ANTHROPIC_API_KEY"),
        )

    @classmethod
    def _check_available(cls) -> bool:
        """Return True if the FSoft ``hyperagent`` package is installed."""
        if cls._HYPERAGENT_AVAILABLE is None:
            try:
                import hyperagent  # noqa: F401
                cls._HYPERAGENT_AVAILABLE = True
            except ImportError:
                cls._HYPERAGENT_AVAILABLE = False
        return cls._HYPERAGENT_AVAILABLE  # type: ignore[return-value]

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def run_task(
        self,
        task: str,
        repo_url: str | None = None,
        repo_path: str | None = None,
        commit: str | None = None,
    ) -> HyperAgentResult:
        """
        Run *task* on a repository using the 4-sub-agent HyperAgent pipeline.

        Provide either *repo_url* (auto-cloned into ``clone_dir``) or an
        already-cloned *repo_path*.
        """
        if repo_path:
            resolved = Path(repo_path)
        elif repo_url:
            resolved = self._clone_repo(repo_url, commit)
        else:
            raise ValueError("Provide either repo_url or repo_path")

        effective_mode = self._mode
        if effective_mode == "auto":
            effective_mode = "patch" if self._check_available() else "bash_fallback"

        logger.info("HyperAgentClient: mode=%s  repo=%s", effective_mode, resolved)

        if effective_mode in ("patch", "predict") and self._check_available():
            return self._run_fsoft(task, resolved, commit or "HEAD", effective_mode)
        return self._run_bash_fallback(task, resolved)

    def evaluate_hypothesis_on_repo(
        self,
        hypothesis: str,
        repo_url: str,
        eval_command: str = "python evaluate.py",
        commit: str | None = None,
    ) -> HyperAgentResult:
        """
        Implement *hypothesis* on *repo_url*, run *eval_command*, parse metric.

        This is the primary interface used by :class:`~agents.experiment_agent.ExperimentAgent`
        for repo-level tasks.  The printed line ``METRIC: <float>`` in
        *eval_command* stdout is parsed as the primary evaluation metric.
        """
        task = (
            "Implement and evaluate the following research hypothesis on this codebase:\n\n"
            f"HYPOTHESIS: {hypothesis}\n\n"
            f"After implementing, run `{eval_command}` and report the primary metric "
            "by printing exactly one line: METRIC: <float>\n"
            "Ensure the implementation is minimal, focused, and verifiable."
        )
        result = self.run_task(task=task, repo_url=repo_url, commit=commit)
        if result.metric is None and result.execution_output:
            result.metric = self._parse_metric(result.execution_output)
        return result

    # ------------------------------------------------------------------ #
    #  FSoft HyperAgent execution path                                     #
    # ------------------------------------------------------------------ #

    def _run_fsoft(
        self, task: str, repo_path: Path, commit: str, mode: str
    ) -> HyperAgentResult:
        """Delegate to the installed FSoft ``hyperagent`` package."""
        try:
            from hyperagent import HyperAgent  # type: ignore[import]
        except ImportError:
            logger.warning("hyperagent import failed; using bash fallback")
            return self._run_bash_fallback(task, repo_path)

        api_key = os.environ.get(self._api_key_env, "")
        config = self._build_fsoft_config(api_key)

        try:
            pilot = HyperAgent(
                str(repo_path),
                commit=commit,
                language=self._language,
                clone_dir=str(self._clone_dir),
                config=config,
            )
            raw = pilot.run(task, mode=mode)
        except Exception as exc:
            logger.error("FSoft HyperAgent raised: %s", exc)
            return HyperAgentResult(
                task=task,
                repo_path=str(repo_path),
                mode=mode,
                success=False,
                error=str(exc),
            )

        return self._normalise_fsoft_output(raw, task, repo_path, mode)

    def _build_fsoft_config(self, api_key: str) -> dict[str, Any]:
        """Return the config dict expected by FSoft HyperAgent."""
        stop = ["\nObservation:"]
        anthropic_base = "https://api.anthropic.com"
        openai_base = "https://api.openai.com/v1"
        openai_key = os.environ.get("OPENAI_API_KEY", "")
        return {
            "name": "autoresearch_v2",
            "nav": [{"model": self._nav_model, "api_key": api_key,
                      "stop_sequences": stop, "base_url": anthropic_base,
                      "api_type": "anthropic"}],
            "edit": [{"model": self._edit_model, "api_key": api_key,
                       "stop_sequences": stop, "base_url": anthropic_base,
                       "api_type": "anthropic"}],
            "plan": [{"model": self._plan_model, "api_key": openai_key,
                       "base_url": openai_base, "api_type": "openai"}],
            "exec": [{"model": self._edit_model, "api_key": api_key,
                       "stop_sequences": stop, "base_url": anthropic_base,
                       "api_type": "anthropic"}],
        }

    @staticmethod
    def _normalise_fsoft_output(
        raw: Any, task: str, repo_path: Path, mode: str
    ) -> HyperAgentResult:
        """Convert FSoft output (dict or str) to :class:`HyperAgentResult`."""
        if isinstance(raw, dict):
            patch = raw.get("patch", "")
            output = raw.get("output", str(raw))
            success = raw.get("success", bool(patch or output))
        elif isinstance(raw, str):
            patch = raw if raw.startswith("---") else ""
            output = raw
            success = bool(raw.strip())
        else:
            patch = ""
            output = str(raw)
            success = bool(raw)

        metric = HyperAgentClient._parse_metric(output)
        return HyperAgentResult(
            task=task,
            repo_path=str(repo_path),
            mode=mode,
            success=success,
            final_patch=patch,
            execution_output=output,
            metric=metric,
        )

    # ------------------------------------------------------------------ #
    #  Bash fallback — 4-stage pipeline, no external deps                  #
    # ------------------------------------------------------------------ #

    def _run_bash_fallback(self, task: str, repo_path: Path) -> HyperAgentResult:
        """
        Self-contained bash-only fallback that mirrors HyperAgent's 4 sub-agents.

        Stage 1 — **Planner**:   decompose the task into ordered steps (rule-based)
        Stage 2 — **Navigator**: locate relevant source files via ``grep``
        Stage 3 — **Code Editor**: apply any embedded unified-diff patch
        Stage 4 — **Executor**:  run the evaluation entry-point and capture stdout
        """
        subtask_results: list[SubtaskResult] = []
        work_dir = Path(tempfile.mkdtemp(prefix="ha_bash_"))

        # Isolated copy so we never mutate the source
        work_repo = work_dir / "repo"
        try:
            shutil.copytree(str(repo_path), str(work_repo))
        except Exception as exc:
            return HyperAgentResult(
                task=task, repo_path=str(repo_path), mode="bash_fallback",
                success=False, error=f"Could not copy repo: {exc}",
            )

        # ── 1. Planner ────────────────────────────────────────────────── #
        steps = self._plan_task(task)
        subtask_results.append(SubtaskResult(
            step=1, description="Planner: decompose task",
            status="ok", output=json.dumps(steps),
        ))

        # ── 2. Navigator ──────────────────────────────────────────────── #
        relevant = self._navigate(work_repo, task)
        subtask_results.append(SubtaskResult(
            step=2, description="Navigator: locate relevant files",
            status="ok",
            output="\n".join(str(f.relative_to(work_repo)) for f in relevant[:10]),
        ))

        # ── 3. Code Editor ────────────────────────────────────────────── #
        edit_status, edit_out, patch = self._apply_patch_if_present(task, work_repo)
        subtask_results.append(SubtaskResult(
            step=3, description="Code Editor: apply patch",
            status=edit_status, output=edit_out, patch=patch,
        ))

        # ── 4. Executor ───────────────────────────────────────────────── #
        exec_status, exec_out = self._execute_in_repo(work_repo, steps)
        subtask_results.append(SubtaskResult(
            step=4, description="Executor: run evaluation",
            status=exec_status, output=exec_out,
        ))

        metric = self._parse_metric(exec_out)
        success = metric is not None or exec_status == "ok"
        return HyperAgentResult(
            task=task,
            repo_path=str(repo_path),
            mode="bash_fallback",
            success=success,
            subtask_results=subtask_results,
            final_patch=patch,
            execution_output=exec_out,
            metric=metric,
        )

    # ------------------------------------------------------------------ #
    #  Bash sub-agent implementations                                      #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _plan_task(task: str) -> list[str]:
        """Rule-based task decomposition — no LLM required."""
        steps: list[str] = []
        lower = task.lower()
        if any(w in lower for w in ("implement", "add", "create", "modify")):
            steps.append("Identify files to modify")
            steps.append("Implement the required changes")
        if any(w in lower for w in ("evaluate", "metric", "score", "benchmark")):
            steps.append("Run the evaluation script")
            steps.append("Parse and report the metric")
        if "test" in lower:
            steps.append("Run the test suite")
        return steps or ["Analyse the repository", "Execute relevant scripts"]

    def _navigate(self, repo_path: Path, task: str) -> list[Path]:
        """Return Python files that contain task-related keywords."""
        keywords = [
            w.strip(".,;:()\"'").lower()
            for w in task.split()
            if len(w) > 4 and w.isalpha()
        ][:5]

        found: set[Path] = set()
        for kw in keywords:
            try:
                out = subprocess.run(
                    ["grep", "-rl", "--include=*.py", kw, str(repo_path)],
                    capture_output=True, text=True, timeout=10,
                )
                for line in out.stdout.splitlines():
                    found.add(Path(line.strip()))
            except (subprocess.TimeoutExpired, FileNotFoundError):
                continue

        return sorted(found) or sorted(repo_path.rglob("*.py"))

    @staticmethod
    def _apply_patch_if_present(
        task: str, repo_path: Path
    ) -> tuple[str, str, str]:
        """Apply an embedded unified-diff patch from *task* if present."""
        patch_lines: list[str] = []
        in_patch = False
        for line in task.splitlines():
            if line.startswith("--- ") or line.startswith("+++ "):
                in_patch = True
            if in_patch:
                patch_lines.append(line)

        if not patch_lines:
            return "skipped", "No embedded patch in task; skipping edit stage.", ""

        patch_text = "\n".join(patch_lines)
        patch_file = repo_path / "_ar2_patch.diff"
        patch_file.write_text(patch_text)
        try:
            res = subprocess.run(
                ["patch", "-p1", "--input", str(patch_file)],
                capture_output=True, text=True, timeout=30,
                cwd=str(repo_path),
            )
            status = "ok" if res.returncode == 0 else "failed"
            output = res.stdout if res.returncode == 0 else res.stderr
            return status, output, patch_text
        except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
            return "failed", str(exc), patch_text
        finally:
            patch_file.unlink(missing_ok=True)

    def _execute_in_repo(
        self, repo_path: Path, steps: list[str]
    ) -> tuple[str, str]:
        """
        Run the first matching evaluation entry-point found in the repo.

        Checks for: evaluate.py, eval.py, run_eval.py, main.py, benchmark.py,
        score.py — then falls back to ``python -m pytest``.
        """
        for name in ("evaluate.py", "eval.py", "run_eval.py",
                     "main.py", "benchmark.py", "score.py"):
            script = repo_path / name
            if script.exists():
                return self._run_script(script, repo_path)

        # pytest fallback
        try:
            res = subprocess.run(
                ["python", "-m", "pytest", "--tb=short", "-q"],
                capture_output=True, text=True,
                timeout=self._exec_timeout,
                cwd=str(repo_path),
            )
            output = (res.stdout + res.stderr)[-3000:]
            return ("ok" if res.returncode == 0 else "failed"), output
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return "skipped", "No evaluation script or test suite found."

    def _run_script(self, script: Path, cwd: Path) -> tuple[str, str]:
        """Execute a Python script and return (status, stdout+stderr)."""
        try:
            res = subprocess.run(
                ["python", str(script)],
                capture_output=True, text=True,
                timeout=self._exec_timeout,
                cwd=str(cwd),
            )
            combined = res.stdout + ("\n" + res.stderr if res.stderr else "")
            return ("ok" if res.returncode == 0 else "failed"), combined[-3000:]
        except subprocess.TimeoutExpired:
            return "failed", f"Timeout after {self._exec_timeout}s"
        except Exception as exc:
            return "failed", str(exc)

    # ------------------------------------------------------------------ #
    #  Helpers                                                             #
    # ------------------------------------------------------------------ #

    def _clone_repo(self, url: str, commit: str | None) -> Path:
        """Shallow-clone *url* into ``clone_dir``; checkout *commit* if given."""
        name = url.rstrip("/").split("/")[-1].removesuffix(".git")
        dest = self._clone_dir / name
        if not dest.exists():
            logger.info("Cloning %s → %s", url, dest)
            subprocess.run(
                ["git", "clone", "--depth=1", url, str(dest)],
                check=True, capture_output=True, timeout=120,
            )
        if commit and commit != "HEAD":
            subprocess.run(
                ["git", "checkout", commit],
                cwd=str(dest), check=True, capture_output=True,
            )
        return dest

    @staticmethod
    def _parse_metric(output: str) -> float | None:
        """Parse the *last* ``METRIC: <float>`` line from stdout."""
        for line in reversed(output.splitlines()):
            stripped = line.strip()
            if stripped.startswith("METRIC:"):
                try:
                    return float(stripped[len("METRIC:"):].strip())
                except ValueError:
                    continue
        return None
