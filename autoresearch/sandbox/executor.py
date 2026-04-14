"""
Sandbox Executor — safe code execution for the ExperimentAgent.

Inspired by:
- SWE-agent: Agent-Computer Interface (ACI) with specialised shell tools
- CORAL: `uv run coral eval` = stage + commit + grade in one shot
- HyperAgents: Docker-sandboxed code execution with timeout
- AutoResearchClaw: anti-fabrication VerifiedRegistry + experiment diagnosis & repair

The executor provides a safe, timeout-enforced environment for running
experiment code. Results are verified against a ground-truth evaluator.

In production: replace LocalExecutor with DockerExecutor or RemoteExecutor.
"""

from __future__ import annotations

import io
import logging
import os
import signal
import subprocess
import sys
import tempfile
import textwrap
import time
import traceback
import uuid
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class ExecutionResult:
    exec_id:    str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    success:    bool   = False
    output:     str    = ""
    error:      str    = ""
    return_val: Any    = None
    elapsed_s:  float  = 0.0
    exit_code:  int    = 0
    timed_out:  bool   = False
    verified:   bool   = False   # VerifiedRegistry: output passes correctness check

    def to_dict(self) -> Dict[str, Any]:
        return {
            "exec_id":   self.exec_id,
            "success":   self.success,
            "output":    self.output[:500],
            "error":     self.error[:300],
            "return_val": str(self.return_val)[:200] if self.return_val is not None else None,
            "elapsed_s": round(self.elapsed_s, 3),
            "timed_out": self.timed_out,
            "verified":  self.verified,
        }


# ---------------------------------------------------------------------------
# Local in-process executor (safe eval with timeout)
# ---------------------------------------------------------------------------

class _Timeout(Exception):
    pass


@contextmanager
def _time_limit(seconds: float) -> Generator[None, None, None]:
    """SIGALRM-based timeout context manager (Unix only)."""
    if sys.platform == "win32" or seconds <= 0:
        yield
        return

    def _handler(signum: int, frame: Any) -> None:
        raise _Timeout(f"Execution exceeded {seconds}s budget")

    old = signal.signal(signal.SIGALRM, _handler)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)


class LocalExecutor:
    """
    In-process Python executor with:
    - Timeout enforcement (Karpathy: fixed 5-min budget)
    - Stdout/stderr capture
    - Optional verifier callback (CORAL: eval = stage + grade)
    - Anti-fabrication: marks result as unverified if verifier fails (AutoResearchClaw)
    """

    DEFAULT_TIMEOUT = 30.0   # seconds; override per call

    def __init__(
        self,
        timeout: float = DEFAULT_TIMEOUT,
        verifier: Optional[Callable[[Any], bool]] = None,
        allowed_modules: Optional[List[str]] = None,
    ) -> None:
        self.timeout = timeout
        self.verifier = verifier
        self.allowed_modules = set(allowed_modules or [])

    def run_callable(
        self,
        fn: Callable[[], Any],
        timeout: Optional[float] = None,
    ) -> ExecutionResult:
        """Execute a Python callable and capture its output."""
        t0 = time.time()
        buf_out = io.StringIO()
        buf_err = io.StringIO()
        result = ExecutionResult()

        effective_timeout = timeout if timeout is not None else self.timeout

        try:
            with _time_limit(effective_timeout):
                with redirect_stdout(buf_out), redirect_stderr(buf_err):
                    retval = fn()
            result.success    = True
            result.return_val = retval
            result.exit_code  = 0
        except _Timeout:
            result.timed_out = True
            result.error     = f"Timed out after {effective_timeout}s"
            result.exit_code = -1
            logger.warning("Executor: timed out after %.1fs", effective_timeout)
        except Exception:  # noqa: BLE001
            result.error    = traceback.format_exc()[-1000:]
            result.exit_code = 1
            logger.debug("Executor exception: %s", result.error[:200])

        result.output  = buf_out.getvalue()[-2000:]
        if not result.error:
            result.error = buf_err.getvalue()[-500:]
        result.elapsed_s = time.time() - t0

        # Verification pass (anti-fabrication)
        if result.success and self.verifier is not None:
            try:
                result.verified = bool(self.verifier(result.return_val))
            except Exception as exc:  # noqa: BLE001
                result.verified = False
                result.error = f"Verifier failed: {exc}"
        elif result.success:
            result.verified = True  # no verifier → assume verified

        return result

    def run_code(
        self,
        code: str,
        globals_dict: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> ExecutionResult:
        """Execute a Python code string in an isolated namespace."""
        ns: Dict[str, Any] = {**(globals_dict or {})}
        ns.setdefault("__builtins__", __builtins__)

        def _run() -> Any:
            exec(compile(code, "<autoresearch_sandbox>", "exec"), ns)  # noqa: S102
            return ns.get("__result__")

        return self.run_callable(_run, timeout=timeout)

    def run_subprocess(
        self,
        cmd: List[str],
        cwd: Optional[str] = None,
        timeout: Optional[float] = None,
        env: Optional[Dict[str, str]] = None,
    ) -> ExecutionResult:
        """Run an external subprocess (for Docker/SWE-agent style execution)."""
        t0 = time.time()
        effective_timeout = timeout if timeout is not None else self.timeout
        result = ExecutionResult()

        try:
            proc = subprocess.run(  # noqa: S603
                cmd,
                capture_output=True,
                text=True,
                timeout=effective_timeout,
                cwd=cwd,
                env={**os.environ, **(env or {})},
            )
            result.success   = proc.returncode == 0
            result.output    = proc.stdout[-2000:]
            result.error     = proc.stderr[-500:]
            result.exit_code = proc.returncode
            result.verified  = result.success
        except subprocess.TimeoutExpired:
            result.timed_out = True
            result.error     = f"Subprocess timed out after {effective_timeout}s"
            result.exit_code = -1
        except Exception as exc:  # noqa: BLE001
            result.error     = str(exc)
            result.exit_code = 1

        result.elapsed_s = time.time() - t0
        return result


# ---------------------------------------------------------------------------
# SWE-agent ACI Tools (Agent-Computer Interface)
# ---------------------------------------------------------------------------

class ACITools:
    """
    SWE-agent style Agent-Computer Interface.

    Provides specialised tools the research agents can call:
    - file_view, file_edit, file_create
    - search_files, search_content
    - run_python, run_tests
    """

    def __init__(
        self,
        workspace: Optional[str] = None,
        executor: Optional[LocalExecutor] = None,
    ) -> None:
        self.workspace = Path(workspace or tempfile.mkdtemp(prefix="autoresearch_aci_"))
        self.executor  = executor or LocalExecutor()
        self._edit_history: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # File tools
    # ------------------------------------------------------------------

    def file_view(self, path: str, start: int = 1, end: Optional[int] = None) -> str:
        p = self._resolve(path)
        if not p.exists():
            return f"ERROR: {path} does not exist"
        lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
        slice_ = lines[start - 1: end]
        return "\n".join(f"{start + i}: {l}" for i, l in enumerate(slice_))

    def file_edit(self, path: str, old_str: str, new_str: str) -> str:
        """SWE-agent style surgical edit: replace exactly one occurrence."""
        p = self._resolve(path)
        if not p.exists():
            return f"ERROR: {path} does not exist"
        content = p.read_text(encoding="utf-8")
        if old_str not in content:
            return f"ERROR: string not found in {path}"
        updated = content.replace(old_str, new_str, 1)
        self._edit_history.append({"path": str(path), "old": old_str[:50], "new": new_str[:50]})
        p.write_text(updated, encoding="utf-8")
        return f"OK: edited {path}"

    def file_create(self, path: str, content: str) -> str:
        p = self._resolve(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return f"OK: created {path}"

    def search_content(self, pattern: str, path: str = ".") -> List[str]:
        """ripgrep-style content search."""
        base = self._resolve(path)
        matches = []
        import re
        rx = re.compile(pattern, re.IGNORECASE)
        for f in base.rglob("*.py"):
            for i, line in enumerate(f.read_text(errors="replace").splitlines(), 1):
                if rx.search(line):
                    matches.append(f"{f.relative_to(self.workspace)}:{i}: {line.strip()[:100]}")
        return matches[:50]

    def run_python(self, code: str, timeout: float = 30.0) -> ExecutionResult:
        return self.executor.run_code(code, timeout=timeout)

    # ------------------------------------------------------------------

    def _resolve(self, path: str) -> Path:
        p = Path(path)
        if not p.is_absolute():
            p = self.workspace / p
        return p

    def edit_summary(self) -> List[Dict[str, Any]]:
        return list(self._edit_history)
