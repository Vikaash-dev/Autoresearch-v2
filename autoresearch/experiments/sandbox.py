"""Sandbox executor with self-healing loop."""
from __future__ import annotations
import logging
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path
from typing import Any, Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from ..config.schema import AutoResearchConfig

logger = logging.getLogger(__name__)


class SandboxExecutor:
    """Executes experiment code in an isolated subprocess."""

    def __init__(self, config: "AutoResearchConfig") -> None:
        self.config = config
        self.sandbox_type = config.experiment.sandbox

    def execute(self, experiment: Dict[str, Any]) -> Dict[str, Any]:
        code = experiment.get("code", "")
        timeout = experiment.get("timeout_seconds", self.config.experiment.timeout_seconds)
        max_retries = experiment.get("max_retries", self.config.experiment.max_retries)

        if not code:
            return {"status": "failed", "error": "No code provided", "stdout": "", "stderr": ""}

        from .self_healer import SelfHealingLoop
        healer = SelfHealingLoop(max_retries=max_retries)
        return healer.run(code, timeout)


def _run_code(code: str, timeout: int) -> Dict[str, Any]:
    """Run Python code string in a subprocess."""
    with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
        f.write(textwrap.dedent(code))
        fname = f.name
    try:
        proc = subprocess.run(
            [sys.executable, fname],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return {
            "status": "success" if proc.returncode == 0 else "failed",
            "returncode": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
        }
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "error": f"Timed out after {timeout}s", "stdout": "", "stderr": ""}
    except Exception as exc:
        return {"status": "error", "error": str(exc), "stdout": "", "stderr": ""}
    finally:
        Path(fname).unlink(missing_ok=True)
