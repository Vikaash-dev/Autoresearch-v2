"""Self-healing execution loop: detect failures, repair, retry."""
from __future__ import annotations
import logging
import re
from typing import Any, Dict

logger = logging.getLogger(__name__)


class SelfHealingLoop:
    """Detect execution failures and attempt code repair with bounded retries."""

    COMMON_FIXES = [
        (r"ModuleNotFoundError: No module named '(\w+)'", "add_import_stub"),
        (r"SyntaxError", "fix_indentation"),
        (r"NameError: name '(\w+)' is not defined", "add_variable_stub"),
    ]

    def __init__(self, max_retries: int = 3) -> None:
        self.max_retries = max_retries

    def run(self, code: str, timeout: int = 300) -> Dict[str, Any]:
        from .sandbox import _run_code

        attempt = 0
        current_code = code
        last_result: Dict[str, Any] = {}

        while attempt <= self.max_retries:
            logger.debug("Attempt %d/%d", attempt + 1, self.max_retries + 1)
            result = _run_code(current_code, timeout)
            last_result = result

            if result["status"] == "success":
                result["attempts"] = attempt + 1
                return result

            if attempt < self.max_retries:
                repaired = self._repair(current_code, result)
                if repaired == current_code:
                    logger.debug("No repair possible; stopping")
                    break
                current_code = repaired
                logger.info("Code repaired (attempt %d)", attempt + 1)
            attempt += 1

        last_result["attempts"] = attempt
        last_result["healed"] = False
        return last_result

    def _repair(self, code: str, result: Dict[str, Any]) -> str:
        stderr = result.get("stderr", "")
        for pattern, fix_type in self.COMMON_FIXES:
            match = re.search(pattern, stderr)
            if match:
                fix_method = getattr(self, f"_fix_{fix_type}", None)
                if fix_method is None:
                    logger.debug("No repair method for fix type '%s'; skipping", fix_type)
                    continue
                return fix_method(code, match)
        return code

    def _fix_add_import_stub(self, code: str, match: re.Match) -> str:
        module = match.group(1)
        stub = f"# Auto-stub for missing module\ntry:\n    import {module}\nexcept ImportError:\n    {module} = None\n"
        return stub + code

    def _fix_fix_indentation(self, code: str, match: re.Match) -> str:
        lines = code.splitlines()
        fixed = []
        for line in lines:
            if line.startswith("\t"):
                line = line.replace("\t", "    ")
            fixed.append(line)
        return "\n".join(fixed)

    def _fix_add_variable_stub(self, code: str, match: re.Match) -> str:
        var = match.group(1)
        stub = f"{var} = None  # Auto-stub\n"
        return stub + code
