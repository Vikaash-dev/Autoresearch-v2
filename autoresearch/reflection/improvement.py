"""Safe self-improvement engine (config/policy-level updates only)."""
from __future__ import annotations
import logging
from typing import Any, Dict, List, TYPE_CHECKING

if TYPE_CHECKING:
    from ..config.schema import AutoResearchConfig
    from ..core.state import RunState

logger = logging.getLogger(__name__)


class SelfImprovementEngine:
    """Applies safe, config-level policy updates based on self-review recommendations."""

    SAFE_PARAMETERS = {
        "arxiv_max_results", "max_retries", "timeout_seconds",
        "adversarial_rounds", "num_hypotheses", "temperature",
    }

    def __init__(self, config: "AutoResearchConfig") -> None:
        self.config = config

    def apply_updates(self, updates: List[Dict[str, Any]], state: "RunState") -> List[Dict[str, Any]]:
        """Apply only safe policy updates, logging all actions."""
        applied = []
        for update in updates:
            if not update.get("safe", False):
                logger.warning("Skipping unsafe update: %s", update)
                continue
            param = update.get("parameter", "")
            if param not in self.SAFE_PARAMETERS:
                logger.warning("Parameter '%s' not in safe-list; skipping", param)
                continue
            logger.info("Applying policy update: %s.%s = %s", update.get("module"), param, update.get("proposed"))
            applied.append({
                "applied": True,
                "module": update["module"],
                "parameter": param,
                "old_value": update.get("current"),
                "new_value": update.get("proposed"),
                "rationale": update.get("rationale", ""),
            })
        state.policy_updates.extend(applied)
        return applied

    def validate_update(self, update: Dict[str, Any]) -> bool:
        """Validate that an update is safe to apply."""
        if not update.get("safe", False):
            return False
        if update.get("parameter") not in self.SAFE_PARAMETERS:
            return False
        if update.get("proposed") is None:
            return False
        return True
