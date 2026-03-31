"""
Trajectory log for Autoresearch-v2.

Records every agent action with full context: inputs, outputs, metrics, and
execution traces. Used by:
  - Self-ToM (analyze_self) — identifies failure patterns and blind spots
  - GEPA evolution engine   — reads traces as Actionable Side Information (ASI)
  - Atropos RL integration  — trajectory format compatible with RL reward signals

Storage: JSON Lines format (one record per line) under trajectory_log_path.
  trajectories/
    <run_id>.jsonl         ← one file per research run
    index.json             ← lightweight index of all runs

Each trajectory record:
    {
      "event_id":   str,       # unique event identifier
      "run_id":     str,
      "agent_id":   str,
      "agent_role": str,
      "action":     str,       # e.g. "generate_hypothesis", "evaluate_node"
      "inputs":     dict,      # what the agent received
      "outputs":    dict,      # what the agent produced
      "metric":     float|null,
      "status":     str,       # "success" | "failure" | "partial"
      "error":      str,       # empty on success
      "trace":      str,       # full execution trace (for GEPA reflection)
      "duration_s": float,
      "timestamp":  float,
    }
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class TrajectoryEvent:
    """A single recorded agent action."""

    event_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    run_id: str = ""
    agent_id: str = ""
    agent_role: str = ""
    action: str = ""
    inputs: dict[str, Any] = field(default_factory=dict)
    outputs: dict[str, Any] = field(default_factory=dict)
    metric: float | None = None
    status: str = "success"   # success | failure | partial
    error: str = ""
    trace: str = ""           # full execution trace (stdout + stderr)
    duration_s: float = 0.0
    timestamp: float = field(default_factory=time.time)


class TrajectoryLog:
    """
    Append-only log of agent trajectory events.

    Usage:
        log = TrajectoryLog(Path("memory/trajectories"))
        token = log.begin(run_id, agent_id, agent_role, action, inputs)
        # ... agent executes ...
        log.end(token, outputs=result, metric=0.85, trace=stdout)

    Query:
        events = log.query(run_id="abc", action="evaluate_node", status="failure")
        analysis = log.analyze_failures(agent_id="experiment_xyz123")
    """

    def __init__(self, root: Path) -> None:
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)
        self._index_path = self._root / "index.json"
        self._pending: dict[str, tuple[TrajectoryEvent, float]] = {}

    # ------------------------------------------------------------------ #
    #  Write API                                                            #
    # ------------------------------------------------------------------ #

    def begin(
        self,
        run_id: str,
        agent_id: str,
        agent_role: str,
        action: str,
        inputs: dict[str, Any] | None = None,
    ) -> str:
        """
        Start recording an action. Returns a token to pass to end().
        """
        event = TrajectoryEvent(
            run_id=run_id,
            agent_id=agent_id,
            agent_role=agent_role,
            action=action,
            inputs=self._safe_dict(inputs),
        )
        self._pending[event.event_id] = (event, time.time())
        return event.event_id

    def end(
        self,
        token: str,
        outputs: dict[str, Any] | None = None,
        metric: float | None = None,
        status: str = "success",
        error: str = "",
        trace: str = "",
    ) -> TrajectoryEvent:
        """
        Complete a recorded action and append it to the log.
        """
        if token not in self._pending:
            raise KeyError(f"Unknown trajectory token: {token!r}")

        event, start_time = self._pending.pop(token)
        event.outputs = self._safe_dict(outputs)
        event.metric = metric
        event.status = status
        event.error = error[:2000]    # cap trace at 2000 chars for storage efficiency
        event.trace = trace[-3000:]   # keep last 3000 chars (most informative)
        event.duration_s = round(time.time() - start_time, 3)

        self._append(event)
        return event

    def log_event(
        self,
        run_id: str,
        agent_id: str,
        agent_role: str,
        action: str,
        inputs: dict[str, Any] | None = None,
        outputs: dict[str, Any] | None = None,
        metric: float | None = None,
        status: str = "success",
        error: str = "",
        trace: str = "",
    ) -> TrajectoryEvent:
        """One-shot convenience method: log a completed event immediately."""
        token = self.begin(run_id, agent_id, agent_role, action, inputs)
        return self.end(token, outputs=outputs, metric=metric,
                        status=status, error=error, trace=trace)

    # ------------------------------------------------------------------ #
    #  Read / Query API                                                     #
    # ------------------------------------------------------------------ #

    def load_run(self, run_id: str) -> list[TrajectoryEvent]:
        """Load all events for a specific run."""
        path = self._root / f"{run_id}.jsonl"
        if not path.exists():
            return []
        events = []
        for line in path.read_text().splitlines():
            line = line.strip()
            if line:
                try:
                    events.append(TrajectoryEvent(**json.loads(line)))
                except Exception:
                    pass
        return events

    def query(
        self,
        run_id: str | None = None,
        agent_id: str | None = None,
        action: str | None = None,
        status: str | None = None,
        min_metric: float | None = None,
    ) -> list[TrajectoryEvent]:
        """
        Query events across all runs with optional filters.
        Returns events sorted by timestamp (oldest first).
        """
        all_events: list[TrajectoryEvent] = []
        if run_id:
            all_events = self.load_run(run_id)
        else:
            for jsonl in sorted(self._root.glob("*.jsonl")):
                all_events.extend(self.load_run(jsonl.stem))

        results = []
        for e in all_events:
            if agent_id and e.agent_id != agent_id:
                continue
            if action and e.action != action:
                continue
            if status and e.status != status:
                continue
            if min_metric is not None and (e.metric is None or e.metric < min_metric):
                continue
            results.append(e)

        return sorted(results, key=lambda e: e.timestamp)

    # ------------------------------------------------------------------ #
    #  Analysis for Self-ToM and GEPA                                       #
    # ------------------------------------------------------------------ #

    def analyze_failures(self, agent_id: str | None = None) -> dict[str, Any]:
        """
        Summarize failure patterns across all recorded events.
        Used by Self-ToM.analyze_self() and the GEPA evolution engine.

        Returns:
            {
              "total_events": int,
              "success_rate": float,
              "failure_by_action": {action: count},
              "common_errors": [str],       # most frequent error substrings
              "worst_actions": [str],        # actions with highest failure rate
              "best_actions":  [str],
              "avg_metric_by_action": {action: float},
            }
        """
        events = self.query(agent_id=agent_id)
        if not events:
            return {
                "total_events": 0,
                "success_rate": 0.0,
                "failure_by_action": {},
                "common_errors": [],
                "worst_actions": [],
                "best_actions": [],
                "avg_metric_by_action": {},
            }

        total = len(events)
        successes = sum(1 for e in events if e.status == "success")

        # Count failures per action
        failure_by_action: dict[str, int] = {}
        total_by_action: dict[str, int] = {}
        metric_by_action: dict[str, list[float]] = {}

        for e in events:
            total_by_action[e.action] = total_by_action.get(e.action, 0) + 1
            if e.status == "failure":
                failure_by_action[e.action] = failure_by_action.get(e.action, 0) + 1
            if e.metric is not None:
                metric_by_action.setdefault(e.action, []).append(e.metric)

        # Most common error substrings
        all_errors = [e.error for e in events if e.error]
        error_freq: dict[str, int] = {}
        for err in all_errors:
            # Use first 60 chars of error as the key
            key = err[:60].strip()
            error_freq[key] = error_freq.get(key, 0) + 1
        common_errors = sorted(error_freq, key=lambda k: error_freq[k], reverse=True)[:5]

        # Worst actions by failure rate
        failure_rates = {
            a: failure_by_action.get(a, 0) / total_by_action[a]
            for a in total_by_action
        }
        worst = sorted(failure_rates, key=lambda k: failure_rates[k], reverse=True)[:3]
        best = sorted(failure_rates, key=lambda k: failure_rates[k])[:3]

        avg_metric = {
            a: round(sum(ms) / len(ms), 4)
            for a, ms in metric_by_action.items()
            if ms
        }

        return {
            "total_events": total,
            "success_rate": round(successes / total, 4),
            "failure_by_action": failure_by_action,
            "common_errors": common_errors,
            "worst_actions": worst,
            "best_actions": best,
            "avg_metric_by_action": avg_metric,
        }

    def list_runs(self) -> list[dict[str, Any]]:
        """Return a summary of all logged runs."""
        index = self._load_index()
        return list(index.values())

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _append(self, event: TrajectoryEvent) -> None:
        """Append a single event to the run's JSONL file and update the index."""
        run_file = self._root / f"{event.run_id}.jsonl"
        with run_file.open("a") as f:
            f.write(json.dumps(asdict(event), default=str) + "\n")

        # Update index
        index = self._load_index()
        if event.run_id not in index:
            index[event.run_id] = {
                "run_id": event.run_id,
                "first_event": event.timestamp,
                "last_event": event.timestamp,
                "total_events": 0,
            }
        entry = index[event.run_id]
        entry["last_event"] = event.timestamp
        entry["total_events"] = entry.get("total_events", 0) + 1
        self._save_index(index)

    def _load_index(self) -> dict[str, Any]:
        if self._index_path.exists():
            try:
                return json.loads(self._index_path.read_text())
            except Exception:
                pass
        return {}

    def _save_index(self, index: dict[str, Any]) -> None:
        tmp = self._index_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(index, indent=2))
        tmp.replace(self._index_path)

    @staticmethod
    def _safe_dict(d: Any) -> dict[str, Any]:
        """Convert any value to a JSON-safe dict."""
        if d is None:
            return {}
        if isinstance(d, dict):
            try:
                json.dumps(d)
                return d
            except (TypeError, ValueError):
                return {"_repr": str(d)[:500]}
        return {"_value": str(d)[:500]}
