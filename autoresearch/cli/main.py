"""CLI entrypoints for AutoResearch v2."""
from __future__ import annotations
import json
import logging
import sys
from pathlib import Path
from typing import Optional
import click
try:
    from rich.console import Console
    from rich.table import Table
    from rich.progress import Progress, SpinnerColumn, TextColumn
    RICH = True
except ImportError:
    RICH = False

from ..config.schema import load_config
from ..core.objective_graph import Objective

console = Console() if RICH else None


def _print(msg: str) -> None:
    if console:
        console.print(msg)
    else:
        print(msg)


@click.group()
@click.version_option(version="2.0.0", prog_name="autoresearch")
def cli() -> None:
    """AutoResearch v2 — Self-Evolving Multi-Agent Research Framework."""


@cli.command("run")
@click.option("--topic", "-t", required=True, help="Research topic")
@click.option("--config", "-c", default=None, help="Path to config YAML/TOML")
@click.option("--run-id", default=None, help="Custom run ID")
@click.option("--max-iter", default=None, type=int, help="Override max_iterations")
@click.option("--verbose", "-v", is_flag=True, default=False)
def run_cmd(topic: str, config: Optional[str], run_id: Optional[str], max_iter: Optional[int], verbose: bool) -> None:
    """Start a new research run."""
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = load_config(config)
    cfg.topic = topic
    if run_id:
        cfg.run_id = run_id
    if max_iter is not None:
        cfg.max_iterations = max_iter

    if RICH:
        _print(f"[bold green]AutoResearch v2[/bold green] — Starting run for: [cyan]{topic}[/cyan]")
    else:
        _print(f"Starting run for: {topic}")
    state = _execute_run(cfg)
    if RICH:
        _print(f"\n✅ Run complete: [bold]{state.run_id}[/bold] | status={state.status}")
    else:
        _print(f"Run complete: {state.run_id} | status={state.status}")
    _print(f"  Evidence: {len(state.evidence)} | Hypotheses: {len(state.hypotheses)} | Experiments: {len(state.experiments)}")


@cli.command("resume")
@click.option("--run-id", "-r", required=True, help="Run ID to resume")
@click.option("--config", "-c", default=None)
def resume_cmd(run_id: str, config: Optional[str]) -> None:
    """Resume a paused/failed run from checkpoint."""
    logging.basicConfig(level=logging.INFO)
    cfg = load_config(config)
    cfg.run_id = run_id
    _print(f"Resuming run: {run_id}")
    state = _execute_run(cfg, resume=True)
    _print(f"Resumed run complete: {state.run_id} | status={state.status}")


@cli.command("review")
@click.option("--run-id", "-r", required=True, help="Run ID to review")
def review_cmd(run_id: str) -> None:
    """Review a completed run's results and reflections."""
    from ..core.state import RunState
    state_path = Path("runs") / run_id / "checkpoints"
    if not (state_path / "state.json").exists():
        if RICH:
            _print(f"[red]Run '{run_id}' not found[/red]")
        else:
            _print(f"Run '{run_id}' not found")
        sys.exit(1)
    state = RunState.load(state_path)
    _print(f"\n{'='*60}")
    _print(f"Run ID: {state.run_id}  |  Topic: {state.topic}")
    _print(f"Status: {state.status}  |  Iteration: {state.iteration}")
    _print(f"Evidence items: {len(state.evidence)}")
    _print(f"Hypotheses: {len(state.hypotheses)}")
    _print(f"Experiments: {len(state.experiments)}")
    _print(f"Reviews: {len(state.reviews)}")
    _print(f"Reflections: {len(state.reflections)}")
    if state.policy_updates:
        _print(f"Policy updates applied: {len(state.policy_updates)}")


@cli.command("export")
@click.option("--run-id", "-r", required=True, help="Run ID to export")
@click.option("--output", "-o", default=None, help="Output file path")
def export_cmd(run_id: str, output: Optional[str]) -> None:
    """Export a run's results as a JSON report."""
    from ..core.state import RunState
    state_path = Path("runs") / run_id / "checkpoints"
    if not (state_path / "state.json").exists():
        _print(f"Run '{run_id}' not found")
        sys.exit(1)
    state = RunState.load(state_path)
    report = {
        "run_id": state.run_id,
        "topic": state.topic,
        "created_at": state.created_at,
        "status": state.status,
        "evidence_count": len(state.evidence),
        "hypotheses": state.hypotheses,
        "experiments": state.experiments,
        "reviews": state.reviews,
        "reflections": state.reflections,
        "policy_updates": state.policy_updates,
    }
    out = output or f"report_{run_id}.json"
    Path(out).write_text(json.dumps(report, indent=2))
    _print(f"Report exported to: {out}")


def _execute_run(cfg, resume: bool = False):
    """Internal helper to build and run the agent runtime."""
    from ..core.runtime import AgentRuntime
    from ..agents.planner import ResearchPlannerAgent
    from ..agents.literature_miner import LiteratureMinerAgent
    from ..agents.hypothesis_generator import HypothesisGeneratorAgent
    from ..agents.experiment_coder import ExperimentCoderAgent
    from ..agents.executor import ExecutorAgent
    from ..agents.reviewer import ReviewerAgent
    from ..agents.meta_agent import MetaAgent
    from ..agents.tom_agent import TomAgent

    agents = {
        "planner": ResearchPlannerAgent(cfg),
        "literature_miner": LiteratureMinerAgent(cfg),
        "hypothesis_generator": HypothesisGeneratorAgent(cfg),
        "experiment_coder": ExperimentCoderAgent(cfg),
        "executor": ExecutorAgent(cfg),
        "reviewer": ReviewerAgent(cfg),
        "meta_agent": MetaAgent(cfg),
        "tom_agent": TomAgent(cfg),
    }

    runtime = AgentRuntime(cfg, agents)

    if resume:
        runtime.resume()

    # Build objective graph
    runtime.add_objective(Objective(id="plan", name="Research Planning", priority=10), "planner")
    runtime.add_objective(Objective(id="mine", name="Literature Mining", dependencies=["plan"], priority=9), "literature_miner")
    runtime.add_objective(Objective(id="hypothesize", name="Hypothesis Generation", dependencies=["mine"], priority=8), "hypothesis_generator")
    runtime.add_objective(Objective(id="tom", name="ToM Negotiation", dependencies=["hypothesize"], priority=7), "tom_agent")
    runtime.add_objective(Objective(id="code", name="Experiment Coding", dependencies=["tom"], priority=6), "experiment_coder")
    runtime.add_objective(Objective(id="execute", name="Experiment Execution", dependencies=["code"], priority=5), "executor")
    runtime.add_objective(Objective(id="review", name="Adversarial Review", dependencies=["execute"], priority=4), "reviewer")
    runtime.add_objective(Objective(id="reflect", name="Self-Reflection", dependencies=["review"], priority=3), "meta_agent")

    return runtime.run()
