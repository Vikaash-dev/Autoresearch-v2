"""
Autoresearch-v2 — main entry point.

Usage:
    python launch.py --idea "my_idea.json" --model gpt-4o --venue NeurIPS

Inspired by:
  - SakanaAI/AI-Scientist v2 launch_scientist_bfts.py
  - karpathy/autoresearch (simple CLI, agent reads program.md)
  - NousResearch/hermes-agent (single command, handles everything)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("autoresearch_v2")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Autoresearch-v2: self-evolving autonomous research system"
    )
    parser.add_argument(
        "--idea",
        type=str,
        default=None,
        help="Path to a JSON file containing a research idea, or a plain text description",
    )
    parser.add_argument(
        "--idea-text",
        type=str,
        default=None,
        help="Inline research idea description (alternative to --idea file)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o",
        help="LLM to use for experimentation",
    )
    parser.add_argument(
        "--model-writeup",
        type=str,
        default="o1-preview-2024-09-12",
        help="LLM to use for paper writing",
    )
    parser.add_argument(
        "--model-review",
        type=str,
        default="gpt-4o",
        help="LLM to use for adversarial review",
    )
    parser.add_argument(
        "--venue",
        type=str,
        default="NeurIPS",
        help="Target conference venue (for Reviewer-ToM)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for experiment output",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to configuration YAML",
    )
    parser.add_argument(
        "--max-nodes",
        type=int,
        default=None,
        help="Override max_nodes in config",
    )
    parser.add_argument(
        "--skip-writeup",
        action="store_true",
        help="Skip the paper writing stage",
    )
    parser.add_argument(
        "--skip-evolution",
        action="store_true",
        help="Skip the self-evolution stage",
    )
    parser.add_argument(
        "--user-description",
        type=str,
        default="",
        help="Description of who you are and what you want (feeds User-ToM)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run pipeline with dummy LLM (no API calls) for testing",
    )
    return parser.parse_args()


def load_idea(args: argparse.Namespace) -> dict:
    """Load idea from file or inline text."""
    if args.idea_text:
        return {
            "Title": args.idea_text[:80],
            "Abstract": args.idea_text,
            "Keywords": "",
            "Hypothesis": args.idea_text,
        }

    if args.idea and Path(args.idea).exists():
        with open(args.idea) as f:
            data = json.load(f)
        if isinstance(data, list):
            return data[0]
        return data

    if args.idea:
        # Treat as inline text
        return {
            "Title": args.idea[:80],
            "Abstract": args.idea,
            "Keywords": "",
            "Hypothesis": args.idea,
        }

    logger.error("No idea provided. Use --idea or --idea-text.")
    sys.exit(1)


def build_llm_fn(model: str, dry_run: bool = False):
    """Build an LLM callable from model name."""
    if dry_run:
        logger.info("Dry run: using dummy LLM (no API calls)")
        def dummy_llm(prompt: str) -> str:
            return json.dumps({
                "hypothesis": "Test hypothesis",
                "novelty_argument": "Novel",
                "test_procedure": "Run experiment",
                "expected_outcome": "Better metric",
                "stated_goal": "Improve performance",
                "implicit_goal": "Publish at venue",
                "implicit_constraints": [],
                "implicit_audience": "ML researchers",
                "time_horizon": "hours",
                "success_criteria": ["metric > 0.8"],
                "likely_pitfalls": [],
            })
        return dummy_llm

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        logger.warning("OPENAI_API_KEY not set — LLM calls will fail")

    try:
        from openai import OpenAI
        client = OpenAI()

        def openai_llm(prompt: str) -> str:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.9,
            )
            return response.choices[0].message.content or ""

        return openai_llm

    except ImportError:
        logger.warning("openai package not installed — using dummy LLM")
        return build_llm_fn(model, dry_run=True)


def main() -> None:
    args = parse_args()
    idea = load_idea(args)

    # Set up output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = idea.get("Title", "run")[:40].replace(" ", "_")
    output_dir = Path(args.output_dir) if args.output_dir else Path(f"experiments/{timestamp}_{run_name}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Log to file as well as console
    fh = logging.FileHandler(output_dir / "run.log")
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    logging.getLogger().addHandler(fh)

    logger.info("Autoresearch-v2 starting")
    logger.info("Idea: %s", idea.get("Title", "Unknown"))
    logger.info("Output: %s", output_dir)

    # Import here to avoid circular imports at module level
    from core.blackboard import Blackboard
    from core.tree_search import BFTSConfig
    from core.loop import LoopConfig
    from tom.engine import TheoryOfMindEngine
    from agents.orchestrator import Orchestrator

    # Build components
    bb = Blackboard(persist_path=output_dir / "blackboard.json")
    tom = TheoryOfMindEngine(
        llm_fn=build_llm_fn(args.model, dry_run=args.dry_run),
        persist_path=output_dir / "tom_state.json",
    )

    bfts_cfg = BFTSConfig()
    if args.max_nodes:
        bfts_cfg.max_nodes = args.max_nodes
    if args.dry_run:
        bfts_cfg.num_seeds = 2
        bfts_cfg.max_nodes = 6

    orchestrator = Orchestrator(
        blackboard=bb,
        tom_engine=tom,
        llm_fn=build_llm_fn(args.model, dry_run=args.dry_run),
        bfts_config=bfts_cfg,
        loop_config=LoopConfig(),
        output_dir=output_dir,
    )

    result_dir = orchestrator.run(
        idea=idea,
        user_description=args.user_description,
        venue=args.venue,
    )

    if result_dir:
        logger.info("Run complete. Results in: %s", result_dir)
    else:
        logger.error("Run failed — check logs in %s", output_dir)
        sys.exit(1)


if __name__ == "__main__":
    main()
