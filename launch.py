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
    parser.add_argument(
        "--tavily-keys",
        type=str,
        default=None,
        help="Comma-separated Tavily API keys (supplements env TAVILY_API_KEY[_N])",
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
        logger.info("Dry run: using context-aware stub LLM (no API calls)")

        def _dry_run_llm(prompt: str) -> str:
            """
            Return a contextually appropriate stub response based on the prompt.
            Each agent type asks structurally different questions — detect them by
            keywords so the pipeline flows end-to-end without any real LLM calls.
            """
            p = prompt.lower()

            # HypothesisAgent: expand() or act() asking for hypotheses
            if "hypothesis" in p or "novel" in p or "falsifiable" in p:
                return json.dumps([
                    {
                        "hypothesis": "Reduce attention complexity from O(n²) to O(n log n) via sparse patterns",
                        "code_patch_description": "Replace dense attention with sparse block-diagonal variant",
                        "expected_improvement": "15% speedup on sequences >512 tokens",
                    },
                    {
                        "hypothesis": "Layer-norm placement before residual connection improves gradient flow",
                        "code_patch_description": "Move LayerNorm before the residual add in each transformer block",
                        "expected_improvement": "0.5–1% validation accuracy gain",
                    },
                    {
                        "hypothesis": "Mixed-precision training with dynamic loss scaling reduces memory by 40%",
                        "code_patch_description": "Wrap model with torch.cuda.amp.autocast and GradScaler",
                        "expected_improvement": "40% memory reduction, 20% throughput gain",
                    },
                ])

            # TheoryOfMindEngine: infer_user_intent
            if "decompose" in p or "implicit_goal" in p or "success_criteria" in p:
                return json.dumps({
                    "stated_goal": "Improve model performance",
                    "implicit_goal": "Achieve a publishable improvement over baseline",
                    "implicit_constraints": ["limited GPU budget", "must complete in one session"],
                    "implicit_audience": "ML conference reviewers",
                    "time_horizon": "hours",
                    "success_criteria": ["metric improves by > 1%", "experiment reproduces"],
                    "likely_pitfalls": ["overfitting on small dataset", "evaluation data leakage"],
                })

            # TheoryOfMindEngine: update_user_model
            if "research_style" in p or "trust_level" in p or "depth_preference" in p:
                return json.dumps({
                    "research_style": "rigorous",
                    "depth_preference": "deep",
                    "domain_vocabulary": ["transformer", "attention", "fine-tuning"],
                    "trust_level": 0.7,
                    "preferred_output_format": "icbinb",
                    "known_dislikes": ["overclaiming", "missing baselines"],
                    "session_history_summary": "Focused on efficiency improvements for transformers.",
                })

            # TheoryOfMindEngine: build_reviewer_model / community_model
            if "hot_topics" in p or "pet_peeves" in p or "reviewer" in p:
                return json.dumps({
                    "hot_topics": ["scaling laws", "emergent abilities", "efficiency"],
                    "pet_peeves": ["overclaiming", "missing ablations", "weak baselines"],
                    "required_sections": ["abstract", "introduction", "related_work",
                                          "method", "experiments", "conclusion"],
                    "likely_rejection_reasons": ["incremental over prior work", "missing baselines"],
                })

            # TheoryOfMindEngine: update_community_model
            if "consensus_beliefs" in p or "active_debate" in p or "open_problems" in p:
                return json.dumps({
                    "consensus_beliefs": ["larger models generally perform better"],
                    "active_debates": ["whether scale alone is sufficient for reasoning"],
                    "open_problems": ["efficient long-context modeling", "compositional generalization"],
                    "recent_pivots": ["shift from supervised to RLHF-based training"],
                    "overcrowded_areas": ["LoRA variants", "instruction tuning"],
                    "high_impact_areas": ["long-context efficiency", "multi-modal reasoning"],
                })

            # TheoryOfMindEngine: analyze_self / predict_review
            if "predict" in p and "review" in p:
                return json.dumps({
                    "technical_score": 6,
                    "novelty_score": 7,
                    "predicted_score": 6,
                    "accept_probability": 0.45,
                    "would_accept": False,
                    "major_concerns": ["missing comparison to recent baselines"],
                    "required_changes": ["add ablation on dataset size", "report variance across seeds"],
                    "strengths": ["clear motivation", "well-structured experiments"],
                    "reasoning": "Solid work but needs stronger empirical evaluation.",
                })

            # LiteratureAgent: summarize_for_hypothesis
            if "open problems" in p or "overcrowded" in p or "under-explored" in p:
                return (
                    "Open problems: efficient attention for long sequences remains unsolved.\n"
                    "Overcrowded: standard LoRA variants — avoid.\n"
                    "Promising: sparse attention + mixture-of-experts combinations.\n"
                    "Active debate: whether RLHF or DPO generalises better cross-domain."
                )

            # Default: return a generic helpful string
            return (
                "Dry-run response: no real LLM configured. "
                "Set OPENAI_API_KEY and remove --dry-run to enable real generation."
            )

        return _dry_run_llm

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

    # Collect Tavily keys: env vars + --tavily-keys CLI + config file
    tavily_keys: list[str] = []
    if args.tavily_keys:
        tavily_keys.extend(k.strip() for k in args.tavily_keys.split(",") if k.strip())

    # Also read from config.yaml if present
    config_path = Path(args.config)
    if config_path.exists():
        try:
            import yaml
            with open(config_path) as f:
                cfg = yaml.safe_load(f) or {}
            tavily_keys.extend(cfg.get("tavily", {}).get("api_keys", []))
        except Exception as exc:
            logger.debug("Could not load config.yaml for Tavily keys: %s", exc)

    orchestrator = Orchestrator(
        blackboard=bb,
        tom_engine=tom,
        llm_fn=build_llm_fn(args.model, dry_run=args.dry_run),
        bfts_config=bfts_cfg,
        loop_config=LoopConfig(),
        output_dir=output_dir,
        tavily_keys=tavily_keys or None,
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
