"""
Writer Agent — generates LaTeX paper drafts from experiment results.

Uses Reviewer-ToM to preemptively address reviewer concerns before writing.
Uses Community-ToM to correctly frame contributions relative to the field.

Based on:
  - AI-Scientist v2 perform_icbinb_writeup.py
  - NousResearch/autonovel multi-phase revision pipeline
  - Draft → Opus Review Loop → targeted revision cycle
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from agents.base_agent import BaseAgent
from core.blackboard import Blackboard, ExperimentNode
from tom.engine import TheoryOfMindEngine, ReviewerModel, CommunityModel, TaskDecomposition

logger = logging.getLogger(__name__)

_WRITER_SYSTEM = """You are a scientific writer producing clear, rigorous ML research papers.
Your writing should be:
- Precise and technical but readable
- Honest about limitations
- Well-structured with a clear narrative
- Properly motivating the problem and contribution
- Supported by the actual experimental results provided
"""


class WriterAgent(BaseAgent):
    """Generates paper drafts using experiment results and ToM guidance."""

    def __init__(
        self,
        blackboard: Blackboard,
        tom_engine: TheoryOfMindEngine,
        llm_fn: Any = None,
    ) -> None:
        super().__init__("writer", blackboard, tom_engine, llm_fn)

    def act(self, context: dict[str, Any]) -> dict[str, Any]:
        """
        Generate a paper draft.

        Context keys:
            best_node: ExperimentNode
            idea: dict
            task: TaskDecomposition
            reviewer_model: ReviewerModel
            community_model: CommunityModel
            output_dir: str
            review_feedback: dict (optional, for revisions)
            is_revision: bool
        """
        best_node: ExperimentNode | None = context.get("best_node")
        idea: dict[str, Any] = context.get("idea", {})
        reviewer: ReviewerModel = context.get("reviewer_model") or self._tom.reviewer_model
        community: CommunityModel = context.get("community_model") or self._tom.community_model
        review_feedback: dict[str, Any] | None = context.get("review_feedback")
        is_revision: bool = context.get("is_revision", False)
        output_dir = Path(context.get("output_dir", "experiments/output"))
        output_dir.mkdir(parents=True, exist_ok=True)

        # Gather all successful experiment nodes for results section
        all_nodes = self._bb.get_all_nodes()
        successful_nodes = [n for n in all_nodes if n.status == "success"]
        results_summary = self._summarize_results(successful_nodes, best_node)

        if is_revision and review_feedback:
            draft_text = self._write_revision(
                idea, results_summary, reviewer, review_feedback, output_dir
            )
        else:
            draft_text = self._write_draft(
                idea, results_summary, reviewer, community, output_dir
            )

        if not draft_text:
            self.record_failure("write_draft", "Empty draft generated")
            return {"status": "failed", "error": "Draft generation failed"}

        # Save draft
        draft_path = output_dir / "draft.md"
        draft_path.write_text(draft_text)
        self.record_success("write_draft")
        self.update_own_belief(last_action="draft_written")

        return {"status": "ok", "draft_text": draft_text, "draft_path": str(draft_path)}

    # ------------------------------------------------------------------ #
    #  Draft generation                                                    #
    # ------------------------------------------------------------------ #

    def _write_draft(
        self,
        idea: dict[str, Any],
        results_summary: str,
        reviewer: ReviewerModel,
        community: CommunityModel,
        output_dir: Path,
    ) -> str:
        """Generate the initial paper draft with proactive reviewer-ToM guidance."""
        prompt = (
            "Write a 4-page scientific paper (ICBINB format) based on these results.\n\n"
            f"Research idea:\n{json.dumps(idea, indent=2)}\n\n"
            f"Experimental results:\n{results_summary}\n\n"
            "Reviewer profile (preemptively address these):\n"
            f"  Hot topics to connect to: {reviewer.hot_topics}\n"
            f"  Pet peeves to avoid: {reviewer.pet_peeves}\n"
            f"  Required sections: {reviewer.required_sections}\n"
            f"  Common rejection reasons: {reviewer.likely_rejection_reasons}\n\n"
            "Community context:\n"
            f"  Active debates to address: {community.active_debates}\n"
            f"  High-impact areas to connect to: {community.high_impact_areas}\n\n"
            "Structure: Abstract, Introduction, Related Work, Method, "
            "Experiments, Conclusion, Limitations.\n"
            "Write in Markdown. Be concrete — include actual numbers from results."
        )

        if self._llm is None:
            return self._placeholder_draft(idea, results_summary)

        try:
            return self.call_llm(prompt, system=_WRITER_SYSTEM)
        except Exception as exc:
            logger.error("Draft generation failed: %s", exc)
            return ""

    def _write_revision(
        self,
        idea: dict[str, Any],
        results_summary: str,
        reviewer: ReviewerModel,
        review_feedback: dict[str, Any],
        output_dir: Path,
    ) -> str:
        """Revise draft based on adversarial review feedback."""
        prompt = (
            "Revise the paper draft to address the following review feedback.\n\n"
            f"Original research: {idea.get('Title', '')}\n"
            f"Results: {results_summary[:500]}\n\n"
            "Review feedback:\n"
            f"  Score: {review_feedback.get('predicted_score', '?')}/10\n"
            f"  Major concerns: {review_feedback.get('major_concerns', [])}\n"
            f"  Required changes: {review_feedback.get('required_changes', [])}\n"
            f"  Reasoning: {review_feedback.get('reasoning', '')}\n\n"
            "Write the complete revised paper in Markdown. "
            "Address every major concern directly. "
            "Add the required ablations/baselines referenced in the review."
        )

        if self._llm is None:
            return self._placeholder_draft(idea, results_summary, is_revision=True)

        try:
            return self.call_llm(prompt, system=_WRITER_SYSTEM)
        except Exception as exc:
            logger.error("Revision generation failed: %s", exc)
            return ""

    # ------------------------------------------------------------------ #
    #  Helpers                                                             #
    # ------------------------------------------------------------------ #

    def _summarize_results(
        self, nodes: list[ExperimentNode], best_node: ExperimentNode | None
    ) -> str:
        lines = []
        if best_node:
            lines.append(f"Best result: metric={best_node.metric:.4f} ({best_node.hypothesis[:80]})")
        lines.append(f"Total experiments: {len(nodes)} successful")
        for n in sorted(nodes, key=lambda x: x.metric or 0, reverse=True)[:5]:
            lines.append(f"  - node={n.node_id}: metric={n.metric:.4f} depth={n.depth}")
        return "\n".join(lines)

    def _placeholder_draft(
        self, idea: dict[str, Any], results: str, is_revision: bool = False
    ) -> str:
        revision_note = " (Revised)" if is_revision else ""
        return (
            f"# {idea.get('Title', 'Research Paper')}{revision_note}\n\n"
            f"## Abstract\n{idea.get('Abstract', 'TBD')}\n\n"
            f"## Experimental Results\n{results}\n\n"
            "## Conclusion\nFurther work required.\n"
        )
