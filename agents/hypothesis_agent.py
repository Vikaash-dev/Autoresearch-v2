"""
Hypothesis Agent — generates and expands research hypotheses.

Implements the BFTS expand_fn:
  Given a parent ExperimentNode on the Blackboard, generate N child hypotheses
  that are targeted to resolve the parent's uncertainty or build on its success.

Uses Community-ToM to avoid overcrowded areas.
Uses Agent-ToM to avoid proposing what other agents already tried.
"""

from __future__ import annotations

import json
import logging
import uuid
from typing import Any

from agents.base_agent import BaseAgent
from core.blackboard import Blackboard, ExperimentNode
from tom.engine import TheoryOfMindEngine

logger = logging.getLogger(__name__)

_HYPOTHESIS_SYSTEM = """You are a creative research scientist generating novel hypotheses.
You have access to:
- The current research idea and its goals
- The community knowledge model (what is known, debated, overcrowded)
- The experiment tree so far (what has been tried and what failed/succeeded)

Generate hypotheses that:
1. Build directly on successful parent experiments
2. Target active debates or open problems in the community model
3. Avoid overcrowded areas and known failure patterns
4. Are falsifiable and experimentally tractable
5. Are novel relative to the literature
"""


class HypothesisAgent(BaseAgent):
    """Generates research hypotheses for the BFTS tree."""

    def __init__(self, blackboard: Blackboard, tom_engine: TheoryOfMindEngine,
                 llm_fn: Any = None) -> None:
        super().__init__("hypothesis", blackboard, tom_engine, llm_fn)

    def act(self, context: dict[str, Any]) -> dict[str, Any]:
        """Generate a single root hypothesis (used for seeding)."""
        idea = context.get("idea", {})
        community = self._tom.community_model

        prompt = (
            f"Research idea: {json.dumps(idea, indent=2)}\n\n"
            f"Community knowledge:\n"
            f"  Consensus: {community.consensus_beliefs}\n"
            f"  Open problems: {community.open_problems}\n"
            f"  Overcrowded: {community.overcrowded_areas}\n\n"
            "Generate a specific, testable hypothesis for this research idea. "
            "State: (1) the hypothesis, (2) why it's novel, (3) how to test it, "
            "(4) expected outcome if true. Be concrete and specific.\n"
            "Return JSON: {hypothesis, novelty_argument, test_procedure, expected_outcome}"
        )
        try:
            raw = self.call_llm(prompt, system=_HYPOTHESIS_SYSTEM)
            data = json.loads(raw.strip().lstrip("```json").rstrip("```"))
        except Exception:
            data = {
                "hypothesis": idea.get("Hypothesis", "Improve baseline performance"),
                "novelty_argument": "Based on research idea",
                "test_procedure": "Run experiment",
                "expected_outcome": "Improved metric",
            }

        self.update_own_belief(
            last_action="generate_root_hypothesis",
            confidence_domain=idea.get("Keywords", ""),
        )
        self.record_success("generate_root_hypothesis")
        return {"status": "ok", "hypothesis": data}

    def expand(self, parent: ExperimentNode, blackboard: Blackboard) -> list[ExperimentNode]:
        """
        BFTS expand_fn: generate child hypothesis nodes from a successful parent.
        Called by the BestFirstTreeSearch tree expander.
        """
        # Agent-to-Agent ToM: see what other agents have already tried
        other_agents = self.observe_other_agents()
        known_failures = self.get_recent_failures_from_board()
        community = self._tom.community_model

        prompt = (
            f"Parent experiment:\n"
            f"  Hypothesis: {parent.hypothesis}\n"
            f"  Metric: {parent.metric}\n"
            f"  Stage: {parent.stage}\n\n"
            f"Already-tried failures (avoid these):\n{json.dumps(known_failures[:5])}\n\n"
            f"Community model:\n"
            f"  Active debates: {community.active_debates}\n"
            f"  High impact areas: {community.high_impact_areas}\n"
            f"  Overcrowded: {community.overcrowded_areas}\n\n"
            "Generate 3 child hypotheses that improve on or build from this parent. "
            "Each should explore a different direction. "
            "Return JSON array of 3 objects: "
            "[{hypothesis, code_patch_description, expected_improvement}]"
        )
        try:
            raw = self.call_llm(prompt, system=_HYPOTHESIS_SYSTEM)
            raw = raw.strip().lstrip("```json").rstrip("```").strip()
            children_data = json.loads(raw)
        except Exception as exc:
            logger.warning("HypothesisAgent.expand failed: %s", exc)
            self.record_failure("expand", str(exc))
            return []

        nodes = []
        for i, child_data in enumerate(children_data[:3]):
            node = ExperimentNode(
                node_id=f"{parent.node_id}_h{i}_{str(uuid.uuid4())[:6]}",
                parent_id=parent.node_id,
                depth=parent.depth + 1,
                hypothesis=child_data.get("hypothesis", ""),
                code_patch=child_data.get("code_patch_description", ""),
                stage=min(parent.stage + 1, 4),
            )
            nodes.append(node)

        self.update_own_belief(
            last_action=f"expanded_node_{parent.node_id}",
            current_uncertainty=0.6,
        )
        self.record_success("expand")
        return nodes
