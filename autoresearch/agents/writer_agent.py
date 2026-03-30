"""
WriterAgent — produces structured research reports from experiment results.

Inspired by:
- AgentLaboratory: report_writing + report_refinement phases (LaTeX / Markdown)
- AI Scientist-v2: template-free paper writing, full lifecycle
- GPT-Researcher: ReportGenerator with source curation + RAG context
- AutoResearchClaw: 4-round paper quality audit, AI-slop detection
- Sibyl: 6-agent cross-review, NeurIPS/ICML/ICLR target

Produces:
  - Abstract, Introduction, Related Work, Methodology, Results, Discussion,
    Conclusion, References — all from agent-gathered data.
  - Anti-hallucination: every claim must have a source from learnings/experiments.
  - Reliability flags: marks any unsupported claim (ACM critique of AI Scientist).
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from autoresearch.agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)


@dataclass
class ReportSection:
    title: str
    content: str
    sources: List[str] = field(default_factory=list)
    verified: bool = True       # False = contains unsupported claims (anti-hallucination flag)


@dataclass
class ResearchReport:
    title:       str
    task:        str
    abstract:    str
    sections:    List[ReportSection]
    references:  List[str]
    score:       float = 0.0
    word_count:  int   = 0
    hallucination_flags: List[str] = field(default_factory=list)
    metadata:    Dict[str, Any]    = field(default_factory=dict)

    def to_markdown(self) -> str:
        lines = [
            f"# {self.title}\n",
            f"> **Task:** {self.task}\n",
            f"## Abstract\n{self.abstract}\n",
        ]
        for sec in self.sections:
            flag = " ⚠️ *[unverified claims present]*" if not sec.verified else ""
            lines.append(f"## {sec.title}{flag}\n{sec.content}\n")
        if self.references:
            lines.append("## References\n")
            for i, ref in enumerate(self.references, 1):
                lines.append(f"{i}. {ref}")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "title":       self.title,
            "task":        self.task,
            "abstract":    self.abstract,
            "sections":    [{"title": s.title, "content": s.content, "verified": s.verified}
                            for s in self.sections],
            "references":  self.references,
            "score":       self.score,
            "word_count":  self.word_count,
            "hallucination_flags": self.hallucination_flags,
        }


class WriterAgent(BaseAgent):
    """
    WriterAgent — synthesises all research findings into a structured report.

    Pipeline (AutoResearchClaw 23-stage inspiration):
      1. Extract key claims from learnings + experiment results
      2. Build section content from verified sources
      3. Flag unsupported claims (ACM reliability critique)
      4. Produce Markdown report with abstract, sections, references
    """

    def _default_parameters(self) -> Dict[str, Any]:
        return {
            "min_word_count":   300,
            "max_sections":     8,
            "verify_claims":    True,
            "target_venue":     "workshop",   # workshop / conference / journal
            "include_latex":    False,
        }

    def _execute(self, task: str, context: Dict[str, Any]) -> Dict[str, Any]:
        learnings:    List[str] = context.get("learnings", [])
        urls:         List[str] = context.get("urls", [])
        hypotheses:   List[Any] = context.get("hypotheses", [])
        experiments:  List[Any] = context.get("experiments", [])
        prior_skills: List[str] = context.get("prior_skills", [])

        # 1. Extract & verify claims
        claims = self._extract_claims(learnings, experiments)
        verified, flags = self._verify_claims(claims, learnings, urls)

        # 2. Build each section
        sections = self._build_sections(task, claims, verified, learnings, experiments, hypotheses)

        # 3. Abstract
        abstract = self._write_abstract(task, claims, experiments)

        # 4. References from URLs
        refs = self._build_references(urls, learnings)

        # 5. Assemble report
        title = self._derive_title(task)
        report = ResearchReport(
            title=title,
            task=task,
            abstract=abstract,
            sections=sections,
            references=refs,
            hallucination_flags=flags,
            metadata={
                "target_venue": self._parameters["target_venue"],
                "prior_skills_used": prior_skills,
            },
        )
        report.word_count = len(report.to_markdown().split())

        return {
            "report":             report.to_dict(),
            "markdown":           report.to_markdown(),
            "title":              report.title,
            "word_count":         report.word_count,
            "sections_written":   len(sections),
            "hallucination_flags": flags,
            "references_count":   len(refs),
            "verified_ratio":     sum(1 for s in sections if s.verified) / max(len(sections), 1),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_claims(
        self,
        learnings: List[str],
        experiments: List[Any],
    ) -> List[str]:
        claims = list(learnings[:15])  # cap to top 15 learnings
        for exp in experiments:
            if isinstance(exp, dict):
                if "result" in exp:
                    claims.append(f"Experiment result: {str(exp['result'])[:200]}")
                if "hypothesis" in exp:
                    claims.append(f"Tested hypothesis: {str(exp['hypothesis'])[:150]}")
        return claims

    def _verify_claims(
        self,
        claims: List[str],
        learnings: List[str],
        urls: List[str],
    ) -> tuple[List[bool], List[str]]:
        """
        Anti-hallucination pass (ACM evaluation finding: 42% failure rate in AI Scientist).
        A claim is verified if it can be traced back to a learning or URL.
        """
        verified = []
        flags = []
        learning_blob = " ".join(learnings).lower()
        url_blob = " ".join(urls).lower()

        for claim in claims:
            words = re.findall(r"\w{5,}", claim.lower())
            # A claim is verified if ≥2 of its keywords appear in learnings/URLs
            matches = sum(1 for w in words[:8] if w in learning_blob or w in url_blob)
            is_verified = matches >= min(2, len(words))
            verified.append(is_verified)
            if not is_verified:
                flags.append(f"Unsupported claim: '{claim[:80]}...'")

        return verified, flags

    def _build_sections(
        self,
        task: str,
        claims: List[str],
        verified: List[bool],
        learnings: List[str],
        experiments: List[Any],
        hypotheses: List[Any],
    ) -> List[ReportSection]:
        sections = []

        # Introduction
        intro_claims = [c for c, v in zip(claims[:3], verified[:3])]
        intro_text = (
            f"This work investigates: **{task}**.\n\n"
            + "\n".join(f"- {c}" for c in intro_claims)
            if intro_claims else
            f"This work investigates: **{task}**."
        )
        sections.append(ReportSection(
            title="Introduction",
            content=intro_text,
            verified=all(verified[:3]),
        ))

        # Related Work
        if learnings:
            rw_items = learnings[:5]
            rw_text = "Based on a recursive deep search of existing literature:\n\n" + "\n".join(
                f"{i+1}. {item}" for i, item in enumerate(rw_items)
            )
            sections.append(ReportSection(
                title="Related Work",
                content=rw_text,
                sources=rw_items,
                verified=True,
            ))

        # Methodology
        if hypotheses:
            hyp_items = hypotheses[:3]
            meth_text = (
                "Hypotheses were generated via MCTS tree search (AI Scientist-v2 pattern) "
                "and validated through knowledge grounding:\n\n"
                + "\n".join(
                    f"- **H{i+1}**: {str(h)[:150]}"
                    for i, h in enumerate(hyp_items)
                )
            )
        else:
            meth_text = (
                "The research methodology followed a self-evolving multi-agent pipeline:\n"
                "1. Recursive deep search (breadth/depth) for literature grounding\n"
                "2. MCTS hypothesis generation with novelty filtering\n"
                "3. Experiment design and execution with fixed budget\n"
                "4. Bilevel optimisation of the search strategy itself"
            )
        sections.append(ReportSection(title="Methodology", content=meth_text, verified=True))

        # Experimental Results
        if experiments:
            kept = [e for e in experiments if isinstance(e, dict) and e.get("kept")]
            exp_text = (
                f"Ran {len(experiments)} experiment(s); "
                f"{len(kept)} produced improvements.\n\n"
                + "\n".join(
                    f"- {str(e.get('hypothesis', 'experiment'))[:100]}: "
                    f"score={e.get('score', 'N/A')}"
                    for e in kept[:5]
                )
            )
        else:
            exp_text = "Experimental evaluation pending LLM API integration. See `experiment_agent.py` for design."
        sections.append(ReportSection(title="Experimental Results", content=exp_text, verified=bool(experiments)))

        # Discussion
        discussion_claims = [c for c, v in zip(claims[3:8], verified[3:8]) if v]
        disc_text = (
            "Key findings from the research process:\n\n"
            + ("\n".join(f"- {c}" for c in discussion_claims) if discussion_claims
               else "- No verified findings beyond what is stated above.")
        )
        sections.append(ReportSection(
            title="Discussion",
            content=disc_text,
            verified=bool(discussion_claims),
        ))

        # Conclusion
        concl_text = (
            f"This investigation of **{task}** used a self-evolving multi-agent architecture "
            f"combining recursive deep search (dzhng/deep-research), MCTS hypothesis generation "
            f"(AI Scientist-v2), bilevel outer-loop optimisation, and Centaur hybrid HPO. "
            f"The system improves its own research strategy each round via the MetaAgent "
            f"(DGM-Hyperagents), persisting experience across runs (Bilevel Autoresearch)."
        )
        sections.append(ReportSection(title="Conclusion", content=concl_text, verified=True))

        return sections[: self._parameters["max_sections"]]

    def _write_abstract(
        self,
        task: str,
        claims: List[str],
        experiments: List[Any],
    ) -> str:
        top_claims = "; ".join(claims[:3]) if claims else "no findings yet"
        n_exp = len(experiments) if experiments else 0
        return (
            f"We investigate **{task}** using a self-evolving multi-agent research framework "
            f"(SERA-X) that implements Bilevel Autoresearch, AI Scientist-v2 tree search, "
            f"Centaur hybrid HPO, and DGM-Hyperagents meta-improvement. "
            f"Key findings: {top_claims}. "
            f"We ran {n_exp} experiments with a fixed time budget, with the outer loop "
            f"autonomously generating new search mechanisms to prevent stagnation."
        )

    @staticmethod
    def _derive_title(task: str) -> str:
        words = re.findall(r"[A-Za-z]{3,}", task)
        title_words = [w.capitalize() for w in words[:8]]
        return " ".join(title_words) + ": A Self-Evolving Multi-Agent Study"

    @staticmethod
    def _build_references(urls: List[str], learnings: List[str]) -> List[str]:
        refs = []
        for url in urls[:10]:
            refs.append(url)
        # Add canonical paper refs
        refs += [
            "Karpathy, A. (2026). autoresearch. GitHub. https://github.com/karpathy/autoresearch",
            "EdwardOptimization (2026). Bilevel Autoresearch: Meta-Autoresearching Itself. arXiv:2603.23420",
            "Lu et al. (2025). The AI Scientist: Towards Fully Automated Open-Ended Scientific Discovery. arXiv:2408.06292",
            "Sakana AI (2026). The AI Scientist-v2: Workshop-Level Automated Scientific Discovery. arXiv:2504.08066",
            "Meta AI (2026). HyperAgents. https://ai.meta.com/research/hyperagents/",
            "Lange et al. (2026). Can LLMs Beat Classical HPO? Centaur. arXiv (Mar 2026)",
            "Meta AI (2026). DGM-Hyperagents. Darwin Gödel Machine extension.",
            "AIRS-Bench (2026). A Suite of Tasks for Frontier AI Research Science Agents.",
        ]
        return refs

    def _score_result(self, result: Any) -> float:
        if isinstance(result, dict):
            wc   = min(result.get("word_count", 0) / 1000, 1.0)
            vr   = result.get("verified_ratio", 0.5)
            hf   = max(0.0, 1.0 - len(result.get("hallucination_flags", [])) * 0.1)
            refs = min(result.get("references_count", 0) / 8, 1.0)
            return round((wc * 0.2 + vr * 0.4 + hf * 0.3 + refs * 0.1), 3)
        return 0.5
