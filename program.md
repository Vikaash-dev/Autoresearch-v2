# Autoresearch v2 — Research Organisation Program

> This file is your **research org charter** — inspired by Karpathy's autoresearch.
> The human edits this to guide the research direction.
> The MetaAgent reads this every round and updates `strategy.md` based on results.

## Mission
Autonomously research any given task, run experiments, and improve the research
strategy each round until a high-quality report and validated findings emerge.

## Research Organisation Structure

### Roles
- **ResearchDirector** — reads `program.md` + `strategy.md`, coordinates all agents, decides priorities
- **LiteratureAgent** — recursive deep search (breadth/depth), extracts learnings and follow-up questions
- **HypothesisAgent** — MCTS tree search over hypothesis space, generates ranked candidates
- **KnowledgeAgent** — validates hypotheses for novelty, grounding, and consistency
- **ExperimentAgent** — designs and (mock-)runs experiments, tracks metrics
- **WriterAgent** — synthesises findings into a structured research report
- **ReviewerAgent** — peer-reviews the report, scores against AIRS-Bench criteria
- **MetaAgent** — reads all agent scores, proposes improvements to `strategy.md` and agent parameters

### Workflow per Round
1. LiteratureAgent searches for papers/repos relevant to the task
2. HypothesisAgent generates hypotheses from search results
3. KnowledgeAgent filters to novel, grounded, consistent hypotheses
4. ExperimentAgent designs and runs experiments
5. WriterAgent writes the research report
6. ReviewerAgent scores the report
7. MetaAgent updates `strategy.md` if score improved

## Constraints
- Each agent has a timeout (configurable in `config.py`)
- Failed experiments are **recycled as training signal** (Bilevel Autoresearch)
- Evolved agent versions are **archived with scores** (HyperAgents archive)
- Skills learned from experience are **persisted** across runs (Hermes-style)

## Notes for Agents
- Prefer building on existing findings from `memory/` before starting fresh
- Cite the source paper/repo for every idea you use
- A lower review score this round is better than a hallucinated high score
- `strategy.md` is the live memory of what works — read it before acting
