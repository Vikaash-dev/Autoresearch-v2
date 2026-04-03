# AutoResearch v2

**A Self-Evolving, Multi-Agent Autonomous Research Framework**

AutoResearch v2 moves beyond linear-pipeline automation toward a *Recursive Discovery Engine*
that executes research as a directed objective graph, integrates Theory-of-Mind (ToM) modeling,
and safely improves its own policies after each run.

## Features

- **8-agent runtime** — Planner, Literature Miner, Hypothesis Generator, Experiment Coder,
  Executor/Debugger, Adversarial Reviewer, Meta-Agent, and ToM Modeling Agent
- **Directed objective graph** — DAG-based execution, not fixed stages; configurable priorities
  and stop criteria
- **Persistent checkpoints** — resume any run from the last completed objective
- **Literature grounding** — arXiv + Semantic Scholar/OpenAlex connectors, `EvidenceRegistry`,
  and `CitationGrounding` to verify every claim
- **Self-healing experiments** — detect execution failures, auto-repair code, retry with
  bounded attempts; capture artifacts and provenance
- **Theory-of-Mind layer** — 4 reviewer personas, collaborator intent profiling, adversarial
  challenge generation, epistemic negotiation loop, minority-report preservation
- **Safe self-improvement** — post-run bottleneck detection, policy update proposals applied
  only within a safe-parameter allowlist
- **CLI** — `run`, `resume`, `review`, `export`
- **Docker + devcontainer** support

## Quick Start

```bash
pip install -e ".[dev]"
autoresearch run --topic "attention mechanisms in LLMs" --verbose
```

See [docs/setup.md](docs/setup.md) for full setup instructions.

## Documentation

| Document | Description |
|---|---|
| [docs/architecture.md](docs/architecture.md) | System architecture and component map |
| [docs/component_contracts.md](docs/component_contracts.md) | API contracts for every module |
| [docs/implementation_roadmap.md](docs/implementation_roadmap.md) | What's done, what's next |
| [docs/research_review_2026.md](docs/research_review_2026.md) | 2026 arXiv + ecosystem review |
| [docs/setup.md](docs/setup.md) | Installation and usage |

## Tests

```bash
pytest tests/ -v   # 27 tests, all passing
```

## Architecture Overview

```
CLI → AgentRuntime → ObjectiveGraph (DAG)
                    ├── ResearchPlannerAgent
                    ├── LiteratureMinerAgent   (arXiv + Scholarly)
                    ├── HypothesisGeneratorAgent
                    ├── TomAgent               (ToM negotiation)
                    ├── ExperimentCoderAgent
                    ├── ExecutorAgent          (sandbox + self-healer)
                    ├── ReviewerAgent          (adversarial personas)
                    └── MetaAgent              (self-reflection + policy updates)
```

## Research Foundation

Built on insights from:
- Sakana AI Scientist v1 & v2 (arXiv:2408.06292, 2502.09601)
- Meta HyperAgents / Darwin-Gödel Machine
- AutoResearchClaw (aiming-lab)
- Theory-of-Mind in LLM agents (2025–2026 literature)
- karpathy/autoresearch, WecoAI/awesome-autoresearch

See [docs/research_review_2026.md](docs/research_review_2026.md) for the full review.
