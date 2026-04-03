# AutoResearch v2 — Setup Guide

## Prerequisites

- Python 3.10 or newer
- pip (or uv / poetry)

---

## Quick Start

### 1. Install

```bash
git clone https://github.com/Vikaash-dev/Autoresearch-v2.git
cd Autoresearch-v2
pip install -e ".[dev]"
```

### 2. Run a research session

```bash
autoresearch run --topic "attention mechanisms in large language models" --verbose
```

This will:
1. Create a research plan
2. Mine arXiv for relevant papers
3. Generate hypotheses
4. Run the Theory-of-Mind negotiation loop
5. Generate and execute experiment code
6. Perform adversarial review
7. Run self-reflection and propose policy updates

The run state is checkpointed after each objective to `runs/{run_id}/checkpoints/state.json`.

### 3. Resume a paused run

```bash
autoresearch resume --run-id <run_id>
```

### 4. Review a completed run

```bash
autoresearch review --run-id <run_id>
```

### 5. Export a report

```bash
autoresearch export --run-id <run_id> --output my_report.json
```

---

## Configuration

Create a `config.yaml` to override defaults:

```yaml
topic: "transformer architectures for protein folding"
max_iterations: 5
llm:
  provider: openai
  model: gpt-4o
  api_key_env: OPENAI_API_KEY  # set this env var
literature:
  arxiv_max_results: 30
  scholarly_backend: semantic_scholar
  semantic_scholar_api_key_env: SEMANTIC_SCHOLAR_API_KEY
experiment:
  sandbox: subprocess
  max_retries: 5
  timeout_seconds: 600
tom:
  enabled: true
  adversarial_rounds: 3
  reviewer_personas:
    - methodological_purist
    - empirical_skeptic
    - novelty_maximizer
    - benchmark_enforcer
reflection:
  enabled: true
  safe_modification_only: true
```

Then run with:

```bash
autoresearch run --topic "..." --config config.yaml
```

---

## Docker

### Build and run

```bash
docker build -t autoresearch-v2 .
docker run -e OPENAI_API_KEY=sk-... -v $(pwd)/runs:/app/runs \
  autoresearch-v2 run --topic "quantum ML"
```

---

## Dev Container (VS Code)

Open the repo in VS Code with the Dev Containers extension. The `.devcontainer/devcontainer.json`
will automatically install all dev dependencies.

---

## Running Tests

```bash
pytest tests/ -v
```

Expected output: 27 passed.

For coverage:

```bash
pytest tests/ --cov=autoresearch --cov-report=term-missing
```

---

## Environment Variables

| Variable | Description | Required |
|---|---|---|
| `OPENAI_API_KEY` | OpenAI API key for LLM calls | For production runs |
| `SEMANTIC_SCHOLAR_API_KEY` | Semantic Scholar API key | Optional (rate limit) |

> **Note:** Without an LLM API key, the agents use a stub `_llm_prompt()` that returns
> placeholder text. The pipeline still runs end-to-end for testing and development.

---

## Project Structure

```
autoresearch/
├── agents/       # All 8 agent implementations
├── cli/          # CLI entrypoints
├── config/       # Pydantic config schema + YAML defaults
├── core/         # Runtime, ObjectiveGraph, RunState, Checkpointing
├── experiments/  # Spec, Generator, Sandbox executor, SelfHealingLoop
├── literature/   # arXiv + Scholarly connectors, EvidenceRegistry, CitationGrounding
├── reflection/   # SelfReviewModule, SelfImprovementEngine
└── tom/          # Reviewer personas, Intent, Adversarial, Negotiation

tests/            # 27 unit + E2E tests
docs/             # Architecture, contracts, roadmap, research review
Dockerfile
.devcontainer/
pyproject.toml
```

---

## Key CLI Commands

```
autoresearch --help
autoresearch run --help
autoresearch resume --help
autoresearch review --help
autoresearch export --help
```
