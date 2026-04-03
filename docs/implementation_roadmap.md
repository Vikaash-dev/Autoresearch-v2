# AutoResearch v2 — Implementation Roadmap

## Milestone 0 — Foundation ✅ (this PR)

**Delivered:**
- Full Python package scaffold (`pyproject.toml`, `Dockerfile`, `.devcontainer`)
- Config system (Pydantic v2 + YAML defaults)
- Core runtime: `ObjectiveGraph` (DAG executor), `RunState` (persistent), `CheckpointManager`
- 8-agent layer: Planner, LitMiner, HypothesisGenerator, ExpCoder, Executor, Reviewer, MetaAgent, TomAgent
- Literature: arXiv connector, Semantic Scholar + OpenAlex connector, `EvidenceRegistry`, `CitationGrounding`
- Experiments: `ExperimentSpec`, `ExperimentGenerator`, `SandboxExecutor`, `SelfHealingLoop`
- ToM: 4 reviewer personas, `CollaboratorIntentProfile`, `AdversarialChallengeGenerator`, `EpistemicNegotiationLoop`
- Reflection: `SelfReviewModule`, `SelfImprovementEngine` (safe allowlist)
- CLI: `run`, `resume`, `review`, `export`
- 27 passing tests (unit + E2E smoke)
- Docs: architecture, contracts, roadmap, research review

**Deferred (TODO):**
- Real LLM API integration (currently stubbed via `_llm_prompt`)
- Docker sandbox execution mode (`sandbox: docker`)
- Structured prompt-level self-improvement (Milestone 2)
- Parallelised objective execution (multi-thread/async branches)
- Report export to Markdown/LaTeX/PDF

---

## Milestone 1 — LLM Integration (30 days)

**Goal:** Replace `_llm_prompt` stub with real LLM calls.

### Tasks
- [ ] Implement `LLMClient` abstraction in `autoresearch/llm/`
  - Backends: OpenAI, Anthropic, local Ollama
  - Retry logic, rate limiting, cost tracking
- [ ] Wire `LLMClient` into all 8 agents
- [ ] Implement structured output parsing (JSON mode / Pydantic validators)
- [ ] Implement `HypothesisGeneratorAgent` with real prompt templates
- [ ] Implement `ResearchPlannerAgent` with multi-step chain-of-thought planning
- [ ] Add `--dry-run` CLI flag (no LLM calls, uses mocked responses)
- [ ] Integration tests with mocked LLM responses

---

## Milestone 2 — Enhanced Experiment Execution (60 days)

**Goal:** Robust sandboxed experiment lifecycle.

### Tasks
- [ ] Docker sandbox backend (`sandbox: docker`)
  - Mount artifacts directory
  - Resource limits (CPU/memory/time)
  - Container cleanup
- [ ] GPU detection and experiment resource allocation
- [ ] Artifact storage: save stdout logs, generated figures, model checkpoints
- [ ] Metrics extraction: parse JSON from stdout, support custom metric schemas
- [ ] Provenance tracking: hash experiment inputs/outputs for reproducibility
- [ ] Extended `SelfHealingLoop`:
  - LLM-assisted code repair for semantic errors
  - Dependency auto-install in virtualenv
  - Multi-language support (Julia, R stubs)

---

## Milestone 3 — Advanced ToM & Adversarial Review (90 days)

**Goal:** Production-quality epistemic negotiation.

### Tasks
- [ ] LLM-backed reviewer persona simulation
  - Fine-tuned on actual NeurIPS/ICLR review corpora
  - Dynamic persona creation based on topic domain
- [ ] Multi-round adversarial review loop
  - Author agent responds to reviewer challenges
  - Iterative manuscript revision tracking
- [ ] Formal belief-state tracking (Bayesian belief updates)
- [ ] Belief-Robustness Score: claim survives N heterogeneous model evaluations
- [ ] Adversarial Persuasion Gap metric
- [ ] Minority-report dashboard in `review` CLI command

---

## Milestone 4 — Self-Improvement & Cross-Run Learning (120 days)

**Goal:** The system improves measurably across research runs.

### Tasks
- [ ] Skill Registry: encode "lessons learned" per module as structured skills
  - Stored in SQLite or JSON lines
  - Injected into future run prompts when similar failures are detected
- [ ] Run-comparison analytics: compare quality scores across runs
- [ ] Prompt-level self-improvement (within sandboxed evaluation)
  - Propose new prompt templates
  - Evaluate on hold-out test topics before applying
- [ ] Policy update validation via A/B simulation (two parallel runs with/without update)
- [ ] DGM-H (Darwin-Gödel Machine) style meta-improvement: the system can propose changes
  to its own objective graph structure

---

## Milestone 5 — Production & Scale (180 days)

**Goal:** Production-ready deployment and ecosystem integrations.

### Tasks
- [ ] Async/parallel objective execution in `AgentRuntime`
- [ ] Distributed run support (Celery / Ray)
- [ ] Web UI / dashboard (FastAPI + React or Streamlit)
- [ ] Cloud Lab integration (Emerald Cloud Lab API for physical experiments)
- [ ] Report export: Markdown → LaTeX → PDF pipeline
- [ ] OpenClaw-style messaging integration (Discord/Telegram bot)
- [ ] Full CI/CD: linting, type checking, coverage gates
- [ ] Public benchmark: AutoResearch v2 vs. Sakana AI Scientist on shared tasks

---

## Known Technical Debt

| Area | Issue | Priority |
|---|---|---|
| `_llm_prompt` | Returns stub string, not real LLM response | High (Milestone 1) |
| `SandboxExecutor` | Only subprocess mode implemented | Medium (Milestone 2) |
| `ArxivConnector` | Synchronous HTTP with `urllib` | Low (use `httpx` async) |
| `ObjectiveGraph` | Sequential execution only | Medium (Milestone 5) |
| `TomAgent` | Belief updates are heuristic, not Bayesian | Medium (Milestone 3) |
| Config | No validation of `run_id` format | Low |
