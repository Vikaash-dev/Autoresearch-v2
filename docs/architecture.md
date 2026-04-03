# AutoResearch v2 — Architecture Overview

## 1. Philosophy

AutoResearch v2 is a **self-evolving, multi-agent autonomous research framework**. It moves
beyond linear-pipeline automation toward a *Recursive Discovery Engine* that:

- Executes research as a **directed objective graph** rather than a fixed sequence of stages.
- Uses **Theory-of-Mind (ToM)** modeling to make its outputs persuasive and epistemically
  robust across multiple reviewer perspectives.
- Supports **safe self-improvement** by proposing and applying bounded policy updates after
  each run.
- Maintains **persistent, checkpointed state** so every run can be paused and resumed.

---

## 2. High-Level Component Map

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        CLI  (autoresearch.cli)                          │
│   run | resume | review | export                                        │
└───────────────────────────┬─────────────────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────────────────┐
│                     AgentRuntime  (autoresearch.core.runtime)            │
│  ┌──────────────────┐   ┌──────────────────┐   ┌────────────────────┐  │
│  │  ObjectiveGraph  │   │    RunState       │   │ CheckpointManager  │  │
│  │  (DAG executor)  │   │  (persistent)     │   │ (resume support)   │  │
│  └──────────────────┘   └──────────────────┘   └────────────────────┘  │
└────────────────────────────────────┬────────────────────────────────────┘
                                     │ dispatches objectives to ↓
┌────────────────────────────────────▼────────────────────────────────────┐
│                          Agent Layer                                     │
│  ┌─────────────┐  ┌──────────────┐  ┌───────────────────────────────┐  │
│  │  Planner    │  │ LitMiner     │  │  HypothesisGenerator          │  │
│  ├─────────────┤  ├──────────────┤  ├───────────────────────────────┤  │
│  │ ExpCoder    │  │  Executor    │  │  Reviewer (adversarial)       │  │
│  ├─────────────┤  ├──────────────┤  ├───────────────────────────────┤  │
│  │  MetaAgent  │  │  TomAgent    │  │  (all extend BaseAgent)       │  │
│  └─────────────┘  └──────────────┘  └───────────────────────────────┘  │
└─────────┬───────────────┬──────────────────┬───────────────────────────┘
          │               │                  │
 ┌────────▼──────┐ ┌──────▼──────┐ ┌────────▼────────────────────┐
 │  Literature   │ │ Experiments │ │  Theory-of-Mind (ToM)        │
 │  ├ ArxivConn  │ │ ├ Spec      │ │  ├ ReviewerPersonaRegistry   │
 │  ├ ScholarConn│ │ ├ Generator │ │  ├ CollaboratorIntentProfile  │
 │  ├ EvidenceReg│ │ ├ Sandbox   │ │  ├ AdversarialChallengeGen   │
 │  └ CitationGnd│ │ └ Healer    │ │  └ EpistemicNegotiationLoop  │
 └───────────────┘ └─────────────┘ └──────────────────────────────┘
                                           │
                              ┌────────────▼──────────┐
                              │     Reflection         │
                              │  ├ SelfReviewModule    │
                              │  └ SelfImprovementEng  │
                              └───────────────────────┘
```

---

## 3. Execution Model: Directed Objective Graph

Research does not execute linearly. Instead, `ObjectiveGraph` maintains a DAG where each node
is an `Objective` with optional dependency IDs. The runtime evaluates which objectives are
"ready" (all dependencies `COMPLETED`) and dispatches them in priority order.

Default objective chain for a single run:

```
plan → mine → hypothesize → tom → code → execute → review → reflect
```

Each step is backed by a specific agent, but the graph is configurable — new objectives can
be injected, existing ones re-ordered, or branches parallelised in future iterations.

---

## 4. Agent Roles

| Agent | Role | Key Outputs |
|---|---|---|
| `ResearchPlannerAgent` | Sets the research agenda | `state.metadata["plan"]` |
| `LiteratureMinerAgent` | Queries arXiv + scholarly APIs | `state.evidence[]` |
| `HypothesisGeneratorAgent` | Generates testable hypotheses | `state.hypotheses[]` |
| `TomAgent` | Epistemic negotiation across reviewer personas | `state.metadata["tom_negotiation"]` |
| `ExperimentCoderAgent` | Produces `ExperimentSpec` with runnable code | `state.experiments[]` |
| `ExecutorAgent` | Runs code in sandbox, captures results | `state.experiments[].result` |
| `ReviewerAgent` | Adversarial challenges from 4 personas | `state.reviews[]` |
| `MetaAgent` | Self-review, bottleneck detection, policy proposals | `state.reflections[]`, `state.policy_updates[]` |

---

## 5. Data Flow

1. **Config** (`AutoResearchConfig`) is loaded from YAML/TOML or defaults.
2. `AgentRuntime` builds an `ObjectiveGraph` and initialises `RunState`.
3. Each agent reads from `RunState` and writes enriched data back.
4. `CheckpointManager` persists `RunState` to `runs/{run_id}/checkpoints/state.json` after
   every objective completes.
5. The `MetaAgent` (last in chain) proposes policy updates; `SelfImprovementEngine` applies
   only those flagged `safe=True` and listed in the allowlist.

---

## 6. Theory-of-Mind (ToM) Layer

The ToM subsystem (`autoresearch/tom/`) models **what different reviewers / collaborators
believe**, not just what is objectively true. Four built-in reviewer personas are provided:

- `methodological_purist` — prioritises statistical rigour and reproducibility
- `empirical_skeptic` — demands real-world empirical validation
- `novelty_maximizer` — values groundbreaking contributions above all
- `benchmark_enforcer` — insists on standard benchmark comparisons

The `EpistemicNegotiationLoop` iteratively updates belief scores for each hypothesis across
multiple adversarial rounds, converging toward claims that are robust across all perspectives.

A **minority-report** mechanism preserves dissenting views and is included in the final state,
preventing the system from collapsing to a single biased narrative.

---

## 7. Self-Improvement Boundaries

AutoResearch v2 enforces **safe self-modification** at config/policy level only:

- Only parameters in the `SAFE_PARAMETERS` allowlist can be updated at runtime.
- All proposed updates are logged in `state.policy_updates` for human review.
- The `safe_modification_only` flag (default `True`) prevents any structural code mutation.
- Future milestone: structured prompt-level updates within sandboxed evaluation.

---

## 8. Directory Structure

```
autoresearch/
├── __init__.py
├── agents/          # All agent implementations
├── cli/             # Click CLI entrypoints
├── config/          # Pydantic config schema + YAML defaults
├── core/            # Runtime, ObjectiveGraph, RunState, Checkpointing
├── experiments/     # Spec, Generator, Sandbox, SelfHealer
├── literature/      # ArxivConnector, ScholarlyConnector, EvidenceRegistry, CitationGrounding
├── reflection/      # SelfReviewModule, SelfImprovementEngine
└── tom/             # ReviewerPersonas, IntentProfile, Adversarial, Negotiation

tests/               # 27 unit + E2E tests
docs/                # Architecture, contracts, roadmap, research review
Dockerfile
.devcontainer/
pyproject.toml
```
