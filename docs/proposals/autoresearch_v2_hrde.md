# AutoResearch v2: Hyper-Recursive Discovery Engine (HRDE)

**A Self-Evolving, Theory-of-Mind-Enhanced Autonomous Research Framework**

> **Status:** Proposal — Draft v0.1  
> **Authors:** AutoResearch Community  
> **Date:** April 2026  
> **Repository:** [Vikaash-dev/Autoresearch-v2](https://github.com/Vikaash-dev/Autoresearch-v2)

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Critique of Prior v1/v2-Style Pipeline Assumptions](#2-critique-of-prior-v1v2-style-pipeline-assumptions)
3. [New Architecture: Hyper-Recursive Discovery Engine (HRDE)](#3-new-architecture-hyper-recursive-discovery-engine-hrde)
4. [Dynamic Objective Graph (DOG) Orchestration Model](#4-dynamic-objective-graph-dog-orchestration-model)
5. [Hyper-Kernel / Meta-Self-Improvement Layer](#5-hyper-kernel--meta-self-improvement-layer)
6. [Discovery Tree/Forest Execution: Branch Pruning and Grafting](#6-discovery-treeforest-execution-branch-pruning-and-grafting)
7. [Theory-of-Mind (ToM) Reviewer Council and Collaborator Intent Modeling](#7-theory-of-mind-tom-reviewer-council-and-collaborator-intent-modeling)
8. [Zero-Trust Epistemic Verification Stack](#8-zero-trust-epistemic-verification-stack)
9. [Compute-Optimal Planning Strategy](#9-compute-optimal-planning-strategy)
10. [Safety, Guardrails, and Governance Model](#10-safety-guardrails-and-governance-model)
11. [Evaluation Metrics and Benchmark Plan](#11-evaluation-metrics-and-benchmark-plan)
12. [System Comparison Table](#12-system-comparison-table)
13. [Implementation Blueprint](#13-implementation-blueprint)
14. [30/60/90-Day Implementation Roadmap](#14-306090-day-implementation-roadmap)
15. [References and Citations](#15-references-and-citations)

---

## 1. Executive Summary

AutoResearch v2 (HRDE) is a proposed next-generation autonomous scientific discovery system designed to move decisively beyond the limitations of static pipeline and template-driven research automation. It synthesizes the most significant advances from the Sakana AI Scientist lineage, Meta-style hyper-agent frameworks, the AutoResearchClaw 23-stage pipeline, and the epistemic architecture patterns observed in the `tonitangpotato/autoresearch-engram` fork and related 2025–2026 arXiv literature.

**The central thesis:** Current AI scientist systems automate research *tasks*. HRDE treats the research *process itself* as the object of optimization — a dynamic, self-modifying system that improves its own discovery strategies across runs, models the cognitive states of reviewers and collaborators, and applies zero-trust epistemic verification to every generated claim.

**Key novelties over prior work:**

| Dimension | Prior Systems | HRDE |
|---|---|---|
| Orchestration | Static stage pipeline | Dynamic Objective Graph (DOG) |
| Self-improvement | Fixed code; occasional prompt tuning | Hyper-Kernel: rewriting sub-agent logic from telemetry |
| Review | Single AI reviewer or fixed adversarial prompts | ToM Reviewer Council: persona-differentiated, 3-round adversarial dialogue |
| Verification | Plausibility checks; citation grounding | Zero-Trust stack: formal logic + code gate + literature triangulation |
| Hypothesis search | Sequential or shallow parallel | Discovery Forest: branch spawning, pruning, and cross-branch grafting |
| Novelty driver | Combinatorial (idea A × idea B) | Anomaly-driven: targets gaps and contradictions in the arXiv corpus |
| Compute allocation | Uniform or undefined | Compute-optimal planner: budget-aware, simulation-first |

**Deliverable of this document:** A sufficiently detailed proposal for engineering teams to begin scoping and building HRDE in a phased 90-day MVP cycle.

> **Note on claims:** Where specific paper results or repository behaviors are referenced, sections marked `[to verify]` indicate that the cited detail should be confirmed against the primary source before inclusion in implementation specifications.

---

## 2. Critique of Prior v1/v2-Style Pipeline Assumptions

Understanding where existing systems fall short is necessary before proposing alternatives.

### 2.1 The Pipeline Fallacy

Systems like The AI Scientist v1 (Sakana AI, 2024) and AutoResearchClaw implement research as an ordered sequence of stages: ideation → search → experiment → write → review. Even when stages are made recursive or parallel (as in AI Scientist v2 `[to verify: exact v2 architecture details]`), the *structure of progression* is fixed at design time.

**Consequence:** The agent cannot discover that for a given domain or problem class, the correct order of operations is different — e.g., that for mathematical theory papers, formal verification should precede experiment design, not follow it. The pipeline imposes a cognitive architecture on the research process that may not match the domain.

### 2.2 The Recombination Ceiling

Most current AI scientist systems excel at **combinatorial innovation**: applying technique A from field X to problem B in field Y. This is what Thomas Kuhn would call "normal science." What these systems lack is a mechanism for **paradigm-level innovation** — identifying anomalies in the existing literature and constructing new theoretical frameworks to explain them.

This is not merely a capability gap; it is a *design assumption*: current systems treat the literature as a source of building blocks, not as a map whose contradictions reveal where new territory lies.

### 2.3 The Homogeneous Reviewer Problem

When both the "author agent" and the "reviewer agent" share the same underlying model (or even the same model class), they share systematic biases. A GPT-4o author and a GPT-4o reviewer are likely to find the same things plausible. This creates an **epistemic echo chamber** — work that looks peer-reviewed but has not actually been stress-tested against genuinely independent epistemic perspectives.

### 2.4 The Compute Uniformity Assumption

Prior systems allocate compute roughly uniformly across pipeline stages, or scale them based on wall-clock time. This misses the fundamental insight from recent inference-time compute scaling research `[to verify: specific citation for optimal compute allocation in agentic systems]` that the **hypothesis formulation and experimental design phases** benefit disproportionately from additional reasoning budget, while formatting and prose generation do not.

### 2.5 The Statefulness Gap

Most prior systems are **stateless across runs**: each new research cycle starts from scratch. The `tonitangpotato/autoresearch-engram` fork attempts to address this by implementing an "engram" (persistent memory structure for learned research patterns), which is one of the most promising ideas in the current ecosystem. HRDE extends this into a structured **Skill Registry** and **Vectorized Research Graph**.

### 2.6 The Physical World Disconnect

Current systems produce papers and code, but their verification loop is entirely digital. Real scientific claims ultimately need grounding in physical reality. HRDE's design includes (as a future milestone, not an MVP requirement) an interface to cloud lab APIs for requesting and ingesting physical experimental data.

---

## 3. New Architecture: Hyper-Recursive Discovery Engine (HRDE)

HRDE is organized around three structural layers, each with distinct responsibilities and interfaces:

```
┌─────────────────────────────────────────────────────────┐
│                    HYPER-KERNEL (Meta-Layer)             │
│  Self-modification engine · Performance telemetry ·     │
│  Agent factory · Strategy optimizer                     │
├─────────────────────────────────────────────────────────┤
│              DYNAMIC OBJECTIVE GRAPH (DOG)              │
│  Objective nodes · Dependency edges · State tracking ·  │
│  Resource allocation · Progress signals                 │
├─────────────────────────────────────────────────────────┤
│             DISCOVERY FOREST (Execution Layer)          │
│  Branch manager · Parallel sandboxes · Pruning logic ·  │
│  Grafting engine · Skill registry                       │
├────────────────┬──────────────────┬─────────────────────┤
│  ToM COUNCIL   │  EPISTEMIC STACK │  COMPUTE PLANNER    │
│  Reviewer sims │  Verification    │  Budget allocator   │
│  Intent model  │  Zero-trust      │  Simulation-first   │
└────────────────┴──────────────────┴─────────────────────┘
```

### 3.1 Core Design Principles

1. **Amorphous orchestration:** No hard-coded stage order. Objectives are nodes in a graph; the agent satisfies them in whatever order the DOG determines is most efficient given current state.
2. **Self-reference:** The Hyper-Kernel can modify the implementation of any sub-agent based on performance telemetry. This includes modifying the Hyper-Kernel's own optimization heuristics (bounded by safety constraints; see Section 10).
3. **Heterogeneous verification:** The epistemic verification stack uses architecturally diverse models (neural + symbolic + statistical) to prevent any single model's blind spots from passing unchallenged.
4. **Simulation-first compute allocation:** Before any code is executed against real data or APIs, the planner runs lightweight internal simulations to prune hypothesis branches.
5. **ToM as first-class citizen:** Modeling reviewer and collaborator mental states is not a post-hoc review step but an active driver of hypothesis selection and manuscript strategy from the beginning.

---

## 4. Dynamic Objective Graph (DOG) Orchestration Model

### 4.1 Overview

The DOG replaces the linear pipeline with a directed acyclic graph (DAG) of research objectives. Each node represents an **objective** (a verifiable research sub-goal), and edges represent dependencies.

```python
# Conceptual schema (not production code)

@dataclass
class Objective:
    id: str
    description: str
    status: Literal["pending", "in_progress", "complete", "failed", "pruned"]
    priority: float          # Updated dynamically by Hyper-Kernel
    assigned_agent: str | None
    dependencies: list[str]  # IDs of prerequisite objectives
    outputs: dict            # Key-value store for results passed to dependents
    compute_budget: float    # In normalized "thought tokens"
    verification_required: bool
```

### 4.2 Objective Types

| Type | Example | Notes |
|---|---|---|
| `LITERATURE_GROUNDING` | "Retrieve and verify 20 papers on sparse attention" | Must pass citation gate |
| `HYPOTHESIS_GENERATION` | "Generate 10 falsifiable hypotheses about X" | Scored by novelty + feasibility |
| `EXPERIMENT_DESIGN` | "Design experiment to test H3" | Requires grounding objective complete |
| `CODE_EXECUTION` | "Implement and run experiment E7" | Runs in isolated sandbox |
| `SYNTHESIS` | "Write Results section for branch B2" | Pulls from execution outputs |
| `REVIEW` | "ToM Council review of manuscript draft" | Requires synthesis complete |
| `META_OPTIMIZATION` | "Analyze run telemetry and update agent strategies" | Hyper-Kernel objective |

### 4.3 DOG Runtime Behavior

- **Dynamic node insertion:** As the agent discovers new sub-problems during execution (e.g., a code execution failure reveals a dependency on a missing library), the DOG adds new nodes automatically.
- **Priority re-scoring:** The Hyper-Kernel continuously re-scores objective priorities based on: (a) estimated compute cost, (b) expected impact on the overall research goal, (c) current branch performance.
- **Objective pruning:** Objectives on low-performing branches are marked `pruned` and their compute budgets reallocated. Pruning decisions are logged for the post-run skill extraction phase.
- **Parallelism:** Independent objectives (no dependency path between them) are executed in parallel up to the configured `max_parallel_objectives` limit.

### 4.4 DOG vs. Traditional Pipeline

| Property | Linear Pipeline | DOG |
|---|---|---|
| Order of execution | Fixed | Dynamic, dependency-driven |
| Response to failures | Retry or abort | Insert repair objectives; re-route |
| Parallelism | Explicit fan-out points | Automatic from dependency analysis |
| Scope of adaptation | None (within a run) | Continuous |

---

## 5. Hyper-Kernel / Meta-Self-Improvement Layer

### 5.1 Concept

The Hyper-Kernel is the meta-agent responsible for **improving the other agents**. It operates at a higher level of abstraction than the research sub-agents (Literature Agent, Coder Agent, Writer Agent, etc.) and monitors their performance to modify their behavior.

This design is inspired by the Darwin Gödel Machine (DGM) concept `[to verify: Lu et al., DGM paper, 2025]` and the self-referential codebase pattern from Meta HyperAgents `[to verify: specific HyperAgents paper/repo]`.

### 5.2 What the Hyper-Kernel Can Modify

The Hyper-Kernel operates within defined **modification scopes** to prevent unbounded self-modification (see Section 10 for safety constraints):

| Scope | What Changes | Trigger Condition |
|---|---|---|
| **Prompt policy** | System prompts and few-shot examples for sub-agents | Agent produces consistently low-quality outputs (measured by scorer) |
| **Tool selection** | Which tools/APIs an agent is allowed to invoke | Agent repeatedly fails with a specific tool |
| **Sampling parameters** | Temperature, top-p, max tokens for each agent | Agent outputs too verbose/short; inconsistent quality |
| **Objective routing** | Which agent type handles which objective type | Agent skill mismatch detected |
| **Pruning thresholds** | When to kill an underperforming branch | Discovery forest efficiency metrics fall below target |

> **Out of scope for Hyper-Kernel modification (MVP):** Core safety constraints, verification logic, output sanitization, human escalation thresholds.

### 5.3 Hyper-Kernel Feedback Loop

```
[Run telemetry] → [Pattern analyzer] → [Modification proposal]
       ↑                                         ↓
[Outcome logging] ←── [Sub-agent behavior] ←── [Apply modification (bounded)]
```

1. **Telemetry collection:** Every sub-agent call logs: input tokens, output tokens, execution time, quality score (from evaluator), downstream objective success rate.
2. **Pattern detection:** The Hyper-Kernel's analyzer identifies failure patterns (e.g., "Coder Agent fails 60% of the time when dealing with CUDA libraries").
3. **Modification proposal:** A modification is proposed with a predicted improvement and estimated risk.
4. **Bounded application:** Modifications that exceed a defined risk threshold are queued for human review rather than applied automatically.
5. **A/B testing:** When possible, the Hyper-Kernel runs a modified and unmodified version of the agent on parallel branches to measure actual improvement before permanent adoption.

### 5.4 Skill Registry

The Hyper-Kernel maintains a persistent **Skill Registry** — a structured store of lessons learned across runs:

```python
@dataclass
class Skill:
    id: str
    domain: str              # e.g., "bioinformatics", "NLP", "quantum chemistry"
    trigger_context: str     # Description of when to apply this skill
    implementation: str      # Prompt snippet, tool config, or code pattern
    success_rate: float      # Measured across applications
    source_run_id: str       # Traceability
    created_at: datetime
```

Skills are retrieved via semantic search when a new run begins, and injected into relevant sub-agents' contexts. This is the direct implementation of the **engram pattern** observed in `tonitangpotato/autoresearch-engram`.

---

## 6. Discovery Tree/Forest Execution: Branch Pruning and Grafting

### 6.1 The Discovery Forest

Rather than a single linear research thread, HRDE maintains a **Discovery Forest**: a set of parallel research branches, each exploring a distinct hypothesis or experimental approach.

```
Root: Research Goal
├── Branch A: Hypothesis H1 (attention-free transformers for protein folding)
│   ├── Sub-branch A1: SSM-based architecture variant
│   └── Sub-branch A2: Linear attention variant
├── Branch B: Hypothesis H2 (cross-domain transfer from NLP benchmarks)
│   └── Sub-branch B1: Fine-tuning approach
└── Branch C: Hypothesis H3 (physics-informed neural architecture)
    ├── Sub-branch C1: PDE-constrained training
    └── Sub-branch C2: Symmetry-aware architecture
```

### 6.2 Branch Lifecycle

Each branch passes through the following states:

| State | Trigger | Action |
|---|---|---|
| `SEEDED` | Hypothesis generated | Assign initial compute budget |
| `GROWING` | Experiment running | Monitor telemetry |
| `PROMISING` | Critic score > threshold | Increase compute budget |
| `STAGNANT` | No improvement for N steps | Issue warning; reduce budget |
| `PRUNED` | Quality score below floor OR budget exhausted | Terminate; extract learnings |
| `GRAFTED` | Elements merged into another branch | Mark as consumed |
| `HARVESTED` | Branch produces publishable result | Package for synthesis |

### 6.3 Pruning Logic

The **Critic Agent** evaluates branches on a configurable interval (default: 30 minutes wall-clock or every 10 objective completions). Pruning criteria:

1. **Statistical significance check:** If preliminary results show p-value > 0.2 with no improvement trend, flag for pruning.
2. **Novelty check:** If the branch's outputs are too similar to existing arXiv papers (cosine similarity > 0.85 against retrieved papers), flag as redundant.
3. **Hallucination check:** If the branch's generated claims contradict its own execution logs (code gate failure), immediately prune and log the failure pattern.
4. **Resource efficiency:** Branches consuming >2× the average compute per unit of quality are candidates for pruning.

### 6.4 Grafting

**Grafting** is the mechanism for combining insights from two branches:

- **Code grafting:** The working code module from Branch A is injected into Branch B as a dependency.
- **Concept grafting:** The theoretical framing from Branch C is applied to the experimental setup of Branch A, creating a new hybrid branch.
- **Negative grafting:** A pruned branch's failure modes are injected into remaining branches as constraints to avoid.

Grafting is proposed by the Critic Agent and approved by the DOG orchestrator. The resulting hybrid branch inherits a weighted compute budget from both parents.

### 6.5 Compute Budget Management

```
Total run budget = Σ branch budgets
Branch budget[i] = base_budget × performance_multiplier[i]
performance_multiplier[i] = f(critic_score, novelty_score, compute_efficiency)
```

When a branch is pruned, its remaining budget is redistributed proportionally to surviving branches, weighted by their current performance multiplier.

---

## 7. Theory-of-Mind (ToM) Reviewer Council and Collaborator Intent Modeling

> For the detailed technical specification of the ToM layer, see the companion document:  
> [`docs/proposals/autoresearch_v2_tom_epistemics.md`](./autoresearch_v2_tom_epistemics.md)

### 7.1 Why Theory of Mind in Scientific Research

Science is fundamentally social: a discovery is not complete until it has been communicated, evaluated, and accepted (or rejected) by a community of minds with distinct priors, standards, and incentives. An autonomous research system that cannot model this social dimension will optimize for outputs that *look* like science without being *accepted* as science.

Theory of Mind (ToM) — the capacity to represent and reason about the mental states (beliefs, intentions, knowledge) of other agents — is the capability that bridges generation and communication.

### 7.2 The ToM Reviewer Council

The council consists of five persistent reviewer personas, each implemented as a distinct LLM prompt configuration with specialized evaluation criteria:

| Persona | Primary Focus | Archetypal Objection Style |
|---|---|---|
| **The Skeptic** | Statistical rigor, p-hacking, overclaiming | "The effect size is too small; the confidence interval crosses zero." |
| **The Engineer** | Reproducibility, computational efficiency, code quality | "The baseline is not properly ablated; the runtime is not reported." |
| **The Visionary** | Novelty, broader impact, "big picture" framing | "This is incremental; it doesn't reframe the field." |
| **The Domain Expert** | Domain-specific correctness, literature coverage | "The authors missed [key paper]; this contradicts [established result]." |
| **The Ethicist** | Dual-use risk, data bias, societal impact | "The training data may contain demographic bias not addressed in limitations." |

> **Architecture note:** These personas should ideally use architecturally diverse base models (see Section 8.3) to avoid the homogeneous reviewer failure mode (Section 2.3). At MVP scale, diverse system prompts with distinct few-shot examples are an acceptable approximation. True architectural diversity is a post-MVP upgrade.

### 7.3 The Adversarial Dialogue Protocol

The review process operates as a structured, multi-round adversarial dialogue:

**Round 1 — Blind Review:** Each reviewer independently scores the manuscript on a 7-point rubric (novelty, rigor, clarity, significance, reproducibility, ethics, related work coverage). No reviewer sees other reviews.

**Round 2 — Rebuttal:** The Author Agent generates a rebuttal addressing each reviewer's specific objections. The rebuttal cites execution logs, arXiv references, or logical arguments — not rhetorical deflection.

**Round 3 — Final Assessment:** Reviewers update their scores based on the rebuttal. The council reaches a decision based on a configurable threshold (default: 4/5 reviewers above acceptance threshold).

**If consensus is not reached:** The manuscript is returned to the Discovery Forest for revision, or the specific failing dimension is flagged for Hyper-Kernel intervention.

### 7.4 Collaborator Intent Modeling

When a human collaborator provides a research directive, the system builds a **Collaborator Intent Model** (CIM):

```python
@dataclass
class CollaboratorIntentModel:
    risk_appetite: float         # 0=conservative, 1=high-risk novelty
    rigor_vs_speed: float        # 0=speed-first, 1=rigor-first
    target_venue: str | None     # e.g., "NeurIPS", "Nature", "arXiv preprint"
    domain_expertise_level: float  # 0=novice, 1=expert
    interpretability_priority: float  # how much to explain reasoning
    update_history: list[dict]   # track intent drift during long runs
```

This model is inferred from:
- The initial research directive (NLP-based intent parsing)
- Explicit preference signals from HITL intervention points
- Feedback on intermediate outputs

The CIM is used to tune the DOG's objective priorities and the ToM Council's scoring weights.

### 7.5 Safety Constraints on ToM

ToM capabilities introduce a risk: the system could optimize for **persuasion** rather than **truth**. The following constraints are mandatory:

1. **Truth-first invariant:** No output modification made by the Intent Alignment Planner may remove or qualify an empirical finding that passed the epistemic verification stack.
2. **Citation-grounded inference only:** Mind-model predictions must be traceable to textual evidence in the paper, author's prior work, or reviewer archetype documentation.
3. **Minority-report preservation:** All reviewer scores and objections are preserved in the audit log. The final manuscript must acknowledge the strongest unresolved objection in its limitations section.
4. **Epistemic audit log:** Every major revision of a claim records: original claim, modification reason, supporting evidence, which reviewer objection triggered the change.

---

## 8. Zero-Trust Epistemic Verification Stack

### 8.1 Motivation

A system that generates plausible-sounding but false claims is worse than no system at all. HRDE adopts a **zero-trust** posture: no claim made by any sub-agent is accepted as true unless it has passed through an independent verification channel.

### 8.2 The Three-Gate Verification Pipeline

Every significant claim in the output manuscript must pass all three gates before inclusion:

```
Claim generated by Author Agent
         ↓
┌────────────────────────────────┐
│  GATE 1: Literature Gate        │
│  Does this contradict known     │
│  results in the arXiv corpus?   │
│  (Semantic search + LLM judge)  │
└────────────┬───────────────────┘
             ↓ PASS
┌────────────────────────────────┐
│  GATE 2: Code Gate              │
│  Do the execution logs confirm  │
│  the numbers cited in the text? │
│  (Deterministic log parsing)    │
└────────────────┬───────────────┘
                 ↓ PASS
┌────────────────────────────────┐
│  GATE 3: Formal Gate (optional) │
│  Can the core claim be verified │
│  by Z3/Lean? (math/algo papers) │
└────────────────┬───────────────┘
                 ↓ PASS
         Claim accepted
```

**Gate failure handling:**
- **Literature gate failure:** Claim is flagged; Author Agent must either revise the claim or provide explicit justification for why the contradiction does not apply.
- **Code gate failure:** Claim is **rejected**; the relevant branch is marked for audit, and the Critic Agent is notified.
- **Formal gate failure:** For papers where formal verification is applicable, this is a blocking failure. For domains where it is not applicable, this gate is skipped.

### 8.3 Architectural Heterogeneity in Verification

To avoid the homogeneous reviewer problem (Section 2.3), the verification stack deliberately uses models from different architectural families:

| Gate | Primary Verifier | Secondary Verifier |
|---|---|---|
| Literature Gate | Transformer-based LLM (e.g., GPT-4o) | Dense retrieval model (e.g., BGE, E5) |
| Code Gate | Deterministic log parser (no LLM) | Static analysis tool |
| Formal Gate | Z3 SMT solver | Lean theorem prover `[to verify: integration feasibility]` |

A claim is flagged for human review if the primary and secondary verifiers disagree.

### 8.4 Anti-Hallucination Measures

1. **DOI anchoring:** Every citation must resolve to a real document via OpenAlex, Semantic Scholar, or arXiv APIs. No citation is accepted without a verified DOI or arXiv ID.
2. **Number tracing:** Every numerical result in the manuscript is linked to a specific line in the execution log. If the log doesn't contain the number, the claim is rejected.
3. **Claim provenance graph:** The system maintains a graph linking every claim in the final output to its source (literature reference, execution log, or explicit inference step).

---

## 9. Compute-Optimal Planning Strategy

### 9.1 The Simulation-First Principle

Before executing any experiment that consumes real compute resources (API calls, code execution, GPU time), HRDE runs a **lightweight internal simulation** to estimate:

- **Expected result:** What is the most likely outcome given prior literature?
- **Novelty delta:** How different is this expected result from the existing literature?
- **Risk of failure:** What is the probability the experiment fails for technical reasons?

If the novelty delta is below the configured threshold (default: "result is essentially a replication"), the experiment is not executed.

### 9.2 Compute Budget Allocation

Inspired by inference-time compute scaling research `[to verify: Snell et al. 2024, or related 2025/2026 work on compute-optimal inference]`:

| Phase | Recommended Budget Share | Rationale |
|---|---|---|
| Hypothesis generation & selection | 35% | High leverage: better hypotheses dramatically improve downstream quality |
| Experiment design & simulation | 25% | Simulation-first prunes expensive failed experiments |
| Code execution | 20% | Bounded by actual runtime; not generally improved by more "thinking" |
| Synthesis & writing | 10% | Mostly mechanical once results are in hand |
| Review & revision | 10% | Focused, targeted revisions based on council feedback |

These are starting-point guidelines, not hard constraints. The Hyper-Kernel updates them based on domain-specific telemetry.

### 9.3 Adaptive Resource Scaling

- **Scale up:** Branches showing strong novelty scores and clean execution logs receive additional compute from the Hyper-Kernel.
- **Scale down:** Branches with declining novelty or increasing failure rates have compute reduced before pruning.
- **Minimum viable branch:** Each branch maintains a minimum compute floor to ensure fair evaluation before pruning.

---

## 10. Safety, Guardrails, and Governance Model

### 10.1 Safety Principles

HRDE's self-modification capability introduces unique safety requirements beyond those of static AI systems:

1. **Modification scope constraints:** The Hyper-Kernel cannot modify the safety layer, verification stack, or human escalation logic.
2. **Rollback capability:** Every agent modification is versioned. Any modification that degrades performance by >20% triggers automatic rollback.
3. **Human escalation thresholds:** A configurable set of conditions always triggers human review before proceeding:
   - Confidence on a key finding below threshold
   - First-time modification of a previously unmodified agent module
   - Code execution in a domain flagged as high-risk (biohazard, weapons, etc.)
   - Reviewer council does not reach consensus after 3 rounds

### 10.2 Dual-Use and Domain Restrictions

The system includes a **Domain Safety Classifier** that evaluates the research topic before initiating a run:

- **Green:** Standard academic research domains (ML, physics, chemistry, biology, social sciences)
- **Yellow:** Domains requiring additional review (gain-of-function biology, advanced materials with dual-use potential) — requires human approval before proceeding
- **Red:** Prohibited domains (weapons development, surveillance systems) — run is blocked

### 10.3 Output Governance

- All generated papers are watermarked with metadata indicating AI generation.
- The claim provenance graph (Section 8.4) is attached to the manuscript as an appendix.
- Execution logs are retained for 90 days for auditability.
- Any dataset used is checked against a list of known biased or problematic datasets.

### 10.4 Compute Governance

- A per-run compute cap is set at initialization and cannot be exceeded without human override.
- Cost estimates are displayed before each run and updated in real-time.
- The system does not autonomously spin up additional cloud resources beyond the configured capacity pool.

---

## 11. Evaluation Metrics and Benchmark Plan

### 11.1 Research Quality Metrics

| Metric | Description | Target (MVP) |
|---|---|---|
| **Acceptance simulation rate** | % of papers rated "accept" by ToM Council | > 60% |
| **Novelty score** | Semantic distance from most similar arXiv paper | > 0.4 (normalized) |
| **Reproducibility score** | % of claims that pass the Code Gate | > 95% |
| **Citation accuracy** | % of citations with verified DOI/arXiv ID | 100% |
| **Formal verification rate** | % of core mathematical claims verified by Z3/Lean | > 80% (for applicable papers) |

### 11.2 System Performance Metrics

| Metric | Description | Target (MVP) |
|---|---|---|
| **Time to first draft** | Wall-clock time from prompt to first complete draft | < 4 hours |
| **Branch efficiency** | % of branches that reach `HARVESTED` state | > 20% |
| **Hyper-Kernel improvement rate** | Performance delta per run cycle | Positive trend |
| **Skill registry hit rate** | % of runs where retrieved skills improve outcomes | > 40% |

### 11.3 ToM-Specific Metrics

| Metric | Description |
|---|---|
| **Belief-Robustness Score** | Claim survives across all 5 ToM reviewer archetypes |
| **Adversarial Persuasion Gap** | Change in Skeptic reviewer score from Round 1 to Round 3 |
| **Intent Fidelity** | Alignment between final output and Collaborator Intent Model |
| **Cross-Paradigm Stability** | Result holds under alternative theoretical framings tested by Domain Expert |

### 11.4 Benchmark Plan

**Phase 1 (Days 1–30):** Establish baselines by running The AI Scientist v1/v2 (or its closest available open-source equivalent) on 5 canonical benchmark tasks across 3 domains.

**Phase 2 (Days 31–60):** Run HRDE MVP on the same tasks. Compare on Research Quality Metrics and System Performance Metrics.

**Phase 3 (Days 61–90):** Introduce adversarial challenges: conflicting literature, deliberately noisy data, ambiguous research directives. Measure robustness differentials.

---

## 12. System Comparison Table

| Feature | AI Scientist v1 (Sakana) | AI Scientist v2 (Sakana) | AutoResearchClaw | HyperAgents / DGM-H | AutoResearch v2 (HRDE) |
|---|---|---|---|---|---|
| **Orchestration** | Linear pipeline | Agentic tree search `[to verify]` | 23-stage pipeline | Agent-as-code, DGM loop | Dynamic Objective Graph |
| **Self-improvement** | None (within run) | Limited `[to verify]` | None | DGM-style code rewriting | Hyper-Kernel with bounded modification scopes |
| **Hypothesis search** | Template + LLM | Parallelized `[to verify]` | LLM + literature | Evolutionary search | Discovery Forest with grafting |
| **Reviewer** | Single LLM reviewer | Improved reviewer `[to verify]` | Multi-round | Not specified | 5-persona ToM Council, 3-round adversarial |
| **Verification** | Citation check | Citation + execution check `[to verify]` | Anti-hallucination layer | Not specified | 3-gate zero-trust stack (literature + code + formal) |
| **Memory across runs** | None | None `[to verify]` | None | Code mutation history | Skill Registry + Vectorized Research Graph |
| **Theory of Mind** | None | None | None | None | CIM + ToM Reviewer Council |
| **Compute optimization** | Uniform | Partially dynamic `[to verify]` | Uniform | Not specified | Simulation-first + budget-aware allocation |
| **Physical world interface** | None | None | None | None | Planned (cloud lab API, post-MVP) |
| **Safety layer** | Basic | Improved `[to verify]` | Domain filtering | Not specified | Domain classifier + modification scope constraints + audit log |
| **Open source** | Yes | Partially `[to verify]` | Yes (partial) | Research paper only `[to verify]` | Proposed open source |
| **Key reference** | [R1] | [R2] | [R3] | [R4] | This document |

> Items marked `[to verify]` should be confirmed against primary sources before inclusion in any published comparison.

---

## 13. Implementation Blueprint

### 13.1 Module Boundaries

```
autoresearch_v2/
├── core/
│   ├── dog/                    # Dynamic Objective Graph
│   │   ├── graph.py            # Graph data structure and traversal
│   │   ├── objective.py        # Objective dataclass and state machine
│   │   ├── scheduler.py        # Parallel objective scheduling
│   │   └── allocator.py        # Compute budget management
│   ├── hyper_kernel/           # Meta-self-improvement layer
│   │   ├── kernel.py           # Main Hyper-Kernel agent
│   │   ├── telemetry.py        # Run telemetry collection
│   │   ├── modifier.py         # Bounded agent modification engine
│   │   └── skill_registry.py   # Persistent skill store
│   └── agents/                 # Research sub-agents
│       ├── base_agent.py       # Abstract base with telemetry hooks
│       ├── literature_agent.py # Literature search and grounding
│       ├── hypothesis_agent.py # Hypothesis generation
│       ├── coder_agent.py      # Code generation and self-healing
│       ├── writer_agent.py     # Manuscript generation
│       └── critic_agent.py     # Branch quality evaluation
├── discovery/
│   ├── forest.py               # Discovery Forest manager
│   ├── branch.py               # Branch lifecycle management
│   ├── pruner.py               # Pruning logic
│   └── grafter.py              # Cross-branch grafting
├── tom/                        # Theory of Mind layer
│   ├── council.py              # ToM Reviewer Council
│   ├── personas/               # Reviewer persona configurations
│   │   ├── skeptic.py
│   │   ├── engineer.py
│   │   ├── visionary.py
│   │   ├── domain_expert.py
│   │   └── ethicist.py
│   ├── dialogue.py             # Adversarial dialogue protocol
│   └── intent_model.py         # Collaborator Intent Model
├── epistemic/                  # Zero-trust verification stack
│   ├── pipeline.py             # Orchestrates three-gate verification
│   ├── literature_gate.py      # Gate 1: literature consistency
│   ├── code_gate.py            # Gate 2: execution log verification
│   ├── formal_gate.py          # Gate 3: Z3/Lean formal verification
│   └── provenance.py           # Claim provenance graph
├── compute/
│   ├── planner.py              # Compute-optimal planning
│   └── simulator.py            # Lightweight pre-execution simulation
├── safety/
│   ├── domain_classifier.py    # Domain safety classification
│   ├── guardrails.py           # Output sanitization and constraints
│   └── audit_log.py            # Immutable audit trail
├── memory/
│   ├── research_graph.py       # Vectorized Research Graph (ChromaDB/Pinecone)
│   └── engram.py               # Cross-run engram persistence (from autoresearch-engram pattern)
├── integrations/
│   ├── arxiv_client.py         # arXiv API
│   ├── openalex_client.py      # OpenAlex API
│   ├── semantic_scholar.py     # Semantic Scholar API
│   └── sandbox.py              # Docker sandbox manager
└── cli.py                      # Main CLI entry point
```

### 13.2 Key Interfaces

**DOG ↔ Agents:**
```python
class BaseAgent:
    def execute(self, objective: Objective, context: dict) -> ObjectiveResult:
        ...
    def get_telemetry(self) -> AgentTelemetry:
        ...
```

**Hyper-Kernel ↔ Agents:**
```python
class HyperKernel:
    def optimize_agent(self, agent_id: str, telemetry: AgentTelemetry) -> AgentModification:
        ...
    def apply_modification(self, modification: AgentModification) -> bool:
        ...
```

**Epistemic Stack ↔ Author Agent:**
```python
class EpistemicPipeline:
    def verify_claim(self, claim: str, evidence: dict) -> VerificationResult:
        ...
    def get_provenance_graph(self) -> ProvenanceGraph:
        ...
```

**ToM Council ↔ Writer Agent:**
```python
class ToMCouncil:
    def review(self, manuscript: Manuscript) -> CouncilVerdict:
        ...
    def run_adversarial_dialogue(self, manuscript: Manuscript, max_rounds: int = 3) -> DialogueResult:
        ...
```

### 13.3 MVP Milestones

**Milestone 1 — Core Infrastructure (Days 1–30):**
- [ ] DOG graph implementation with basic scheduler
- [ ] 4 core sub-agents (Literature, Hypothesis, Coder, Writer) with base telemetry
- [ ] Docker sandbox integration
- [ ] Basic CLI (`autoresearch_v2.py --topic "..." --branches N`)
- [ ] Output: A runnable system that produces a draft paper from a text prompt

**Milestone 2 — Discovery Forest + ToM (Days 31–60):**
- [ ] Discovery Forest with pruning logic
- [ ] ToM Reviewer Council (5 personas, 3-round dialogue)
- [ ] 2-gate epistemic verification (Literature + Code gates)
- [ ] Collaborator Intent Model (basic)
- [ ] Output: System that produces a peer-reviewed draft with traced claims

**Milestone 3 — Hyper-Kernel + Full Stack (Days 61–90):**
- [ ] Hyper-Kernel with prompt-policy and tool-selection modification
- [ ] Skill Registry with cross-run persistence
- [ ] Formal gate (Z3 integration for math/algorithm papers)
- [ ] Branch grafting
- [ ] Full evaluation metrics dashboard
- [ ] Output: Self-improving system with measurable improvement across runs

---

## 14. 30/60/90-Day Implementation Roadmap

### Days 1–30: Foundation

**Goal:** A working end-to-end pipeline producing draft papers.

| Week | Deliverable |
|---|---|
| 1 | Repository structure, base agent framework, DOG data structures |
| 2 | Literature Agent + arXiv/OpenAlex integration; basic DOI verification |
| 3 | Hypothesis Agent + Coder Agent + Docker sandbox |
| 4 | Writer Agent + basic CLI; integration test across 2 domains |

**Team requirements:** 2 backend engineers, 1 ML engineer, 1 DevOps engineer.

**Technical risks:**
- Docker-in-Docker security configuration (mitigate: use `gVisor` or `Firecracker` isolation)
- API rate limits on literature sources (mitigate: implement caching layer early)

---

### Days 31–60: Intelligence Layer

**Goal:** Discovery Forest active; ToM Council producing meaningful review feedback.

| Week | Deliverable |
|---|---|
| 5 | Discovery Forest (branch spawning, basic pruning) |
| 6 | ToM Council (3 personas functional; Round 1 review) |
| 7 | ToM Council (all 5 personas; full 3-round dialogue); Literature Gate |
| 8 | Code Gate + claim provenance graph; Collaborator Intent Model v1 |

**Team requirements:** +1 ML engineer for ToM persona development; +1 QA engineer.

**Technical risks:**
- ToM personas producing circular validation (homogeneous base model) — mitigate by using diverse few-shot examples and structurally distinct evaluation rubrics before architectural diversity is feasible
- Code Gate false positives (execution logs have valid variance) — mitigate with configurable numeric tolerance

---

### Days 61–90: Self-Improvement Layer

**Goal:** Hyper-Kernel operational; measurable improvement across at least 3 run cycles.

| Week | Deliverable |
|---|---|
| 9 | Hyper-Kernel telemetry collection + pattern analyzer |
| 10 | Bounded modification engine (prompt-policy scope only, MVP) |
| 11 | Skill Registry + cross-run persistence; branch grafting |
| 12 | Evaluation metrics dashboard; formal gate (Z3); full benchmark run |

**Team requirements:** +1 research engineer for Hyper-Kernel design; +1 for evaluation infrastructure.

**Technical risks:**
- Hyper-Kernel modifications degrading performance (mitigate: mandatory A/B testing before permanent adoption; 20% regression triggers auto-rollback)
- Skill Registry quality degradation (mitigate: skills have expiry and are subject to periodic re-evaluation)

---

### Post-90-Day Roadmap (Future Milestones)

- [ ] Lean theorem prover integration for formal gate
- [ ] Architectural diversity in ToM Council (heterogeneous base models)
- [ ] Cloud lab API integration (Emerald Cloud Lab / Strateos) for physical experiment requests
- [ ] Multi-collaborator intent modeling
- [ ] Open-source release with documentation

---

## 15. References and Citations

The following references influenced the design decisions in this proposal. Items marked `[to verify]` require confirmation before inclusion in any published version of this document.

### Primary Inspirations

**[R1] The AI Scientist: Towards Fully Automated Open-Ended Scientific Discovery**
- Authors: Lu et al. (Sakana AI)
- Year: 2024
- arXiv: `[to verify: confirm exact arXiv ID]` — search arXiv for "AI Scientist Sakana"
- Influenced: Pipeline critique (Section 2.1), Comparison Table (Section 12)

**[R2] The AI Scientist v2 (or successor work from Sakana AI)**
- `[to verify: confirm paper title, authors, arXiv ID, and exact v2 feature set]`
- Influenced: Agentic tree search design (Section 6), Comparison Table (Section 12)

**[R3] AutoResearchClaw**
- Repository: `[to verify: confirm canonical GitHub URL]`
- Influenced: 23-stage pipeline critique (Section 2.1), DOG design (Section 4), anti-hallucination measures (Section 8.4)

**[R4] Darwin Gödel Machine / Self-Referential Self-Improvement**
- `[to verify: confirm canonical paper — possibly "OMNI-EPIC" or "Darwin Gödel Machine: Open-Ended Evolution of Self-Replicating and Self-Improving Programs"; arXiv 2411.09838 or similar 2024/2025 work]`
- Influenced: Hyper-Kernel design (Section 5), self-modification bounds (Section 10.1)

**[R5] Meta HyperAgents / Agent-as-Code Paradigm**
- `[to verify: confirm canonical paper or repository for Meta HyperAgents]`
- Influenced: Agent-as-code pattern (Section 5.1), self-referential codebase concept

**[R6] tonitangpotato/autoresearch-engram**
- Repository: https://github.com/tonitangpotato/autoresearch-engram
- Influenced: Skill Registry design (Section 5.4), cross-run memory (Section 13.1 `memory/engram.py`), engram pattern adoption

### Supporting Literature

**[R7] Inference-Time Compute Scaling**
- `[to verify: e.g., Snell et al. 2024 "Scaling LLM Test-Time Compute Optimally"; confirm arXiv ID and relevance to multi-agent agentic systems]`
- Influenced: Compute-optimal planning strategy (Section 9)

**[R8] Theory of Mind in Large Language Models**
- `[to verify: relevant 2023–2026 papers on ToM in LLMs — e.g., Bubeck et al. 2023 "Sparks of AGI", or specific ToM benchmark papers]`
- Influenced: ToM Reviewer Council design (Section 7)

**[R9] Zero-Trust Security Principles Applied to AI Systems**
- `[to verify: relevant 2024–2026 papers applying zero-trust concepts to AI agent outputs]`
- Influenced: Zero-Trust Epistemic Verification Stack (Section 8)

**[R10] Formal Verification with Z3/Lean for AI-Generated Code**
- `[to verify: relevant papers on LLM + formal verification integration, e.g., work on LLM-guided theorem proving]`
- Influenced: Formal Gate design (Section 8.2)

**[R11] Compute-Optimal Training and Inference (Chinchilla, Scaling Laws)**
- `[to verify: Hoffmann et al. 2022 "Training Compute-Optimal Large Language Models"; extend to inference-time analogs]`
- Influenced: Compute budget allocation (Section 9.2)

**[R12] Multi-Agent Scientific Discovery (2025–2026 arXiv)**
- `[to verify: search arXiv for multi-agent scientific discovery papers from 2025–2026; recommended queries: "autonomous scientific discovery multi-agent 2025", "self-improving research agent 2026"]`
- Influenced: General architecture framing

### Recommended Literature Review Queries

For maintainers updating the references section, the following arXiv search queries are recommended:

```
"autonomous research agent" site:arxiv.org
"AI scientist self-improving" site:arxiv.org
"multi-agent hypothesis generation" site:arxiv.org date:2025-2026
"theory of mind language model scientific review" site:arxiv.org
"inference time compute scientific discovery" site:arxiv.org
```

---

*End of Document*

**Contributing:** If you identify papers that should fill the `[to verify]` placeholders, please submit a PR updating this references section with confirmed titles, authors, arXiv IDs, and a one-sentence note on how the paper influenced the relevant design decision.
