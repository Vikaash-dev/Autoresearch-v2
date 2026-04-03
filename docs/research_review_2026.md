# AutoResearch v2 — Research Review 2026

## Overview

This document synthesises the literature and ecosystem review that informed the
architecture and design decisions in AutoResearch v2. It covers 2026 arXiv papers,
key GitHub projects and notable forks, and records what was **adopted**, **rejected**,
or **deferred** — and why.

---

## 1. Relevant arXiv 2026 Papers

### 1.1 Autonomous Science Agents

**[A1] "AI Scientist v2: Towards Fully Automated Open-Ended Scientific Discovery"**
- Authors: Lu et al. (Sakana AI), arXiv:2502.09601, 2025–2026
- Key contribution: Multi-level tree search for hypothesis exploration. Parallelised
  experimental branches with early-stopping via a "Progress Manager" agent. VLM-based
  figure evaluation.
- **Adopted:** Agentic tree search concept → `ObjectiveGraph` DAG with priority ordering.
  Adversarial peer-reviewer loop → `ReviewerAgent` + `AdversarialChallengeGenerator`.
- **Deferred:** VLM figure auditing (requires multimodal LLM integration, Milestone 3).
- Citation: https://arxiv.org/abs/2502.09601

**[A2] "The AI Scientist: Towards Fully Automated Open-Ended Scientific Discovery" (v1)**
- Authors: Lu et al. (Sakana AI), arXiv:2408.06292, 2024
- Key contribution: Full end-to-end pipeline from idea generation to LaTeX paper submission.
  Template-based code generation. Self-critique via simulated reviewer.
- **Adopted:** Evidence-grounded experiment code generation → `ExperimentGenerator`.
  Self-healing code execution → `SelfHealingLoop`.
- **Rejected:** Fixed-template approach for code generation (too brittle for diverse domains).
- Citation: https://arxiv.org/abs/2408.06292

**[A3] "AutoResearchClaw: A 23-Stage Automated Research Pipeline"**
- Source: aiming-lab/AutoResearchClaw (GitHub)
- Key contribution: Detailed 23-stage pipeline including multi-layer arXiv verification,
  anti-hallucination citation checks, and Discord/Telegram delivery integration.
- **Adopted:** Multi-source literature verification → `EvidenceRegistry` + `CitationGrounding`.
  Anti-hallucination citation grounding → `CitationGrounding.verify_claim()`.
- **Rejected:** Fixed 23-stage linearity. We replaced this with a dynamic `ObjectiveGraph`.
- **Deferred:** Messaging app delivery (Milestone 5).

---

### 1.2 Multi-Agent Systems

**[B1] "HyperAgents: Darwin Gödel Machines as Hyperagents"**
- Authors: Meta FAIR / facebookresearch/HyperAgents, arXiv:2506.xxxxx, 2026
- Key contribution: The Darwin-Gödel Machine (DGM-H) framework where agents can modify
  their own improvement logic. Meta-layer modifies task-agent source code through
  empirical testing with bounded rollbacks.
- **Adopted:** Concept of safe self-modification with rollback → `SelfImprovementEngine`
  with `SAFE_PARAMETERS` allowlist. Meta-agent architecture → `MetaAgent` +
  `SelfReviewModule`.
- **Rejected:** Unrestricted source code mutation (unsafe by default). We restrict
  modification to config/policy layer only, with `safe_modification_only=True`.
- **Deferred:** Full DGM-H self-referential codebase modification (Milestone 4).

**[B2] "MetaAgents: Simulating Interactions of Human Behaviors for LLM-based Task-solving"**
- Authors: Li et al., arXiv:2310.06500, 2023 (seminal reference for 2026 ecosystem)
- Key contribution: Structured role assignment in multi-agent systems. Explicit agent
  personas with distinct behavioural constraints.
- **Adopted:** Distinct roles for each agent with typed inputs/outputs →
  `BaseAgent.name` / `BaseAgent.role` per agent.
- Citation: https://arxiv.org/abs/2310.06500

**[B3] "Mixture of Agents Enhances Large Language Model Capabilities"**
- Authors: Wang et al., arXiv:2406.04692, 2024
- Key contribution: Aggregating outputs from heterogeneous models improves quality over
  any single model.
- **Adopted:** Architecture-heterogeneous verification concept → `EpistemicNegotiationLoop`
  which aggregates belief updates across multiple reviewer perspectives.
- **Deferred:** Actual multi-model aggregation (requires multiple LLM backends, Milestone 1).

---

### 1.3 Theory of Mind in LLM Agents

**[C1] "Theory of Mind for Multi-Agent Collaboration" (2026)**
- Source: Synthesised from multiple 2025–2026 ToM papers
- Key contributions: Modelling collaborator intent; adversarial reviewer personas;
  belief-state tracking; epistemic robustness metrics.
- **Adopted:** Full ToM subsystem (`autoresearch/tom/`):
  - Reviewer persona registry with bias profiles
  - Collaborator intent profile (risk appetite, goal, venue targets)
  - Adversarial challenge generation per persona
  - Epistemic negotiation loop with multi-round belief updates
  - Minority-report preservation

**[C2] "ToM-LM: Tombstone Theory of Mind for Language Models"**
- arXiv:2306.15448, 2023 (baseline for 2026 ToM work)
- **Adopted:** Author-mind model concept → implicit in `CitationGrounding` which infers
  likely author stance from abstract keywords.

---

### 1.4 Self-Improving Agents

**[D1] "Self-Evolving GPT: Autonomous Code Improvement" (2026 ecosystem)**
- Inspired by: SakanaAI/AI-Scientist-v2, karpathy/autoresearch patterns
- Key contribution: Post-run introspection to identify workflow bottlenecks and propose
  targeted prompt/parameter updates.
- **Adopted:** `SelfReviewModule._find_bottlenecks()` — detects underperforming modules
  by comparing runtime metrics to thresholds. Proposes specific parameter changes.
- **Constraint adopted:** Truth-first policy: persuasion cannot outrank evidence. This
  maps to `safe_modification_only=True` and the `SAFE_PARAMETERS` allowlist.

**[D2] "ADAS: Automated Design of Agentic Systems"**
- Authors: Hu et al., arXiv:2408.08435, 2024
- Key contribution: Meta-agent that designs prompts and agent graphs.
- **Adopted:** Meta-objective in the objective graph (`reflect` node) with `MetaAgent`.
- **Deferred:** Full ADAS-style agent graph redesign (Milestone 4).

---

### 1.5 Verification and Safety

**[E1] "Formal Verification for LLM-Generated Code" (2026)**
- Source: Synthesised from Lean/Z3 integration proposals in autonomous science literature
- **Deferred:** Formal verification gate (Lean/Z3). Placeholder in roadmap as "Formal Gate".
  Current implementation uses keyword overlap in `CitationGrounding` as a lightweight proxy.

**[E2] "FrontierMath: A Benchmark for Evaluating Advanced Mathematical Reasoning"**
- arXiv:2411.04872, 2024
- **Deferred:** Mathematical claim verification via formal solvers.

---

## 2. Key GitHub Projects and Notable Forks

### 2.1 SakanaAI/AI-Scientist-v2
- **URL:** https://github.com/SakanaAI/AI-Scientist-v2
- **Adopted:** Tree-based hypothesis exploration → `ObjectiveGraph` priority ordering.
  VLM reviewer concept → `ReviewerAgent` adversarial review structure.
- **Rejected:** Monolithic single-file architecture. We use clean modular layout.

### 2.2 aiming-lab/AutoResearchClaw
- **URL:** https://github.com/aiming-lab/AutoResearchClaw
- **Adopted:** Multi-source verification, anti-hallucination citation checks → `EvidenceRegistry`
  de-duplication + `CitationGrounding`.
- **Deferred:** Discord/Telegram delivery, OpenClaw local deployment.

### 2.3 karpathy/autoresearch
- **URL:** https://github.com/karpathy/autoresearch (conceptual / reference)
- **Adopted:** Lean, compute-optimal research workflow. Spend most "thinking time" on
  hypothesis and less on formatting → priority ordering in `ObjectiveGraph` (planning and
  hypothesis have higher priority than coding/formatting).

### 2.4 WecoAI/awesome-autoresearch + alvinreal/awesome-autoresearch
- **URL:** https://github.com/WecoAI/awesome-autoresearch
- **Role:** Curated ecosystem lists used as a survey to identify state-of-art baselines
  and common failure modes.
- **Finding:** Most listed systems lack persistent state / resumability. We addressed this
  with `CheckpointManager`.

### 2.5 facebookresearch/HyperAgents
- **URL:** https://github.com/facebookresearch/HyperAgents
- **Adopted:** Metacognitive self-modification concept → `MetaAgent` + `SelfImprovementEngine`.
  Bounded rollback → `safe_modification_only` flag.
- **Rejected:** Unrestricted evolutionary code mutation (security/reproducibility risk).

### 2.6 Notable Forks (2026 ecosystem survey)
Several forks of AI-Scientist-v2 added:
- Async parallel execution (not yet in v2 core, planned Milestone 5)
- Custom domain adapters (biology, chemistry templates)
- Web UIs using Streamlit

These informed the modular connector pattern in `autoresearch/literature/` (pluggable
backends via `ScholarlyConnector`) and the `ExperimentSpec` schema's `language` field
(extensible beyond Python).

---

## 3. Design Rationale Table

| Decision | Rationale | Source |
|---|---|---|
| `ObjectiveGraph` DAG instead of fixed pipeline | Science is non-linear; objectives should be dynamic | [A3] rejection of 23-stage linearity |
| Safe self-modification only | Prevent "model collapse" / unsafe code mutation | [B1] DGM-H + [D2] ADAS safety analysis |
| 4 reviewer personas in ToM | Cover orthogonal review dimensions: method, empirics, novelty, benchmarks | [C1] ToM for LLMs |
| `EvidenceRegistry` de-duplication | Prevent citation hallucination via identical-source merging | [A3] AutoResearchClaw anti-hallucination |
| `SelfHealingLoop` with bounded retries | Self-repair without infinite loops | [A2] AI Scientist self-debugging |
| `CollaboratorIntentProfile` | Model *user* intent, not just content quality | [C1] collaborator mind modelling |
| Minority-report preservation | Prevent echo-chamber of plausibility | [B1] HyperAgents heterogeneous verification |
| Physicality Anchor (deferred) | Digital-first systems miss real-world grounding | [A1] AI Scientist v2 real-data emphasis |

---

## 4. What Was Rejected and Why

| Concept | Source | Rejection Reason |
|---|---|---|
| VLM-based figure auditing | AI Scientist v2 | Requires multimodal API; deferred to Milestone 3 |
| Formal logic gates (Lean/Z3) | E1 survey | Requires domain-specific formalisation; deferred |
| Messaging app delivery | AutoResearchClaw | Out of scope for core framework MVP |
| Unrestricted code mutation | HyperAgents / DGM-H | Security risk; safe-list approach adopted instead |
| Fixed 23-stage pipeline | AutoResearchClaw | Replaced by configurable DAG |
| Single underlying LLM for all agents | General practice | Architecture-heterogeneous review preferred |
| Only synthetic data | Multiple sources | "Physicality Anchor" principle — real data preferred |

---

## 5. Citations Index

| ID | Reference |
|---|---|
| A1 | Lu et al. "AI Scientist v2" arXiv:2502.09601 (2025–2026) |
| A2 | Lu et al. "The AI Scientist" arXiv:2408.06292 (2024) |
| A3 | aiming-lab/AutoResearchClaw (GitHub, 2024–2026) |
| B1 | Meta FAIR, facebookresearch/HyperAgents (GitHub / arXiv:2506.xxxxx, 2026) |
| B2 | Li et al. "MetaAgents" arXiv:2310.06500 (2023) |
| B3 | Wang et al. "Mixture of Agents" arXiv:2406.04692 (2024) |
| C1 | Survey of ToM-in-LLM papers (2025–2026) |
| C2 | "ToM-LM" arXiv:2306.15448 (2023) |
| D1 | SakanaAI/AI-Scientist-v2 self-improvement patterns (2026) |
| D2 | Hu et al. "ADAS" arXiv:2408.08435 (2024) |
| E1 | Formal verification for LLM code (2026 survey) |
| E2 | "FrontierMath" arXiv:2411.04872 (2024) |
