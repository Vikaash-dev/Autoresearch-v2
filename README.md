# Autoresearch-v2

AutoResearch v2 aims to surpass current auto paper generation, auto research and multi-agent setups by using self-evolving agents that adapt to each task, incorporate lessons from research papers and new GitHub approaches, and build on existing work as a starting point.

## Proposals

| Document | Description |
|---|---|
| [`docs/proposals/autoresearch_v2_hrde.md`](docs/proposals/autoresearch_v2_hrde.md) | **Main proposal:** AutoResearch v2 — Hyper-Recursive Discovery Engine (HRDE). Covers the full architecture, Dynamic Objective Graph orchestration, Hyper-Kernel self-improvement, Discovery Forest execution, ToM Reviewer Council, Zero-Trust Epistemic Verification, compute-optimal planning, safety model, evaluation plan, system comparison table, implementation blueprint, and 30/60/90-day roadmap. |
| [`docs/proposals/autoresearch_v2_tom_epistemics.md`](docs/proposals/autoresearch_v2_tom_epistemics.md) | **Companion doc:** Detailed specification of the Theory-of-Mind (ToM) Reviewer Council, Collaborator Intent Modeling, and the Zero-Trust Epistemic Verification Stack (Literature Gate, Code Gate, Formal Gate, Claim Provenance Graph). |

## Key Ideas

- **Dynamic Objective Graph (DOG):** Replaces static pipelines with a dependency-driven graph of research objectives, dynamically reprioritized and extended during a run.
- **Hyper-Kernel:** A meta-agent that monitors sub-agent telemetry and modifies their prompt policies, tool configurations, and routing logic — bounded by safety constraints.
- **Discovery Forest:** Parallel hypothesis branches with branch spawning, pruning, and cross-branch grafting to efficiently explore large hypothesis spaces.
- **ToM Reviewer Council:** Five reviewer personas (Skeptic, Engineer, Visionary, Domain Expert, Ethicist) that conduct a 3-round adversarial dialogue with the Author Agent before a manuscript is accepted.
- **Zero-Trust Epistemic Stack:** Every claim must pass three independent gates (literature consistency, code-log correspondence, and optional formal verification) before inclusion in output.
- **Skill Registry:** Persistent cross-run memory that stores lessons learned, inspired by the engram pattern from [tonitangpotato/autoresearch-engram](https://github.com/tonitangpotato/autoresearch-engram).
