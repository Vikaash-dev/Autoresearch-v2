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

## Implementation Bootstrap

An initial Python implementation scaffold now exists under `src/autoresearch_v2/`:

- `dog/` — Objective model and dynamic graph scheduling primitives
- `discovery/` — Branch lifecycle and forest pruning primitives
- `tom/` — ToM reviewer council baseline
- `epistemic/` — Zero-trust verification baseline
- `core/runtime.py` — End-to-end bootstrap run wiring
- `cli.py` — Minimal CLI entrypoint

## Retrieval Capability Matrix

| Capability | Status |
|---|---|
| Offline semantic retrieval over local corpus | ✅ Enabled by default |
| API-free operation in restricted environments | ✅ Supported |
| arXiv/OpenAlex/SemanticScholar online connectors | ⚠️ Present as stubs / optional future adapters |
| Ranked evidence output for literature grounding | ✅ Enabled in `LiteratureAgent` |

The bootstrap runtime now defaults to a local semantic retriever and can run without external network APIs.
It includes seeded entries for Hyperagents (2026) and related self-evolving-agent works, with explicit
disambiguation from HyperAgent (2024).

Run bootstrap CLI:

```bash
PYTHONPATH=src python -m autoresearch_v2.cli --topic "self-correcting LLMs" --branches 3
```
