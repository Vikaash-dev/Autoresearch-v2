# Research Framework and arXiv Review (Environment-Constrained Snapshot)

Date: 2026-04-03

## Scope

Requested: review research-agent frameworks on GitHub and new arXiv research papers before finishing implementation.

## Access Constraints Observed

- arXiv endpoints were unreachable from this execution environment (`fetch failed` for arxiv.org/export/rss endpoints).
- Several general web research domains were also blocked/unreachable.

## GitHub Framework Snapshot (retrieved via GitHub search/API)

### High-signal frameworks observed

1. **microsoft/autogen**
   - Multi-agent framework with layered architecture and explicit runtime abstractions.
   - Strong emphasis on orchestration patterns, tools, and extensibility.
2. **langchain-ai/langgraph**
   - Stateful graph orchestration with durable execution and human-in-the-loop controls.
   - Clear fit for long-running research workflows.
3. **crewAIInc/crewAI**
   - Role-based multi-agent collaboration and production workflow framing.
4. **FoundationAgents/MetaGPT**
   - Multi-agent software-company metaphor and role decomposition.
5. **camel-ai/camel**
   - Multi-agent communication and scaling-law framing.
6. **assafelovic/gpt-researcher**
   - Research-specific planner/executor pipeline with citation-oriented output.

## Extracted design implications mapped to AutoResearch v2

- **Graph-first orchestration** is table stakes → reflected in DOG + scheduler.
- **Durable/inspectable state** is critical → reflected in provenance graph, audit log, telemetry structures.
- **Role-specialized agents** improve decomposition → reflected in literature/hypothesis/coder/writer/critic agents.
- **Adversarial review before acceptance** improves output quality → reflected in ToM dialogue protocol.
- **Evidence-gated claims** are necessary for trust → reflected in epistemic 3-gate pipeline skeleton.
- **Safety/human escalation hooks** are required for sensitive domains → reflected in domain classifier + guardrails.

## arXiv review status

Direct arXiv review could not be completed due to network access restrictions from this environment.  
To compensate, the implementation preserves explicit extension points (`integrations/arxiv_client.py`, epistemic gates, provenance) so real arXiv-backed retrieval and validation can be plugged in without architecture changes once connectivity is available.

## Implementation impact summary

This review directly informed the completion of missing MVP modules and wiring:

- DOG: scheduler + allocator
- Discovery: pruner + grafter
- ToM: dialogue + intent model + persona stubs
- Epistemic: literature/code/formal gates + provenance + pipeline
- Agents: base + 5 functional stubs
- Compute: planner + simulator
- Safety: classifier + guardrails + audit log
- Memory: research graph + engram
- Integrations: arXiv/OpenAlex/SemanticScholar/sandbox client stubs
- Hyper-kernel: telemetry + modifier + registry + kernel

