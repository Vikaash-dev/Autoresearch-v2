# Autoresearch-v2

This project aims to surpass current auto paper generation, auto research, and multi-agent setups by using self-evolving agents that modify themselves based on the task. It incorporates insights from the latest research papers and new GitHub approaches as a starting point, building on existing work.

---

## Motivation & Research Context

The automated research landscape has evolved rapidly from 2025 to early 2026. While Andrej Karpathy's autoresearch project and Meta's HyperAgents popularized the idea of an AI agent that iteratively edits its own training code, newer research addresses their single-track nature and reliability bottlenecks. This project is informed by the following key works.

---

## Key Papers Surpassing Initial HyperAgent Proposals

### 1. Bilevel Autoresearch (2026)
Addresses the "single-track iteration" limitation of Karpathy's original proposal. Introduces a **multi-batch persistent experience model** that identifies bottlenecks and optimises hyperparameter search as a bilevel problem, rather than simple trial-and-error.

### 2. The AI Scientist-v2 (2026) — Sakana AI
Extends the original AI Scientist by removing human-coded templates and using an **experiment manager agent** with **progressive agentic tree search** for hypothesis generation. Notable for producing the first fully AI-generated paper accepted at a scientific conference. Cites the v1 paper and directly addresses its "shallow experimentation" limitation.

### 3. Centaur (2026) — "Can LLMs Beat Classical Hyperparameter Optimization?"
A hybrid system that surpasses pure LLM-based HyperAgents by sharing internal states (covariance matrices, step-size) from classical algorithms like **CMA-ES** with the LLM. Combines human-like domain knowledge with mathematical precision where LLMs typically struggle.
- arXiv: *Can LLMs Beat Classical Hyperparameter Optimization?* (March 2026)

### 4. AIRS-Bench (2026)
A benchmark suite for **Frontier AI Research Science Agents**. Highlights that current agents frequently fail to match human state-of-the-art (SOTA) in complex research tasks and provides a roadmap for next-generation agents targeting "theoretical performance ceilings".
- Published: February 10, 2026 (NLP track, AI at Meta)

### 5. AutoResearcher (2025)
A **knowledge-grounded** multi-agent system including dedicated "Knowledge Grounding" and "Novelty" agents to ensure generated ideas are scientifically sound and original — unlike HyperAgent, which focuses primarily on code execution.

### 6. DGM-Hyperagents (March 2026) — Meta Research
Extends the Darwin Gödel Machine (DGM) framework so that the **improvement procedure itself evolves** — the agent does not just improve at a task, but improves at learning how to improve. Features persistent memory and performance tracking that emerged without explicit programming.

---

## Papers Citing HyperAgents & The AI Scientist

| Paper | Focus Area | Cited Improvement over Predecessors |
|---|---|---|
| AI Scientist-v2 (Sakana AI, Apr 2025) | Autonomy | Removes human code templates; uses agentic tree search |
| DGM-Hyperagents (Meta, Mar 2026) | Meta-Learning | Self-referential improvement of the "Meta Agent" itself |
| Evaluating Sakana's AI Scientist (ACM, Oct 2025) | Reliability | Benchmarks high failure rates (42%) and hallucinated data |
| A Survey on the Rise of the AI Scientists (May 2025) | Overview | Synthesises transition from LLM-assistants to full AI Scientists |
| Toward Super Agent System with Hybrid AI Routers (Apr 2025) | Efficiency | Hybrid routers cut cost of HyperAgent-style iterative loops |

---

## Comparison: HyperAgent vs. Newer Approaches

| Feature | HyperAgent (Original) | Newer Proposals (e.g., Centaur, Bilevel) |
|---|---|---|
| Search Strategy | Unconstrained code editing | Hybrid (LLM + Classical HPO) |
| Logic Flow | Sequential / Single-track | Multi-batch / Parallel tree search |
| Reliability | High failure rate in complex tasks | Focused on "recycling failures" and persistent memory |
| Scope | Code / Hyperparameter tuning | Full paper writing and hypothesis testing |

---

## References

1. Karpathy, A. *Autoresearch* (2026). [GitHub](https://github.com/karpathy/autoresearch)
2. Meta AI. *HyperAgents* (2026). [AI at Meta](https://ai.meta.com)
3. Sakana AI. *The AI Scientist-v2: Workshop-Level Automated Scientific Discovery via Agentic Tree Search* (April 2025).
4. Meta Research. *DGM-Hyperagents* (March 2026).
5. *Can LLMs Beat Classical Hyperparameter Optimization? Centaur* (March 2026). arXiv.
6. *AIRS-Bench: a Suite of Tasks for Frontier AI Research Science Agents* (February 2026). AI at Meta.
7. *AutoResearcher* (2025).
8. *Evaluating Sakana's AI Scientist: Bold Claims, Mixed Results* (October 2025). ACM.
9. *A Survey on the Rise of the AI Scientists* (May 2025).
10. *Toward Super Agent System with Hybrid AI Routers* (April 2025).

