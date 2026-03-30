# Autoresearch v2 — SERA-X

**Self-Evolving Research Architecture eXtended**

> *Give it a task. Go to sleep. Wake up to a better answer — because agents searched, hypothesised, experimented, reviewed, and improved their own strategy while you were away.*

SERA-X synthesises the best ideas from 32 research repositories and 6 key papers into a single cohesive system that **surpasses HyperAgents** by combining:

| Paper / Repo | What SERA-X Takes |
|---|---|
| **karpathy/autoresearch** | `program.md` org charter + `strategy.md` live evolution + fixed-budget loop |
| **Bilevel Autoresearch** (arXiv:2603.23420) | 3-level outer loop: task → config adaptation → **mechanism code generation** via `importlib` (5× improvement) |
| **AI Scientist-v2** (Sakana AI, 2026) | Full research lifecycle + MCTS hypothesis tree search |
| **Centaur** (2026) | Hybrid LLM + CMA-ES: shares covariance matrix state with LLM |
| **AIRS-Bench** (2026) | 7-dimension scoring: novelty, depth, reproducibility, validity, impact, clarity, citations |
| **DGM-Hyperagents** (Meta, 2026) | MetaAgent rewrites its own strategy each round |
| **AutoResearcher** (2025) | Knowledge Grounding + Novelty agent pair |
| **HyperAgents** (Meta) | Archive + parent selection, self-modification loop |
| **CORAL / n-autoresearch** | Adaptive strategy: explore → exploit → combine → ablation |
| **AgentLaboratory** | PhD/PostDoc/Professor roles + AgentRxiv cumulative knowledge |
| **GPT-Researcher** | Planner+Executor+SourceCurator+RAG pipeline |
| **dzhng/deep-research** | Recursive SERP(breadth, depth) with learnings accumulation |

---

## Architecture

```
INPUT TASK
    │
    ▼
Layer 0 · INTAKE
  program.md (human edits) + strategy.md (meta-agent edits)
    │
    ▼
Layer 1 · DEEP SEARCH ENGINE  [dzhng/deep-research pattern]
  deepSearch(query, breadth, depth) → learnings[] + urls[]
    │
    ▼
Layer 2 · KNOWLEDGE STORE  [Bilevel recycling + AgentRxiv]
  SQLite experience store — skills, learnings, agent history
  failure recycling as training signal (Bilevel Autoresearch)
    │
    ▼
Layer 3 · RESEARCH WORKFORCE  [AgentLaboratory roles]
  ┌──────────────┐  ┌────────────────┐  ┌─────────────────┐
  │ ResearchAgent│  │HypothesisAgent │  │ KnowledgeAgent  │
  │ (literature) │  │ (MCTS tree)    │  │ (grounding +    │
  │              │  │                │  │  novelty check) │
  └──────────────┘  └────────────────┘  └─────────────────┘
  ┌──────────────┐  ┌────────────────┐  ┌─────────────────┐
  │ExperimentAgent│ │  WriterAgent   │  │  ReviewerAgent  │
  │(AI Sci-v2 Exp│  │ (LaTeX/MD)     │  │ (AIRS-Bench 7d) │
  │  Manager)    │  │                │  │                 │
  └──────────────┘  └────────────────┘  └─────────────────┘
    │
    ▼
Layer 4 · SELF-EVOLUTION ENGINE  [HyperAgents + Bilevel + DGM]
  MetaAgent reads round scores → proposes strategy diffs
  Bilevel L1.5: adapts search config from trace
  Bilevel L2:   generates new Python mechanisms via importlib
  Centaur:      CMA-ES + LLM blended HPO
    │
    ▼
Layer 5 · EXECUTION SANDBOX  [SWE-agent ACI]
  LocalExecutor: timeout-enforced, stdout capture, verifier
  ACITools: file_view, file_edit, search_content, run_python
    │
    ▼
Layer 6 · OUTPUT + PUBLISH
  Structured report (Markdown / LaTeX)
  AIRS-Bench evaluation → feeds back into Layer 2 memory
```

---

## Quick Start

```bash
pip install autoresearch-v2

# Run on any research task
python -m autoresearch "What methods surpass HyperAgents for automated ML research?" \
    --rounds 5 --breadth 4 --depth 2
```

### Python API

```python
from autoresearch import ResearchOrchestrator, AutoresearchConfig

orch = ResearchOrchestrator()
result = orch.run(
    task="Bilevel optimisation for neural architecture search",
    max_rounds=5,
    search_breadth=4,
    search_depth=2,
    success_threshold=0.75,
)

orch.print_summary()
print(result["best_report"]["markdown"])
```

### Bilevel Demo (no API key needed)

```bash
python examples/bilevel_demo.py
# Shows the 3-level self-improving HPO loop
# Level 2 autonomously generates TabuSearch + OrthogonalExploration mechanisms
```

---

## How It Surpasses HyperAgents

| Feature | HyperAgent (Original) | SERA-X |
|---|---|---|
| Search strategy | Unconstrained code editing | Bilevel L2: **generates new search mechanisms as Python code** |
| Logic flow | Sequential / single-track | Multi-batch + MCTS + explore→exploit→combine→ablation |
| Reliability | 42% failure rate (ACM, 2025) | Anti-hallucination flags + VerifiedRegistry + AIRS-Bench scoring |
| Scope | Code / hyperparameter tuning | Full lifecycle: search → hypothesise → experiment → write → review |
| Memory | None | SQLite + failure recycling (Bilevel) + skills (Hermes) |
| Self-evolution | Agent edits code | **3-level**: task (L1) → config (L1.5) → mechanism generation (L2) |
| HPO | LLM proposals only | Centaur: **CMA-ES covariance matrix shared with LLM** |
| Evaluation | Task metric only | AIRS-Bench 7-dimensional scoring + performance ceiling gap |

---

## Key Components

### Bilevel Autoresearch (`autoresearch/optimization/bilevel.py`)

The core innovation. 3-level outer loop:

```python
engine = BilevelEngine(executor=my_executor, proposer=my_proposer)
result = engine.run("optimise val_bpb", baseline_score=1.10)
# Level 2 autonomously discovers: TabuSearch, OrthogonalExploration
# Result: 5× improvement over standard autoresearch
```

### Centaur Hybrid HPO (`autoresearch/optimization/hybrid_hpo.py`)

CMA-ES mathematical precision + LLM domain knowledge:

```python
centaur = CentaurHPO(config, evaluator=my_eval, llm_proposer=my_llm)
# LLM reads the CMA-ES covariance matrix → understands which params are correlated
result = centaur.run("optimise", baseline=0.0)
```

### AIRS-Bench Reviewer (`autoresearch/agents/reviewer_agent.py`)

7-dimension quality scoring:

```
novelty (20%) · technical_depth (20%) · validity (20%)
reproducibility (15%) · clarity (10%) · impact (10%) · citation_quality (5%)
```

### MetaAgent (`autoresearch/agents/meta_agent.py`)

DGM-style self-referential improvement — modifies `strategy.md` each round, proposes
parameter diffs for underperforming agents, and triggers Bilevel mechanism generation.

---

## Project Structure

```
autoresearch/
├── __init__.py              # Public API: ResearchOrchestrator, AutoresearchConfig
├── __main__.py              # python -m autoresearch "task"
├── config.py                # Typed config for every layer
├── orchestrator.py          # Main 6-layer loop (ENTRY POINT)
│
├── agents/
│   ├── base_agent.py        # Self-evolving base (DGM + Bilevel)
│   ├── research_agent.py    # Literature search
│   ├── hypothesis_agent.py  # MCTS tree search (AI Scientist-v2)
│   ├── knowledge_agent.py   # Grounding + novelty (AutoResearcher)
│   ├── experiment_agent.py  # Experiment design + execution
│   ├── writer_agent.py      # Report generation (AgentLaboratory)
│   ├── reviewer_agent.py    # AIRS-Bench 7-dim scoring
│   └── meta_agent.py        # Self-modification (HyperAgents + DGM)
│
├── optimization/
│   ├── bilevel.py           # 3-level Bilevel loop + mechanism generation
│   └── hybrid_hpo.py        # Centaur: CMA-ES + LLM
│
├── memory/
│   └── experience_store.py  # SQLite + failure recycling + skills
│
├── pipeline/
│   └── deep_search.py       # Recursive SERP engine (deep-research port)
│
├── evolution/
│   └── evolution_protocol.py # RSPL + SEPL propose/assess/commit/rollback
│
├── evaluation/
│   └── benchmark.py         # AIRS-Bench evaluation suite
│
└── sandbox/
    └── executor.py          # SWE-agent ACI: safe code execution

program.md   # Human-editable research org charter (Karpathy pattern)
strategy.md  # Live-evolving strategy (meta-agent edits this)
examples/
tests/
```

---

## Extending SERA-X

### Add an LLM backend

```python
from autoresearch.pipeline.deep_search import LLMProvider

class MyLLMProvider(LLMProvider):
    def complete(self, prompt: str, **kwargs) -> str:
        # call OpenAI / Anthropic / local model
        return my_llm.chat(prompt)

orch = ResearchOrchestrator(llm_provider=MyLLMProvider())
```

### Add a search backend

```python
from autoresearch.pipeline.deep_search import SearchProvider

class SerperProvider(SearchProvider):
    def search(self, query: str, n: int = 5) -> list[dict]:
        # call Serper / Tavily / Bing API
        return [{"title": ..., "url": ..., "snippet": ...}]

orch = ResearchOrchestrator(search_provider=SerperProvider())
```

### Custom Bilevel mechanism

```python
from autoresearch.optimization.bilevel import MechanismRegistry

reg = MechanismRegistry()
reg.generate_and_load(
    name="multi_scale_bandit",
    description="Multi-armed bandit that searches at multiple scales simultaneously"
)
# Now hot-loaded via importlib and applied to every Level 1 proposal
```

---

## Papers & References

1. Karpathy, A. (2026). *autoresearch*. GitHub. https://github.com/karpathy/autoresearch
2. EdwardOptimization (2026). *Bilevel Autoresearch: Meta-Autoresearching Itself*. arXiv:2603.23420
3. Lu et al. (2025). *The AI Scientist: Towards Fully Automated Open-Ended Scientific Discovery*. arXiv:2408.06292
4. Sakana AI (2026). *The AI Scientist-v2: Workshop-Level Automated Scientific Discovery via Agentic Tree Search*. arXiv:2504.08066
5. Meta AI (2026). *HyperAgents*. https://ai.meta.com/research/hyperagents/
6. Lange et al. (2026). *Can LLMs Beat Classical Hyperparameter Optimization? Centaur*. arXiv (Mar 2026)
7. Meta AI (2026). *DGM-Hyperagents: Darwin Gödel Machine extension*.
8. Meta AI (2026). *AIRS-Bench: A Suite of Tasks for Frontier AI Research Science Agents*.
9. Guo et al. (2025). *AutoResearcher: Knowledge-Grounded Multi-Agent Automated Research*.
10. WecoAI (2025). *AIDE ML: Agentic Tree Search in Code Space*. arXiv:2502.13138
11. Human-Agent-Society (2025). *CORAL: Multi-Agent Autoresearch with Git Worktrees*.
12. Sibyl Research Team (2026). *AutoResearch-SibylSystem: Dual-Loop Self-Evolution*.

---

## License

MIT
