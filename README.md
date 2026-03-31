# Autoresearch-v2

**The self-evolving autonomous research system — surpassing AI-Scientist v2.**

Autoresearch-v2 combines the best ideas from the 2026 autonomous research landscape into a unified, self-improving pipeline that generates novel, publishable research while continuously improving its own agents, prompts, and skills.

> *Original goal: surpass current auto paper generation, auto research and multi-agent setups by using self-evolving agents which make changes to themselves based on the task, incorporating changes using research papers and new GitHub approaches, using existing work as a starting point.*

---

## What This Surpasses

| System | What it does well | What it lacks |
|---|---|---|
| [SakanaAI/AI-Scientist v2](https://github.com/SakanaAI/AI-Scientist-v2) | BFTS tree search, paper writing | No self-evolution, no ToM, templates needed |
| [karpathy/autoresearch](https://github.com/karpathy/autoresearch) | Elegant modify-evaluate-keep loop | Single file, no paper generation, no ToM |
| [NousResearch/hermes-agent](https://github.com/NousResearch/hermes-agent) | Skills system, user modeling, memory | Not research-focused |
| [WecoAI/aideml](https://github.com/WecoAI/aideml) | LLM tree search in code space | No hypothesis generation, no writing |

Autoresearch-v2 integrates all of these while adding:
- **7-layer Theory of Mind** (user, agent, self, reviewer, task, tool, community)
- **GEPA self-evolution** (ICLR 2026 Oral — 35x faster than RL, 90x cheaper)
- **Adversarial Reviewer-ToM** (predicts and addresses reviewer concerns before writing)
- **Community-ToM** (targets open problems, avoids overcrowded areas)
- **Cross-run memory** (every run improves from all previous runs)

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    UNIFIED ToM STATE STORE                       │
│   User-ToM | Agent-ToM | Self-ToM | Reviewer-ToM | Community-ToM│
└──────────────────────────┬──────────────────────────────────────┘
                           │ reads/writes
┌──────────────────────────▼──────────────────────────────────────┐
│                       BLACKBOARD                                 │
│   ExperimentNodes | AgentBeliefs | ToM observations | RunState  │
└──┬──────────┬──────────┬──────────┬──────────┬──────────────────┘
   │          │          │          │          │
   ▼          ▼          ▼          ▼          ▼
Hypothesis  Experiment  Literature  Writer   Reviewer
 Agent       Agent       Agent      Agent    Agent
   │                                          │
   └──────────┬───────────────────────────────┘
              │ orchestrates
┌─────────────▼──────────────────────────────────────────────────┐
│               BEST-FIRST TREE SEARCH (BFTS)                     │
│   UCB1 priority | stage-gated | parallel | debug-retry         │
└─────────────────────────────────────────────────────────────────┘
              │ triggers after run
┌─────────────▼──────────────────────────────────────────────────┐
│           GEPA SELF-EVOLUTION ENGINE                            │
│  Read trace → Reflect → Mutate → Pareto select → Constraint gate│
│  Evolves: skills, tool descriptions, system prompts, code       │
└─────────────────────────────────────────────────────────────────┘
```

---

## Key Papers & Repos (2026)

| Source | Key Contribution | Integrated |
|---|---|---|
| [AI-Scientist v2](https://arxiv.org/abs/2504.08066) (SakanaAI) | BFTS for scientific experiments | ✅ `core/tree_search.py` |
| [GEPA](https://arxiv.org/abs/2507.19457) (ICLR 2026 Oral) | Genetic-Pareto prompt/code evolution | ✅ `evolution/gepa_optimizer.py` |
| [hermes-agent](https://github.com/NousResearch/hermes-agent) | Skills, memory, user modeling | ✅ `agents/`, `memory/` |
| [hermes-agent-self-evolution](https://github.com/NousResearch/hermes-agent-self-evolution) | GEPA + DSPy skill evolution | ✅ `evolution/` |
| [karpathy/autoresearch](https://github.com/karpathy/autoresearch) (March 2026) | Modify-evaluate-keep loop | ✅ `core/loop.py`, `program.md` |
| [atropos](https://github.com/NousResearch/atropos) | RL gym for LLM trajectories | 🔲 `memory/trajectory_log.py` (planned) |
| [autonovel](https://github.com/NousResearch/autonovel) | Adversarial revision loop | ✅ `agents/reviewer_agent.py` |
| [honcho](https://github.com/plastic-labs/honcho) | Dialectic user modeling (ToM) | ✅ `tom/engine.py` |
| [AIDE](https://github.com/WecoAI/aideml) | LLM tree search in code space | ✅ `core/tree_search.py` |
| [DSPy](https://github.com/stanfordnlp/dspy) | Declarative LLM pipelines | ✅ `requirements.txt` |

---

## Quick Start

```bash
# 1. Clone and install
git clone https://github.com/Vikaash-dev/Autoresearch-v2
cd Autoresearch-v2
pip install -r requirements.txt

# 2. Set API keys
export OPENAI_API_KEY="your-key"
export S2_API_KEY="your-s2-key"  # optional, for Semantic Scholar

# 3. Dry run (no API calls, tests the pipeline)
python launch.py --idea-text "Improve few-shot learning on long-tail tasks" --dry-run

# 4. Real run
python launch.py \
  --idea-text "Efficient attention mechanisms for long-context transformers" \
  --model gpt-4o \
  --model-writeup o1-preview-2024-09-12 \
  --model-review claude-opus-4 \
  --venue NeurIPS \
  --user-description "ML researcher, want publishable ICML result, have 4xA100"
```

---

## File Structure

```
autoresearch_v2/
├── core/
│   ├── blackboard.py       # Shared multi-agent state store (thread-safe, persistent)
│   ├── loop.py             # Modify-evaluate-keep/discard loop (karpathy-style)
│   └── tree_search.py      # BFTS with UCB1 priority + debug-retry (AI-Scientist v2)
├── agents/
│   ├── base_agent.py       # Base: ToM access, failure tracking, evolution triggers
│   ├── hypothesis_agent.py # Generates novel hypotheses (Community-ToM guided)
│   ├── experiment_agent.py # Executes experiments, parses metrics (sandboxed)
│   ├── literature_agent.py # Searches arXiv + Semantic Scholar
│   ├── writer_agent.py     # Writes paper drafts (Reviewer-ToM guided)
│   ├── reviewer_agent.py   # Adversarial dual-persona review
│   └── orchestrator.py     # Coordinates all agents through pipeline stages
├── evolution/
│   └── gepa_optimizer.py   # GEPA: Genetic-Pareto evolution for prompts/skills/code
├── memory/                 # Skill store, trajectory log (planned)
├── tom/
│   └── engine.py           # 7-layer Theory of Mind engine
├── tools/                  # Additional tools (planned)
├── config/
│   └── config.yaml         # BFTS + GEPA + model + ToM configuration
├── program.md              # Human-programmable research org instructions
├── launch.py               # Main entry point
└── requirements.txt
```

---

## The 7 ToM Layers

| Layer | What It Models | Where Used |
|---|---|---|
| **User-ToM** | Intent, style, domain, trust level | `orchestrator.run()` — guides whole pipeline |
| **Agent-ToM** | What other agents know/tried/believe | `base_agent.observe_other_agents()` — prevents duplicates |
| **Self-ToM** | Own failure patterns, blind spots | `tom.analyze_self()` → GEPA evolution trigger |
| **Reviewer-ToM** | Venue-specific reviewer concerns | `writer_agent._write_draft()` — preemptive fix |
| **Task-ToM** | Decompose underspecified task text | `orchestrator.run()` before Stage 1 |
| **Tool-ToM** | Tool capabilities and failure modes | *(planned: `tools/tool_registry.py`)* |
| **Community-ToM** | Field consensus, debates, open problems | `hypothesis_agent.expand()` — targets gaps |

---

## Self-Evolution Loop

After every run, the GEPA engine analyzes Self-ToM observations and:
1. Identifies subtasks with ≥3 consecutive failures
2. Reads full execution traces (Actionable Side Information)
3. Proposes targeted mutations to the responsible skill/prompt
4. Evaluates mutations through constraint gates (tests + size + semantic preservation)
5. Promotes successful variants via PR (never direct commit)

```
Run 1: hypothesis_agent fails "expand" 4 times
  → GEPA evolves hypothesis_agent system prompt
  → mutation: "Prompt doesn't specify avoiding overcrowded areas"
  → new prompt: 68% → 81% success rate on eval set

Run 2: uses evolved prompt → fewer wasted tree nodes
  → Self-ToM sees improvement → GEPA targets next bottleneck
```

---

## Configuration

Edit `config/config.yaml` to tune tree search, LLM models, GEPA, and ToM parameters.

Edit `program.md` to tune research philosophy, writing guidelines, and evolution constraints — this is the "research org code" you program, inspired by karpathy/autoresearch.

---

## License

MIT — contributions welcome.
