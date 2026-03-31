"""
HyperOptimizer — Meta-level optimization that evolves optimizer strategies.

Based on the HyperAgents meta-optimization paradigm (Meta AI / AgentAugi):
  https://ai.meta.com/research/publications/hyperagents/
  https://github.com/tymines/AgentAugi

Core insight
------------
Standard self-evolving systems (GEPA, TextGrad, MIPRO, AFlow) optimize
*what* agents do — their prompts, skills, and code.

HyperOptimizer operates one level higher: it optimizes *how* agents are
optimized.  It maintains a population of OptimizerStrategy objects, each
describing a full optimization recipe (algorithm choice + hyperparameters +
mutation operators).  LLM-guided mutation and crossover breed better
strategies.  A fitness tracker records which strategy produced the best
improvement on which task type, and the best-matched strategy is selected for
each new task.

Architecture position
---------------------
    User intent (Task-ToM)
         │
    HyperOptimizer                  ← this module
    ├── selects best strategy for task type
    ├── evolves strategies across runs
    └── delegates execution to chosen strategy:
         ├── GEPAOptimizer           (evolution/gepa_optimizer.py)
         ├── TextGradStrategy        (gradient-based prompt tuning)
         ├── AFlowStrategy           (MCTS workflow evolution)
         ├── MIPROStrategy           (model-agnostic iterative opt.)
         └── EvoPromptStrategy       (feedback-driven evolutionary)

Integration
-----------
    hyper_opt = HyperOptimizer.from_config(cfg)
    best_strategy = hyper_opt.select_strategy(task_type="hypothesis")
    result = best_strategy.run(artifact, evaluate_fn)
    hyper_opt.record_fitness(best_strategy, result.improvement)
    hyper_opt.evolve()  # mutate + crossover after N runs
"""

from __future__ import annotations

import json
import logging
import random
import time
import uuid
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────────────── #
#  Enumerations                                                               #
# ────────────────────────────────────────────────────────────────────────── #

class OptimizerAlgorithm(str, Enum):
    """Base optimization algorithms that a strategy can select."""
    GEPA       = "gepa"           # Genetic-Pareto (ICLR 2026 Oral)
    TEXTGRAD   = "textgrad"       # Gradient-based prompt opt (Nature 2025)
    AFLOW      = "aflow"          # MCTS workflow evolution (arXiv:2410.10762)
    MIPRO      = "mipro"          # Model-agnostic iterative opt (arXiv:2406.11695)
    EVOPROMPT  = "evoprompt"      # Feedback-driven evolutionary (arXiv:2309.08532)
    HYBRID     = "hybrid"         # LLM-generated novel combination


class TaskType(str, Enum):
    """Task types for strategy specialisation."""
    HYPOTHESIS     = "hypothesis"
    EXPERIMENT     = "experiment"
    LITERATURE     = "literature"
    WRITING        = "writing"
    REVIEW         = "review"
    TOOL_SELECTION = "tool_selection"
    GENERAL        = "general"


# ────────────────────────────────────────────────────────────────────────── #
#  OptimizerStrategy — one individual in the meta-optimization population    #
# ────────────────────────────────────────────────────────────────────────── #

@dataclass
class OptimizerStrategy:
    """
    One optimizer strategy in the HyperOptimizer population.

    Each strategy describes a complete optimization recipe:
      - which base algorithm to use
      - its hyperparameters
      - mutation operators to apply
      - task types it is specialised for
    """

    strategy_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    algorithm: OptimizerAlgorithm = OptimizerAlgorithm.GEPA
    task_types: list[TaskType] = field(default_factory=lambda: list(TaskType))

    # ── Shared hyperparameters ─────────────────────────────────────────── #
    mutation_temperature: float = 0.9
    population_size: int = 8
    max_generations: int = 20
    max_metric_calls: int = 150
    pareto_selection: bool = True
    elitism_ratio: float = 0.2      # fraction of top performers kept unchanged

    # ── Algorithm-specific knobs ──────────────────────────────────────── #
    # GEPA
    gepa_phase_mask: list[str] = field(
        default_factory=lambda: ["skills", "tools", "prompts", "code"]
    )
    # AFlow / MCTS
    mcts_exploration: float = 1.41
    mcts_max_depth: int = 8
    # TextGrad
    textgrad_lr: float = 0.3
    textgrad_steps: int = 10
    # MIPRO
    mipro_num_candidates: int = 10
    mipro_num_trials: int = 20
    # EvoPrompt
    evoprompt_crossover_prob: float = 0.7
    evoprompt_mutation_prob: float = 0.3

    # ── Mutation operators list ────────────────────────────────────────── #
    mutation_operators: list[str] = field(default_factory=lambda: [
        "parameter_nudge",       # small numeric change
        "operator_swap",         # swap one mutation operator for another
        "phase_toggle",          # enable/disable a GEPA phase
        "algorithm_blend",       # mix two strategies' hyperparameters
        "llm_propose",           # ask LLM to propose a new strategy
    ])

    # ── Fitness history ───────────────────────────────────────────────── #
    fitness_history: list[float] = field(default_factory=list)
    fitness_by_task: dict[str, list[float]] = field(default_factory=dict)
    generation: int = 0
    parent_ids: list[str] = field(default_factory=list)
    description: str = ""          # LLM-generated human-readable description

    # ── Computed ──────────────────────────────────────────────────────── #
    @property
    def mean_fitness(self) -> float:
        if not self.fitness_history:
            return 0.0
        return sum(self.fitness_history) / len(self.fitness_history)

    @property
    def task_fitness(self) -> dict[str, float]:
        return {
            t: sum(v) / len(v)
            for t, v in self.fitness_by_task.items()
            if v
        }

    def record_fitness(self, fitness: float, task_type: str = "general") -> None:
        self.fitness_history.append(fitness)
        self.fitness_by_task.setdefault(task_type, []).append(fitness)

    def to_dict(self) -> dict[str, Any]:
        d = {
            "strategy_id": self.strategy_id,
            "algorithm": self.algorithm.value,
            "task_types": [t.value for t in self.task_types],
            "mutation_temperature": self.mutation_temperature,
            "population_size": self.population_size,
            "max_generations": self.max_generations,
            "max_metric_calls": self.max_metric_calls,
            "pareto_selection": self.pareto_selection,
            "elitism_ratio": self.elitism_ratio,
            "gepa_phase_mask": self.gepa_phase_mask,
            "mcts_exploration": self.mcts_exploration,
            "mcts_max_depth": self.mcts_max_depth,
            "textgrad_lr": self.textgrad_lr,
            "textgrad_steps": self.textgrad_steps,
            "mipro_num_candidates": self.mipro_num_candidates,
            "mipro_num_trials": self.mipro_num_trials,
            "evoprompt_crossover_prob": self.evoprompt_crossover_prob,
            "evoprompt_mutation_prob": self.evoprompt_mutation_prob,
            "mutation_operators": self.mutation_operators,
            "fitness_history": self.fitness_history,
            "fitness_by_task": self.fitness_by_task,
            "generation": self.generation,
            "parent_ids": self.parent_ids,
            "description": self.description,
        }
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "OptimizerStrategy":
        s = cls(
            strategy_id=d.get("strategy_id", str(uuid.uuid4())[:8]),
            algorithm=OptimizerAlgorithm(d.get("algorithm", "gepa")),
            task_types=[TaskType(t) for t in d.get("task_types", [])],
            mutation_temperature=d.get("mutation_temperature", 0.9),
            population_size=d.get("population_size", 8),
            max_generations=d.get("max_generations", 20),
            max_metric_calls=d.get("max_metric_calls", 150),
            pareto_selection=d.get("pareto_selection", True),
            elitism_ratio=d.get("elitism_ratio", 0.2),
            gepa_phase_mask=d.get("gepa_phase_mask", ["skills", "tools", "prompts", "code"]),
            mcts_exploration=d.get("mcts_exploration", 1.41),
            mcts_max_depth=d.get("mcts_max_depth", 8),
            textgrad_lr=d.get("textgrad_lr", 0.3),
            textgrad_steps=d.get("textgrad_steps", 10),
            mipro_num_candidates=d.get("mipro_num_candidates", 10),
            mipro_num_trials=d.get("mipro_num_trials", 20),
            evoprompt_crossover_prob=d.get("evoprompt_crossover_prob", 0.7),
            evoprompt_mutation_prob=d.get("evoprompt_mutation_prob", 0.3),
            mutation_operators=d.get("mutation_operators", []),
            description=d.get("description", ""),
            generation=d.get("generation", 0),
            parent_ids=d.get("parent_ids", []),
        )
        s.fitness_history = d.get("fitness_history", [])
        s.fitness_by_task = d.get("fitness_by_task", {})
        return s


# ────────────────────────────────────────────────────────────────────────── #
#  HyperOptimizer                                                             #
# ────────────────────────────────────────────────────────────────────────── #

class HyperOptimizer:
    """
    Meta-level optimizer that evolves optimization strategies.

    The HyperOptimizer sits *above* individual optimization algorithms
    (GEPA, TextGrad, AFlow, MIPRO, EvoPrompt).  After each research run it:

    1. Records how much improvement each strategy produced.
    2. Ranks strategies by fitness (Pareto-optimal across improvement + cost).
    3. Mutates low-performing strategies (parameter nudge, operator swap,
       LLM-proposed novel hybrid).
    4. Breeds high-performing strategies via crossover.
    5. For the next task, selects the strategy with the best task-specific
       fitness history.

    Usage::

        hyper_opt = HyperOptimizer.from_config(cfg, llm_fn=my_llm)

        # Before a research run:
        strategy = hyper_opt.select_strategy(task_type="hypothesis")

        # After a research run:
        hyper_opt.record_fitness(strategy, improvement=0.08, task_type="hypothesis")

        # Periodically (every N runs):
        hyper_opt.evolve()
    """

    _SEED_STRATEGIES: list[dict[str, Any]] = [
        {"algorithm": "gepa",      "description": "GEPA baseline (Genetic-Pareto, ICLR 2026)"},
        {"algorithm": "textgrad",  "description": "TextGrad gradient-based prompt optimisation"},
        {"algorithm": "aflow",     "description": "AFlow MCTS workflow search"},
        {"algorithm": "mipro",     "description": "MIPRO model-agnostic iterative opt."},
        {"algorithm": "evoprompt", "description": "EvoPrompt feedback-driven evolution"},
    ]

    def __init__(
        self,
        *,
        population_size: int = 6,
        max_generations: int = 30,
        evolve_every_n_runs: int = 5,
        persist_path: str = "memory/hyper_optimizer_state.json",
        llm_fn: Any = None,
        rng_seed: int | None = None,
    ) -> None:
        self._pop_size = population_size
        self._max_gen = max_generations
        self._evolve_every = evolve_every_n_runs
        self._persist_path = Path(persist_path)
        self._llm = llm_fn
        self._rng = random.Random(rng_seed)

        self._population: list[OptimizerStrategy] = []
        self._run_counter: int = 0
        self._generation: int = 0
        self._run_history: list[dict[str, Any]] = []

        self._load_or_seed()

    # ------------------------------------------------------------------ #
    #  Construction                                                        #
    # ------------------------------------------------------------------ #

    @classmethod
    def from_config(cls, cfg: dict[str, Any], llm_fn: Any = None) -> "HyperOptimizer":
        ho = cfg.get("hyper_optimizer", {})
        return cls(
            population_size=ho.get("population_size", 6),
            max_generations=ho.get("max_generations", 30),
            evolve_every_n_runs=ho.get("evolve_every_n_runs", 5),
            persist_path=ho.get("persist_path", "memory/hyper_optimizer_state.json"),
            llm_fn=llm_fn,
            rng_seed=ho.get("rng_seed"),
        )

    def _load_or_seed(self) -> None:
        """Load persisted state or initialise from seed strategies."""
        if self._persist_path.exists():
            try:
                data = json.loads(self._persist_path.read_text())
                self._population = [
                    OptimizerStrategy.from_dict(s) for s in data.get("population", [])
                ]
                self._run_counter = data.get("run_counter", 0)
                self._generation = data.get("generation", 0)
                self._run_history = data.get("run_history", [])
                logger.info(
                    "HyperOptimizer: loaded %d strategies (gen %d)",
                    len(self._population), self._generation,
                )
                return
            except Exception as exc:
                logger.warning("HyperOptimizer: failed to load state (%s); re-seeding", exc)

        # Seed from built-in strategies
        for seed in self._SEED_STRATEGIES:
            s = OptimizerStrategy(
                algorithm=OptimizerAlgorithm(seed["algorithm"]),
                description=seed["description"],
                task_types=list(TaskType),
            )
            self._population.append(s)
        logger.info("HyperOptimizer: seeded %d strategies", len(self._population))

    # ------------------------------------------------------------------ #
    #  Selection                                                          #
    # ------------------------------------------------------------------ #

    def select_strategy(
        self, task_type: str = "general"
    ) -> OptimizerStrategy:
        """
        Return the best strategy for *task_type* based on fitness history.

        Falls back to round-robin if no fitness data exists yet.
        """
        candidates = [
            s for s in self._population
            if not s.task_types or TaskType(task_type) in s.task_types
        ] or self._population

        # Rank by task-specific fitness, break ties by mean fitness
        ranked = sorted(
            candidates,
            key=lambda s: (
                s.fitness_by_task.get(task_type, [0])[-1] if s.fitness_by_task.get(task_type) else 0,
                s.mean_fitness,
            ),
            reverse=True,
        )

        # ε-greedy exploration: 20 % chance to try a non-top strategy
        if self._rng.random() < 0.2 and len(ranked) > 1:
            chosen = self._rng.choice(ranked[1:])
            logger.debug("HyperOptimizer: ε-greedy → %s", chosen.strategy_id)
        else:
            chosen = ranked[0]

        logger.info(
            "HyperOptimizer: selected strategy=%s algorithm=%s for task=%s",
            chosen.strategy_id, chosen.algorithm.value, task_type,
        )
        return chosen

    def to_gepa_config(self, strategy: OptimizerStrategy) -> dict[str, Any]:
        """
        Convert *strategy* to the kwargs expected by :class:`~evolution.gepa_optimizer.GEPAOptimizer`.

        This is how HyperOptimizer controls GEPA's behaviour.
        """
        return {
            "max_generations": strategy.max_generations,
            "population_size": strategy.population_size,
            "mutation_temperature": strategy.mutation_temperature,
            "pareto_selection": strategy.pareto_selection,
            "max_metric_calls": strategy.max_metric_calls,
            "phase1_skills": "skills" in strategy.gepa_phase_mask,
            "phase2_tools": "tools" in strategy.gepa_phase_mask,
            "phase3_prompts": "prompts" in strategy.gepa_phase_mask,
            "phase4_code": "code" in strategy.gepa_phase_mask,
        }

    # ------------------------------------------------------------------ #
    #  Fitness recording                                                  #
    # ------------------------------------------------------------------ #

    def record_fitness(
        self,
        strategy: OptimizerStrategy,
        improvement: float,
        task_type: str = "general",
        cost_tokens: int = 0,
    ) -> None:
        """Record observed fitness for *strategy* after a run."""
        # Normalise by cost (Pareto trade-off: improvement / sqrt(cost))
        cost_normalised = improvement / (1.0 + (cost_tokens / 100_000) ** 0.5)
        strategy.record_fitness(cost_normalised, task_type)

        self._run_counter += 1
        self._run_history.append({
            "run": self._run_counter,
            "strategy_id": strategy.strategy_id,
            "algorithm": strategy.algorithm.value,
            "task_type": task_type,
            "improvement": improvement,
            "cost_tokens": cost_tokens,
            "fitness": cost_normalised,
            "timestamp": time.time(),
        })

        logger.info(
            "HyperOptimizer: strategy=%s task=%s improvement=%.4f fitness=%.4f",
            strategy.strategy_id, task_type, improvement, cost_normalised,
        )

        # Auto-evolve periodically
        if self._run_counter % self._evolve_every == 0:
            self.evolve()
        self._save()

    # ------------------------------------------------------------------ #
    #  Evolution                                                          #
    # ------------------------------------------------------------------ #

    def evolve(self) -> None:
        """
        Run one generation of meta-optimization:

        1. Rank population by mean fitness.
        2. Keep top ``elitism_ratio`` unchanged (elites).
        3. Fill the rest with mutants and crossover offspring.
        4. Optionally ask the LLM to propose a novel hybrid strategy.
        """
        if len(self._population) < 2:
            return

        self._generation += 1
        ranked = sorted(self._population, key=lambda s: s.mean_fitness, reverse=True)

        n_elite = max(1, int(len(ranked) * 0.2))
        elites = ranked[:n_elite]
        new_pop: list[OptimizerStrategy] = list(elites)

        # Mutate non-elites
        for s in ranked[n_elite:]:
            mutant = self._mutate(s)
            new_pop.append(mutant)

        # Crossover best pair
        if len(elites) >= 2:
            child = self._crossover(elites[0], elites[1])
            new_pop.append(child)

        # LLM-proposed hybrid (if LLM available and budget allows)
        if self._llm is not None and self._generation % 3 == 0:
            novel = self._llm_propose_strategy(elites)
            if novel:
                new_pop.append(novel)

        # Trim to population size
        new_pop = sorted(new_pop, key=lambda s: s.mean_fitness, reverse=True)
        self._population = new_pop[:self._pop_size]

        logger.info(
            "HyperOptimizer: evolved to gen=%d pop=%d best=%s(%.4f)",
            self._generation,
            len(self._population),
            self._population[0].strategy_id,
            self._population[0].mean_fitness,
        )
        self._save()

    # ------------------------------------------------------------------ #
    #  Mutation operators                                                 #
    # ------------------------------------------------------------------ #

    def _mutate(self, strategy: OptimizerStrategy) -> OptimizerStrategy:
        """Apply a random mutation operator to *strategy* and return the child."""
        child = deepcopy(strategy)
        child.strategy_id = str(uuid.uuid4())[:8]
        child.generation = self._generation
        child.parent_ids = [strategy.strategy_id]
        child.fitness_history = []
        child.fitness_by_task = {}

        op = self._rng.choice(strategy.mutation_operators or ["parameter_nudge"])

        if op == "parameter_nudge":
            child = self._nudge_parameters(child)
        elif op == "operator_swap":
            child = self._swap_operator(child)
        elif op == "phase_toggle":
            child = self._toggle_phase(child)
        elif op == "algorithm_blend":
            other = self._rng.choice(self._population)
            child = self._blend_hyperparams(child, other)
        # llm_propose is handled separately in evolve()

        child.description = f"Mutated({op}) from {strategy.strategy_id}"
        return child

    def _nudge_parameters(self, s: OptimizerStrategy) -> OptimizerStrategy:
        """Slightly perturb numeric hyperparameters."""
        def nudge(v: float, lo: float, hi: float) -> float:
            delta = self._rng.gauss(0, 0.1) * (hi - lo)
            return max(lo, min(hi, v + delta))

        s.mutation_temperature = nudge(s.mutation_temperature, 0.1, 2.0)
        s.mcts_exploration = nudge(s.mcts_exploration, 0.5, 3.0)
        s.textgrad_lr = nudge(s.textgrad_lr, 0.01, 1.0)
        s.evoprompt_crossover_prob = nudge(s.evoprompt_crossover_prob, 0.1, 0.95)
        s.population_size = max(2, s.population_size + self._rng.randint(-2, 2))
        return s

    def _swap_operator(self, s: OptimizerStrategy) -> OptimizerStrategy:
        """Replace one mutation operator with a randomly chosen alternative."""
        all_ops = [
            "parameter_nudge", "operator_swap", "phase_toggle",
            "algorithm_blend", "llm_propose",
        ]
        if s.mutation_operators:
            idx = self._rng.randrange(len(s.mutation_operators))
            s.mutation_operators[idx] = self._rng.choice(all_ops)
        return s

    def _toggle_phase(self, s: OptimizerStrategy) -> OptimizerStrategy:
        """Enable or disable one GEPA phase in the strategy."""
        phases = ["skills", "tools", "prompts", "code"]
        phase = self._rng.choice(phases)
        mask = list(s.gepa_phase_mask)
        if phase in mask:
            mask.remove(phase)
        else:
            mask.append(phase)
        s.gepa_phase_mask = mask or ["prompts"]  # always keep at least one
        return s

    def _blend_hyperparams(
        self, s: OptimizerStrategy, other: OptimizerStrategy
    ) -> OptimizerStrategy:
        """Blend numeric hyperparameters between *s* and *other* (BLX-α crossover)."""
        α = self._rng.random()
        s.mutation_temperature = α * s.mutation_temperature + (1 - α) * other.mutation_temperature
        s.population_size = int(α * s.population_size + (1 - α) * other.population_size)
        s.mcts_exploration = α * s.mcts_exploration + (1 - α) * other.mcts_exploration
        return s

    # ------------------------------------------------------------------ #
    #  Crossover                                                          #
    # ------------------------------------------------------------------ #

    def _crossover(
        self, parent_a: OptimizerStrategy, parent_b: OptimizerStrategy
    ) -> OptimizerStrategy:
        """
        Produce a child strategy that combines *parent_a* and *parent_b*.

        Uniform crossover: each hyperparameter is drawn from one parent at random.
        """
        child = deepcopy(parent_a)
        child.strategy_id = str(uuid.uuid4())[:8]
        child.generation = self._generation
        child.parent_ids = [parent_a.strategy_id, parent_b.strategy_id]
        child.fitness_history = []
        child.fitness_by_task = {}

        # Swap algorithm 50 % of the time
        if self._rng.random() < 0.5:
            child.algorithm = parent_b.algorithm

        # Blend hyperparameters
        α = self._rng.random()
        child.mutation_temperature = (
            α * parent_a.mutation_temperature + (1 - α) * parent_b.mutation_temperature
        )
        child.population_size = int(
            α * parent_a.population_size + (1 - α) * parent_b.population_size
        )
        child.mcts_exploration = (
            α * parent_a.mcts_exploration + (1 - α) * parent_b.mcts_exploration
        )

        # Merge mutation operator sets (union, capped at 5)
        ops = list(set(parent_a.mutation_operators + parent_b.mutation_operators))
        child.mutation_operators = self._rng.sample(ops, min(5, len(ops)))

        child.description = (
            f"Crossover({parent_a.strategy_id}×{parent_b.strategy_id})"
        )
        return child

    # ------------------------------------------------------------------ #
    #  LLM-proposed novel strategy                                        #
    # ------------------------------------------------------------------ #

    def _llm_propose_strategy(
        self, elites: list[OptimizerStrategy]
    ) -> OptimizerStrategy | None:
        """Ask the LLM to propose a novel hybrid optimization strategy."""
        elite_summaries = [
            f"  - id={e.strategy_id} alg={e.algorithm.value} "
            f"fitness={e.mean_fitness:.4f} phases={e.gepa_phase_mask}"
            for e in elites
        ]
        prompt = (
            "You are a meta-optimization expert.  The following optimization strategies "
            "have been the best performers:\n\n"
            + "\n".join(elite_summaries)
            + "\n\nPropose a NEW hybrid optimization strategy that combines their strengths "
            "and addresses their weaknesses.  Return a JSON object with these fields:\n"
            "  algorithm: one of gepa/textgrad/aflow/mipro/evoprompt/hybrid\n"
            "  mutation_temperature: float 0.1–2.0\n"
            "  population_size: int 2–20\n"
            "  max_generations: int 5–50\n"
            "  gepa_phase_mask: list of skills/tools/prompts/code\n"
            "  mcts_exploration: float 0.5–3.0\n"
            "  description: one-sentence description\n"
            "Return ONLY valid JSON."
        )
        try:
            raw = self._llm(prompt)
            if "```json" in raw:
                raw = raw.split("```json")[1].split("```")[0]
            elif "```" in raw:
                raw = raw.split("```")[1].split("```")[0]
            data = json.loads(raw.strip())

            s = OptimizerStrategy(
                algorithm=OptimizerAlgorithm(data.get("algorithm", "hybrid")),
                description=data.get("description", "LLM-proposed hybrid"),
                mutation_temperature=float(data.get("mutation_temperature", 0.9)),
                population_size=int(data.get("population_size", 8)),
                max_generations=int(data.get("max_generations", 20)),
                gepa_phase_mask=data.get("gepa_phase_mask", ["skills", "prompts"]),
                mcts_exploration=float(data.get("mcts_exploration", 1.41)),
                generation=self._generation,
                parent_ids=[e.strategy_id for e in elites],
            )
            logger.info(
                "HyperOptimizer: LLM proposed strategy=%s (%s)",
                s.strategy_id, s.description,
            )
            return s
        except Exception as exc:
            logger.warning("HyperOptimizer: LLM proposal failed (%s)", exc)
            return None

    # ------------------------------------------------------------------ #
    #  Reporting                                                          #
    # ------------------------------------------------------------------ #

    def report(self) -> str:
        """Return a human-readable summary of the current population."""
        lines = [
            f"HyperOptimizer — generation {self._generation}  "
            f"runs {self._run_counter}  pop {len(self._population)}",
            "",
            f"{'ID':>10}  {'Algorithm':>12}  {'MeanFit':>8}  {'Runs':>5}  Description",
            "-" * 70,
        ]
        for s in sorted(self._population, key=lambda x: x.mean_fitness, reverse=True):
            lines.append(
                f"{s.strategy_id:>10}  {s.algorithm.value:>12}  "
                f"{s.mean_fitness:>8.4f}  {len(s.fitness_history):>5}  {s.description[:35]}"
            )
        return "\n".join(lines)

    def best_strategy(self, task_type: str = "general") -> OptimizerStrategy:
        """Return the highest-fitness strategy for *task_type* (no ε-greedy)."""
        candidates = [
            s for s in self._population
            if not s.task_types or TaskType(task_type) in s.task_types
        ] or self._population
        return max(candidates, key=lambda s: s.mean_fitness)

    # ------------------------------------------------------------------ #
    #  Persistence                                                        #
    # ------------------------------------------------------------------ #

    def _save(self) -> None:
        self._persist_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "generation": self._generation,
            "run_counter": self._run_counter,
            "population": [s.to_dict() for s in self._population],
            "run_history": self._run_history[-500:],  # keep last 500 run records
        }
        self._persist_path.write_text(json.dumps(data, indent=2))

    def load(self) -> None:
        """Reload state from disk (useful after external modifications)."""
        self._load_or_seed()
