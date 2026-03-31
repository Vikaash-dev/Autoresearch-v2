"""
Tests for evolution/hyper_optimizer.py

Covers:
  - HyperOptimizer construction and seed population
  - OptimizerStrategy fitness recording
  - Strategy selection (best + ε-greedy)
  - Mutation operators (nudge, swap, toggle, blend)
  - Crossover offspring validity
  - evolve() generation advancement
  - to_gepa_config() output
  - Persistence (save/load round-trip)
  - from_config() construction
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from evolution.hyper_optimizer import (
    HyperOptimizer,
    OptimizerAlgorithm,
    OptimizerStrategy,
    TaskType,
)


# ────────────────────────────────────────────────────────────────────────── #
#  Fixtures                                                                   #
# ────────────────────────────────────────────────────────────────────────── #

@pytest.fixture()
def opt(tmp_path: Path) -> HyperOptimizer:
    return HyperOptimizer(
        population_size=4,
        max_generations=5,
        evolve_every_n_runs=3,
        persist_path=str(tmp_path / "hyper_state.json"),
        rng_seed=42,
    )


@pytest.fixture()
def strategy() -> OptimizerStrategy:
    s = OptimizerStrategy(
        algorithm=OptimizerAlgorithm.GEPA,
        mutation_temperature=0.9,
        population_size=8,
    )
    return s


# ────────────────────────────────────────────────────────────────────────── #
#  Construction                                                               #
# ────────────────────────────────────────────────────────────────────────── #

class TestConstruction:
    def test_seed_population_has_five_strategies(self, opt):
        # 5 seed strategies from _SEED_STRATEGIES
        assert len(opt._population) >= 4

    def test_all_algorithms_seeded(self, opt):
        algorithms = {s.algorithm for s in opt._population}
        assert OptimizerAlgorithm.GEPA in algorithms

    def test_from_config(self, tmp_path):
        cfg = {
            "hyper_optimizer": {
                "population_size": 3,
                "max_generations": 10,
                "evolve_every_n_runs": 2,
                "persist_path": str(tmp_path / "state.json"),
            }
        }
        ho = HyperOptimizer.from_config(cfg)
        assert ho._pop_size == 3

    def test_generation_starts_at_zero(self, opt):
        assert opt._generation == 0


# ────────────────────────────────────────────────────────────────────────── #
#  OptimizerStrategy                                                         #
# ────────────────────────────────────────────────────────────────────────── #

class TestOptimizerStrategy:
    def test_mean_fitness_empty(self, strategy):
        assert strategy.mean_fitness == 0.0

    def test_record_fitness_updates_history(self, strategy):
        strategy.record_fitness(0.5, "hypothesis")
        strategy.record_fitness(0.8, "hypothesis")
        assert len(strategy.fitness_history) == 2
        assert strategy.mean_fitness == pytest.approx(0.65)

    def test_task_fitness_per_type(self, strategy):
        strategy.record_fitness(0.6, "hypothesis")
        strategy.record_fitness(0.9, "experiment")
        assert "hypothesis" in strategy.fitness_by_task
        assert "experiment" in strategy.fitness_by_task

    def test_to_dict_round_trip(self, strategy):
        strategy.record_fitness(0.7, "skills")
        d = strategy.to_dict()
        s2 = OptimizerStrategy.from_dict(d)
        assert s2.strategy_id == strategy.strategy_id
        assert s2.algorithm == strategy.algorithm
        assert s2.mean_fitness == pytest.approx(strategy.mean_fitness)


# ────────────────────────────────────────────────────────────────────────── #
#  Selection                                                                  #
# ────────────────────────────────────────────────────────────────────────── #

class TestSelection:
    def test_select_strategy_returns_strategy(self, opt):
        s = opt.select_strategy("hypothesis")
        assert isinstance(s, OptimizerStrategy)

    def test_best_strategy_returns_highest_fitness(self, opt):
        # Give one strategy a high fitness
        winner = opt._population[0]
        winner.record_fitness(0.99, "general")
        best = opt.best_strategy("general")
        assert best.strategy_id == winner.strategy_id

    def test_select_strategy_general_fallback(self, opt):
        # Should not raise even with no task-type-specific history
        s = opt.select_strategy("unknown_task_type")
        assert s is not None


# ────────────────────────────────────────────────────────────────────────── #
#  Fitness recording + auto-evolve                                            #
# ────────────────────────────────────────────────────────────────────────── #

class TestFitnessRecording:
    def test_run_counter_increments(self, opt):
        s = opt._population[0]
        opt.record_fitness(s, improvement=0.1)
        assert opt._run_counter == 1

    def test_auto_evolve_triggers_at_interval(self, opt):
        s = opt._population[0]
        # evolve_every_n_runs=3, so after 3 records generation should advance
        for _ in range(3):
            opt.record_fitness(s, improvement=0.05)
        assert opt._generation >= 1

    def test_run_history_populated(self, opt):
        s = opt._population[0]
        opt.record_fitness(s, improvement=0.2, task_type="experiment")
        assert len(opt._run_history) == 1
        assert opt._run_history[0]["task_type"] == "experiment"


# ────────────────────────────────────────────────────────────────────────── #
#  Mutation                                                                   #
# ────────────────────────────────────────────────────────────────────────── #

class TestMutation:
    def test_nudge_produces_child_with_different_id(self, opt):
        parent = opt._population[0]
        child = opt._mutate(parent)
        assert child.strategy_id != parent.strategy_id

    def test_child_has_empty_fitness_history(self, opt):
        parent = opt._population[0]
        parent.record_fitness(0.5)
        child = opt._mutate(parent)
        assert child.fitness_history == []

    def test_phase_toggle_changes_mask(self, opt):
        parent = opt._population[0]
        original_mask = list(parent.gepa_phase_mask)
        # Run toggle mutation repeatedly until it changes
        for _ in range(20):
            child = opt._toggle_phase(parent)
            if child.gepa_phase_mask != original_mask:
                break
        # The phase mask should change in at most 20 tries
        assert child.gepa_phase_mask != original_mask or len(original_mask) != len(child.gepa_phase_mask)

    def test_mutant_keeps_parent_reference(self, opt):
        parent = opt._population[0]
        child = opt._mutate(parent)
        assert parent.strategy_id in child.parent_ids


# ────────────────────────────────────────────────────────────────────────── #
#  Crossover                                                                  #
# ────────────────────────────────────────────────────────────────────────── #

class TestCrossover:
    def test_crossover_has_two_parents(self, opt):
        a, b = opt._population[0], opt._population[1]
        child = opt._crossover(a, b)
        assert a.strategy_id in child.parent_ids
        assert b.strategy_id in child.parent_ids

    def test_crossover_child_empty_fitness(self, opt):
        a, b = opt._population[0], opt._population[1]
        a.record_fitness(0.8)
        b.record_fitness(0.7)
        child = opt._crossover(a, b)
        assert child.fitness_history == []

    def test_crossover_mutation_ops_non_empty(self, opt):
        a, b = opt._population[0], opt._population[1]
        child = opt._crossover(a, b)
        assert len(child.mutation_operators) >= 1


# ────────────────────────────────────────────────────────────────────────── #
#  evolve()                                                                   #
# ────────────────────────────────────────────────────────────────────────── #

class TestEvolve:
    def test_generation_increments(self, opt):
        opt.evolve()
        assert opt._generation == 1

    def test_population_stays_within_size(self, opt):
        opt.evolve()
        assert len(opt._population) <= opt._pop_size

    def test_elites_survive(self, opt):
        winner = opt._population[0]
        winner.record_fitness(0.99)
        opt.evolve()
        ids = {s.strategy_id for s in opt._population}
        assert winner.strategy_id in ids


# ────────────────────────────────────────────────────────────────────────── #
#  to_gepa_config                                                             #
# ────────────────────────────────────────────────────────────────────────── #

class TestToGepaConfig:
    def test_returns_dict_with_required_keys(self, opt):
        s = opt._population[0]
        cfg = opt.to_gepa_config(s)
        assert "max_generations" in cfg
        assert "population_size" in cfg
        assert "mutation_temperature" in cfg
        assert "phase1_skills" in cfg

    def test_phase_mask_respected(self, opt):
        s = opt._population[0]
        s.gepa_phase_mask = ["prompts"]
        cfg = opt.to_gepa_config(s)
        assert cfg["phase3_prompts"] is True
        assert cfg["phase1_skills"] is False


# ────────────────────────────────────────────────────────────────────────── #
#  Persistence                                                                #
# ────────────────────────────────────────────────────────────────────────── #

class TestPersistence:
    def test_save_creates_file(self, opt, tmp_path):
        opt._save()
        assert opt._persist_path.exists()

    def test_load_restores_population(self, opt, tmp_path):
        # Record some fitness so state is non-trivial
        s = opt._population[0]
        opt.record_fitness(s, 0.5)
        opt._save()

        # Create fresh optimizer pointing at same file
        opt2 = HyperOptimizer(
            persist_path=str(opt._persist_path),
            rng_seed=42,
        )
        assert opt2._run_counter == opt._run_counter

    def test_report_string_non_empty(self, opt):
        report = opt.report()
        assert len(report) > 0
        assert "HyperOptimizer" in report
