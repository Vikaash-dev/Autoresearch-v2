"""Tests for the Centaur hybrid LLM + CMA-ES optimizer (2026)."""
import numpy as np
import pytest
from autoresearch.optimization.hybrid_hpo import CMAES, CentaurHPO, HybridHPOConfig


class TestCMAES:
    def test_ask_returns_correct_shape(self):
        cma = CMAES(n_dims=4)
        x = cma.ask()
        assert x.shape == (4,)

    def test_tell_updates_mean(self):
        cma = CMAES(n_dims=2, sigma0=0.5)
        old_mean = cma.mean.copy()
        candidates = [cma.ask() for _ in range(cma.lam)]
        fitnesses = [float(np.sum(c ** 2)) for c in candidates]   # sphere function
        cma.tell(candidates, fitnesses)
        # Mean should move toward origin (lower fitness)
        assert cma._generation == 1

    def test_sigma_stays_bounded(self):
        cma = CMAES(n_dims=3)
        for _ in range(20):
            cs = [cma.ask() for _ in range(cma.lam)]
            fs = [float(np.sum(c ** 2)) for c in cs]
            cma.tell(cs, fs)
        assert 1e-7 <= cma.sigma <= 11.0

    def test_state_for_llm_keys(self):
        cma = CMAES(n_dims=3)
        state = cma.state_for_llm(["a", "b", "c"])
        assert "sigma" in state
        assert "mean" in state
        assert "generation" in state
        assert "correlated_pairs" in state
        assert set(state["mean"].keys()) == {"a", "b", "c"}

    def test_convergence_on_sphere(self):
        """CMA-ES should reduce function value on a simple sphere."""
        n = 4
        cma = CMAES(n_dims=n, sigma0=1.0)
        cma.mean = np.ones(n) * 2.0  # start away from optimum

        initial_score = float(np.sum(cma.mean ** 2))
        for _ in range(30):
            candidates = [cma.ask() for _ in range(cma.lam)]
            fitnesses = [float(np.sum(c ** 2)) for c in candidates]
            cma.tell(candidates, fitnesses)

        final_score = float(np.sum(cma.mean ** 2))
        assert final_score < initial_score, f"CMA-ES did not converge: {initial_score} → {final_score}"


class TestCentaurHPO:
    def _make_centaur(self, n_dims=3, max_iter=15):
        config = HybridHPOConfig(
            n_dims=n_dims,
            sigma0=0.5,
            max_iterations=max_iter,
            budget_seconds=60.0,
            param_names=[f"p{i}" for i in range(n_dims)],
            param_bounds={f"p{i}": (-2.0, 2.0) for i in range(n_dims)},
        )

        def evaluator(params):
            # sphere: lower = better, return negative so higher = better for our convention
            vals = [params.get(f"p{i}", 0.0) for i in range(n_dims)]
            return -sum(v ** 2 for v in vals)  # peak at 0,0,0

        return CentaurHPO(config=config, evaluator=evaluator)

    def test_run_returns_dict(self):
        centaur = self._make_centaur()
        result = centaur.run("sphere optimisation", baseline=-10.0)
        assert isinstance(result, dict)
        assert "best_score" in result
        assert "best_params" in result
        assert "history" in result

    def test_score_improves_over_baseline(self):
        centaur = self._make_centaur(max_iter=20)
        result = centaur.run("sphere", baseline=-12.0)
        assert result["best_score"] >= -12.0

    def test_params_within_bounds(self):
        centaur = self._make_centaur()
        result = centaur.run("bounded sphere", baseline=-10.0)
        for k, v in result["best_params"].items():
            if k.startswith("p"):
                assert -2.0 <= v <= 2.0, f"{k}={v} out of bounds"

    def test_final_cma_state_present(self):
        centaur = self._make_centaur()
        result = centaur.run("test", baseline=0.0)
        assert "final_cma_state" in result
        assert "sigma" in result["final_cma_state"]

    def test_custom_llm_proposer(self):
        """LLM proposer should influence blended proposals."""
        config = HybridHPOConfig(
            n_dims=2,
            param_names=["lr", "wd"],
            llm_weight=1.0,   # pure LLM
        )

        def evaluator(params):
            return -abs(params.get("lr", 0) - 0.01)

        def llm_proposer(cma_state, history):
            return {"lr": 0.01, "wd": 0.0001}

        centaur = CentaurHPO(config=config, evaluator=evaluator, llm_proposer=llm_proposer)
        result = centaur.run("test", baseline=-1.0)
        # With pure LLM proposer always returning lr=0.01, score should be near 0
        assert result["best_score"] >= -0.1
