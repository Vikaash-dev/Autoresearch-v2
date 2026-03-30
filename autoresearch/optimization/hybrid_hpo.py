"""
Hybrid HPO — Centaur: LLM + CMA-ES (2026).

Paper: "Can LLMs Beat Classical Hyperparameter Optimization?" (arXiv, Mar 2026)
Result: Centaur outperforms pure LLM-based HyperAgents by sharing internal CMA-ES
state (covariance matrix, step-size σ) with the LLM, letting it combine domain
knowledge with mathematical precision.

Architecture:
  - CMA-ES handles the mathematics (covariance, step-size adaptation)
  - LLM provides domain knowledge (priors, constraints, semantic understanding)
  - Shared state: LLM READS the covariance matrix to understand which parameters
    are correlated, then PROPOSES changes that respect those correlations.

CMA-ES core:
  m_{t+1} = m_t + σ_t * Σ_{i=1}^{λ} w_i * (x_{i:λ} - m_t) / σ_t
  σ_{t+1} = σ_t * exp((‖p_σ‖ / E‖N(0,I)‖ - 1) * c_σ / d_σ)
  C_{t+1} = (1 - c_1 - c_μ) * C_t + c_1 * p_c * p_c^T + c_μ * Σ w_i * (...)
"""

from __future__ import annotations

import logging
import math
import random
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CMA-ES core (pure Python + numpy — no scipy dependency)
# ---------------------------------------------------------------------------

class CMAES:
    """
    Simple (μ/μ_w, λ)-CMA-ES implementation.
    Tracks: mean vector m, step-size σ, covariance matrix C,
    evolution paths p_σ and p_c.

    This state is SHARED with the LLM (Centaur's key contribution).
    """

    def __init__(
        self,
        n_dims: int,
        sigma0: float = 0.3,
        popsize: Optional[int] = None,
    ) -> None:
        self.n = n_dims
        self.sigma = sigma0

        # Population size
        self.lam = popsize or (4 + int(3 * math.log(n_dims)))
        self.mu  = self.lam // 2

        # Weights
        raw_w = [math.log((self.lam + 1) / 2) - math.log(i + 1) for i in range(self.mu)]
        w_sum = sum(raw_w)
        self.weights = [w / w_sum for w in raw_w]
        self.mu_eff  = 1.0 / sum(w ** 2 for w in self.weights)

        # Adaptation constants
        self.cc   = (4 + self.mu_eff / n_dims) / (n_dims + 4 + 2 * self.mu_eff / n_dims)
        self.cs   = (self.mu_eff + 2) / (n_dims + self.mu_eff + 5)
        self.c1   = 2 / ((n_dims + 1.3) ** 2 + self.mu_eff)
        self.cmu  = min(
            1 - self.c1,
            2 * (self.mu_eff - 2 + 1 / self.mu_eff) / ((n_dims + 2) ** 2 + self.mu_eff),
        )
        self.damps = 1 + 2 * max(0, math.sqrt((self.mu_eff - 1) / (n_dims + 1)) - 1) + self.cs
        self.chiN  = n_dims ** 0.5 * (1 - 1 / (4 * n_dims) + 1 / (21 * n_dims ** 2))

        # State
        self.mean = np.zeros(n_dims)
        self.C    = np.eye(n_dims)
        self.pc   = np.zeros(n_dims)
        self.ps   = np.zeros(n_dims)
        self.eigenvalues  = np.ones(n_dims)
        self.eigenvectors = np.eye(n_dims)
        self._generation  = 0
        self._decomp_stale = True

    # ------------------------------------------------------------------

    def ask(self) -> np.ndarray:
        """Sample one candidate from N(m, σ²C)."""
        self._maybe_decompose()
        z = np.random.randn(self.n)
        return self.mean + self.sigma * self.eigenvectors @ (self.eigenvalues ** 0.5 * z)

    def tell(self, candidates: List[np.ndarray], fitness: List[float]) -> None:
        """Update distribution from evaluated candidates (lower fitness = better)."""
        # Sort by fitness (ascending = better)
        ranked = [x for _, x in sorted(zip(fitness, candidates), key=lambda t: t[0])]
        selected = ranked[: self.mu]

        old_mean = self.mean.copy()
        self.mean = sum(w * x for w, x in zip(self.weights, selected))

        # Evolution paths
        y_w = (self.mean - old_mean) / self.sigma
        self.ps = (1 - self.cs) * self.ps + math.sqrt(self.cs * (2 - self.cs) * self.mu_eff) * (
            np.linalg.solve(self.eigenvectors, y_w / (self.eigenvalues ** 0.5))
        )
        hsig = (
            np.linalg.norm(self.ps) / math.sqrt(1 - (1 - self.cs) ** (2 * (self._generation + 1))) / self.chiN
            < 1.4 + 2 / (self.n + 1)
        )
        self.pc = (1 - self.cc) * self.pc + hsig * math.sqrt(self.cc * (2 - self.cc) * self.mu_eff) * y_w

        # Covariance matrix update
        rank_one = np.outer(self.pc, self.pc)
        rank_mu  = sum(
            w * np.outer((x - old_mean) / self.sigma, (x - old_mean) / self.sigma)
            for w, x in zip(self.weights, selected)
        )
        self.C = (1 - self.c1 - self.cmu) * self.C + self.c1 * rank_one + self.cmu * rank_mu

        # Step-size update
        self.sigma *= math.exp((self.cs / self.damps) * (np.linalg.norm(self.ps) / self.chiN - 1))
        self.sigma = max(1e-8, min(self.sigma, 10.0))  # clamp

        self._generation += 1
        self._decomp_stale = True

    def _maybe_decompose(self) -> None:
        if not self._decomp_stale:
            return
        try:
            vals, vecs = np.linalg.eigh(self.C)
            self.eigenvalues  = np.maximum(vals, 1e-20)
            self.eigenvectors = vecs
        except np.linalg.LinAlgError:
            self.eigenvalues  = np.ones(self.n)
            self.eigenvectors = np.eye(self.n)
        self._decomp_stale = False

    # ------------------------------------------------------------------
    # State serialisation — shared with LLM (Centaur's key insight)
    # ------------------------------------------------------------------

    def state_for_llm(self, param_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Return CMA-ES internal state in human-readable form for the LLM.
        The LLM uses this to understand parameter correlations and step-sizes.
        """
        names = param_names or [f"p{i}" for i in range(self.n)]
        self._maybe_decompose()

        # Top-3 correlated pairs
        corr_pairs = []
        if self.n > 1:
            C_norm = self.C.copy()
            diag    = np.sqrt(np.diag(C_norm))
            outer_d = np.outer(diag, diag)
            with np.errstate(divide="ignore", invalid="ignore"):
                corr = np.where(outer_d > 0, C_norm / outer_d, 0)
            np.fill_diagonal(corr, 0)
            flat_idx = np.argpartition(np.abs(corr).ravel(), -min(6, corr.size))[-min(6, corr.size):]
            for idx in flat_idx:
                i, j = divmod(idx, self.n)
                if i < j:
                    corr_pairs.append({
                        "params": [names[i], names[j]],
                        "correlation": round(float(corr[i, j]), 3),
                    })

        return {
            "generation":       self._generation,
            "sigma":            round(float(self.sigma), 6),
            "mean":             {names[i]: round(float(self.mean[i]), 6) for i in range(self.n)},
            "top_eigenvalues":  [round(float(v), 6) for v in sorted(self.eigenvalues, reverse=True)[:5]],
            "correlated_pairs": sorted(corr_pairs, key=lambda x: abs(x["correlation"]), reverse=True)[:3],
            "effective_dims":   int(sum(1 for v in self.eigenvalues if v > 0.01 * max(self.eigenvalues))),
        }


# ---------------------------------------------------------------------------
# Centaur: hybrid LLM + CMA-ES proposer
# ---------------------------------------------------------------------------

@dataclass
class HybridHPOConfig:
    n_dims:           int   = 8
    sigma0:           float = 0.3
    max_iterations:   int   = 50
    budget_seconds:   float = 600.0
    llm_weight:       float = 0.5   # how much LLM shifts the CMA-ES mean (0=pure CMA-ES, 1=pure LLM)
    param_names:      List[str] = field(default_factory=list)
    param_bounds:     Dict[str, Tuple[float, float]] = field(default_factory=dict)


class CentaurHPO:
    """
    Centaur-style Hybrid HPO: LLM + CMA-ES.

    The CMA-ES provides mathematical precision (covariance matrix tracks
    parameter correlations). The LLM provides domain knowledge (priors,
    semantic constraints). They share state each iteration.

    Usage:
        def my_evaluator(params: dict) -> float: ...
        def my_llm_proposer(cma_state: dict, history: list) -> dict: ...

        centaur = CentaurHPO(config, my_evaluator, my_llm_proposer)
        result  = centaur.run("optimize val_bpb", baseline=1.10)
    """

    def __init__(
        self,
        config:    HybridHPOConfig,
        evaluator: Callable[[Dict[str, Any]], float],
        llm_proposer: Optional[Callable[[Dict[str, Any], List[Dict]], Dict[str, Any]]] = None,
    ) -> None:
        self.cfg = config
        self.evaluator = evaluator
        self.llm_proposer = llm_proposer or self._default_llm_proposer

        n = max(config.n_dims, len(config.param_names) or 1)
        self.cma = CMAES(n_dims=n, sigma0=config.sigma0)
        if config.param_names:
            self.cma.mean = np.zeros(len(config.param_names))

        self._history: List[Dict[str, Any]] = []
        self._best_score   = float("-inf")
        self._best_params: Dict[str, Any] = {}
        self._start_time: float = 0.0

    def run(self, task: str, baseline: float = 0.0) -> Dict[str, Any]:
        self._start_time = time.time()
        self._best_score = baseline

        for i in range(self.cfg.max_iterations):
            if self._over_budget():
                break

            # 1. CMA-ES sample candidate vector
            cma_vector = self.cma.ask()

            # 2. Share CMA state with LLM — Centaur's key contribution
            cma_state = self.cma.state_for_llm(self.cfg.param_names or None)

            # 3. LLM proposes adjustments using domain knowledge + CMA state
            llm_proposal = self.llm_proposer(cma_state, self._history[-5:])

            # 4. Blend: CMA-ES vector + LLM proposal (weighted combination)
            params = self._blend(cma_vector, llm_proposal)

            # 5. Evaluate
            try:
                score = self.evaluator(params)
                status = "ok"
            except Exception as exc:  # noqa: BLE001
                logger.warning("Centaur evaluator error at %d: %s", i, exc)
                score, status = baseline, "error"

            improvement = score - self._best_score

            # 6. Tell CMA-ES the result (in CMA-ES convention: minimise → negate)
            fitness_val = -score  # CMA-ES minimises
            self.cma.tell([cma_vector], [fitness_val])

            if improvement > 0:
                self._best_score  = score
                self._best_params = dict(params)

            record = {
                "iteration":   i,
                "score":       score,
                "improvement": improvement,
                "params":      params,
                "cma_sigma":   round(self.cma.sigma, 6),
                "status":      status,
            }
            self._history.append(record)

            logger.debug(
                "Centaur [%d] score=%.4f sigma=%.4f improvement=%+.4f",
                i, score, self.cma.sigma, improvement,
            )

        elapsed = time.time() - self._start_time
        return {
            "task":            task,
            "baseline":        baseline,
            "best_score":      self._best_score,
            "best_params":     self._best_params,
            "total_improvement": self._best_score - baseline,
            "iterations":      len(self._history),
            "elapsed_seconds": elapsed,
            "final_sigma":     round(self.cma.sigma, 6),
            "final_cma_state": self.cma.state_for_llm(self.cfg.param_names or None),
            "history":         self._history,
        }

    # ------------------------------------------------------------------

    def _blend(self, cma_vector: np.ndarray, llm_proposal: Dict[str, Any]) -> Dict[str, Any]:
        """Blend CMA-ES vector with LLM proposal into a parameter dict."""
        names = self.cfg.param_names
        bounds = self.cfg.param_bounds
        result: Dict[str, Any] = {}

        for i, name in enumerate(names):
            cma_val = float(cma_vector[i]) if i < len(cma_vector) else 0.0
            llm_val = float(llm_proposal.get(name, cma_val))
            blended = (1 - self.cfg.llm_weight) * cma_val + self.cfg.llm_weight * llm_val

            # Apply bounds
            if name in bounds:
                lo, hi = bounds[name]
                blended = max(lo, min(hi, blended))

            result[name] = blended

        # Add any extra LLM proposals not in param_names (categorical, etc.)
        for k, v in llm_proposal.items():
            if k not in result:
                result[k] = v

        return result

    def _over_budget(self) -> bool:
        return (time.time() - self._start_time) >= self.cfg.budget_seconds

    @staticmethod
    def _default_llm_proposer(cma_state: Dict[str, Any], history: List[Dict]) -> Dict[str, Any]:
        """Fallback LLM proposer — uses CMA mean as proposal (pure CMA-ES mode)."""
        return {k: v for k, v in cma_state.get("mean", {}).items()}
