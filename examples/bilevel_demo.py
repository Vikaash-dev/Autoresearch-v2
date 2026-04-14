#!/usr/bin/env python
"""
Bilevel Autoresearch demo — shows the 3-level self-improving loop.

Level 1  (inner): propose → execute → evaluate → keep/discard
Level 1.5 (outer-cfg): analyze trace → adjust search parameters
Level 2  (outer-mech): plateau detection → generate new Python mechanisms → importlib

python examples/bilevel_demo.py
"""
import logging
import random
from autoresearch.optimization.bilevel import BilevelEngine, BilevelConfig, SearchStrategy

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(name)s: %(message)s")


# ---  Simulated ML training function (fixed metric, Karpathy-style) ----------

OPTIMAL_LR = 0.003
OPTIMAL_BS = 64


def simulate_training(changes: dict) -> tuple[float, str]:
    """
    Simulate val_bpb metric for a language model.
    Lower val_bpb = better.  Return inverted (higher = better) for the engine.
    """
    lr = changes.get("learning_rate", 0.001)
    bs = changes.get("batch_size", 32)
    wd = changes.get("weight_decay", 0.0001)

    # Add noise to simulate real training variance
    noise = random.gauss(0, 0.02)

    # Simple quadratic landscape around optimal
    lr_penalty = ((lr - OPTIMAL_LR) / OPTIMAL_LR) ** 2
    bs_penalty = ((bs - OPTIMAL_BS) / OPTIMAL_BS) ** 2
    wd_penalty = (wd * 100) ** 2

    val_bpb = 1.10 - 0.30 * (1 - lr_penalty) - 0.10 * (1 - bs_penalty) - 0.05 * (1 - wd_penalty) + noise
    val_bpb = max(0.7, min(1.5, val_bpb))

    # Occasional crash (42% failure rate as in ACM evaluation)
    if random.random() < 0.05:
        return 0.0, "crashed"

    # Return negative val_bpb so higher = better
    return round(-val_bpb + 1.5, 4), "completed"  # scale to [0, 0.8]


# ---  LLM-like proposer (heuristic, no actual LLM required) ------------------

HP_SEARCH_SPACE = {
    "learning_rate": [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2],
    "batch_size":    [16, 32, 64, 128, 256],
    "weight_decay":  [0.0, 1e-5, 1e-4, 1e-3, 1e-2],
}


def proposer(history, strategy: SearchStrategy, current_config: dict) -> dict:
    """Simulate LLM hyperparameter proposals based on strategy."""
    if strategy == SearchStrategy.EXPLOIT and current_config:
        # Exploit: small perturbation around best config
        lr = current_config.get("learning_rate", OPTIMAL_LR) * random.uniform(0.7, 1.3)
        bs = current_config.get("batch_size", 32)
        wd = current_config.get("weight_decay", 0.0001) * random.uniform(0.5, 2.0)
    elif strategy == SearchStrategy.COMBINE and len(history) >= 2:
        # Combine: mix two recent configs
        a, b = random.sample(history[-6:], min(2, len(history[-6:])))
        lr = (a.changes.get("learning_rate", 0.001) + b.changes.get("learning_rate", 0.001)) / 2
        bs = random.choice([a.changes.get("batch_size", 32), b.changes.get("batch_size", 64)])
        wd = (a.changes.get("weight_decay", 0.0001) + b.changes.get("weight_decay", 0.0001)) / 2
    else:
        # Explore: random from search space
        lr = random.choice(HP_SEARCH_SPACE["learning_rate"])
        bs = random.choice(HP_SEARCH_SPACE["batch_size"])
        wd = random.choice(HP_SEARCH_SPACE["weight_decay"])

    return {
        "learning_rate": round(max(1e-5, lr), 6),
        "batch_size":    int(max(8, bs)),
        "weight_decay":  round(max(0.0, wd), 6),
    }


# --- Main ------------------------------------------------------------------

def main():
    print("=" * 60)
    print("Bilevel Autoresearch Demo (arXiv:2603.23420)")
    print("3-level self-improving hyperparameter search")
    print("=" * 60)
    print(f"Optimal: lr={OPTIMAL_LR}, bs={OPTIMAL_BS}")
    print("Baseline: random configuration")
    print()

    baseline_score, _ = simulate_training({"learning_rate": 0.1, "batch_size": 16, "weight_decay": 0.0})
    print(f"Baseline score (poor config): {baseline_score:.4f}")

    config = BilevelConfig(
        inner_max_iterations=30,
        inner_budget_seconds=300.0,
        config_adaptation_every=5,
        mechanism_generation_every=10,
        plateau_threshold=5,
        verbose=True,
    )

    def on_progress(prog):
        if prog["iteration"] % 5 == 0:
            print(f"  Iter {prog['iteration']:2d}: best={prog['best_score']:.4f}  "
                  f"strategy={prog['strategy']:8s}  mechanisms={prog['mechanisms']}")

    engine = BilevelEngine(
        executor=simulate_training,
        proposer=proposer,
        config=config,
        on_progress=on_progress,
    )

    result = engine.run("minimise val_bpb", baseline_score=baseline_score)

    print()
    print("=" * 60)
    print("Results:")
    print(f"  Baseline score:    {baseline_score:.4f}")
    print(f"  Best score:        {result['best_score']:.4f}")
    print(f"  Improvement:       {result['total_improvement']:+.4f}")
    print(f"  Experiments run:   {result['experiments_run']}")
    print(f"  Experiments kept:  {result['experiments_kept']}")
    print(f"  Keep rate:         {result['keep_rate']:.1%}")
    print(f"  Mechanisms used:   {result['mechanisms_used']}")
    print(f"  Tabu blocks:       {result['tabu_summary'].get('total_blocks', 0)}")
    print(f"  Best config:       {result['best_config']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
