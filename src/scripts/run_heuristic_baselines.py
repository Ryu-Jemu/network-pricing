"""Evaluate hand-tuned heuristic policies (B-add-3, B-add-4) on the paper env.

Both baselines run on `extended=False` env (no cohort, no asymmetric γ —
i.e. the original published env), churn-multiplier swept m ∈ {1, 3, 5, 10}.

Output: `results/heuristic_baselines.json`
"""
from __future__ import annotations

import json
import os
import time

from src.train.config import get_env_config
from src.train.heuristic_baselines import (
    evaluate_policy_factory,
    make_load_threshold_policy,
    make_peak_offpeak_policy,
)

PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RESULTS_DIR = os.path.join(PROJ_ROOT, "results")

MULTIPLIERS = [1, 3, 5, 10]
N_EVAL = 20
EVAL_BASE_SEED = 1000


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    results = {
        "config": {
            "extended_env": False,
            "n_eval_episodes": N_EVAL,
            "eval_base_seed": EVAL_BASE_SEED,
            "multipliers": MULTIPLIERS,
            "policies": ["load_threshold (B-add-3)", "peak_offpeak (B-add-4)"],
            "note": (
                "Hand-tuned heuristic baselines on the published paper env. "
                "Compared against PPO (paper) and Max-Price values in "
                "results/churn_sweep_results.json under the same m sweep."
            ),
        },
        "multipliers": {},
    }

    for m in MULTIPLIERS:
        cfg = get_env_config(extended=False, churn_multiplier=m)
        print(f"\n[m={m}] gamma0={cfg['gamma0']}")

        t0 = time.time()
        lt = evaluate_policy_factory(
            cfg,
            policy_factory=lambda: make_load_threshold_policy(),
            n_episodes=N_EVAL,
            base_seed=EVAL_BASE_SEED,
        )
        print(f"  [Load-threshold] reward={lt['mean_reward']:.1f} "
              f"N_E={lt['mean_final_N_E']:.0f} ({time.time()-t0:.1f}s)")

        t0 = time.time()
        po = evaluate_policy_factory(
            cfg,
            policy_factory=lambda: make_peak_offpeak_policy(),
            n_episodes=N_EVAL,
            base_seed=EVAL_BASE_SEED,
        )
        print(f"  [Peak/off-peak] reward={po['mean_reward']:.1f} "
              f"N_E={po['mean_final_N_E']:.0f} ({time.time()-t0:.1f}s)")

        results["multipliers"][str(m)] = {
            "multiplier": int(m),
            "gamma0_adjusted": cfg["gamma0"],
            "load_threshold": lt,
            "peak_offpeak": po,
        }

    out = os.path.join(RESULTS_DIR, "heuristic_baselines.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[saved] {out}")


if __name__ == "__main__":
    main()
