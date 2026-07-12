"""7-seed paper-env churn sweep (Phase 9 P2-b implementation).

Re-trains PPO at m ∈ {1, 3, 5, 10} with **seven** seeds instead of three.
Seed set [42, 123, 456, 789, 1011, 1213, 1415] mirrors the improvement-3
(`seed-power`) branch convention.

This script is identical in structure to `run_churn_sweep.py` except for:
  - SEEDS list (7 entries)
  - MODELS_DIR and RESULTS path (separate output directories so the
    published 3-seed artifacts are not overwritten)
  - PPO trains on the **paper** env (`get_env_config(extended=False)`)
    — i.e. the same env as the published paper, just with more seeds.

Output:
  - models/seed_power_paper/ppo_mult{M}_seed{S}.zip   (28 files)
  - results/seed_power_paper.json

Runtime: ~7–14 h CPU (28 × 15–30 min/run). Recommended to launch in the
background and monitor `training_logs/run_churn_sweep_7seeds.log`.

Usage:
  PYTHONPATH=. python3 src/scripts/run_churn_sweep_7seeds.py

  # or background with logging:
  PYTHONPATH=. nohup python3 -u src/scripts/run_churn_sweep_7seeds.py \
      > training_logs/run_churn_sweep_7seeds.log 2>&1 &
"""
from __future__ import annotations

import json
import math
import os
import time
import copy

import numpy as np
from stable_baselines3 import PPO

from src.env.network_slicing_env import NetworkSlicingEnv
from src.train.config import ENV_CONFIG, PPO_CONFIG


PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RESULTS_DIR = os.path.join(PROJ_ROOT, "results")
MODELS_DIR = os.path.join(PROJ_ROOT, "models", "seed_power_paper")
LOGS_DIR = os.path.join(PROJ_ROOT, "training_logs")

CHURN_MULTIPLIERS = [1, 3, 5, 10]
SEEDS = [42, 123, 456, 789, 1011, 1213, 1415]  # 7-seed extension
N_EVAL_EPISODES = 20
EVAL_BASE_SEED = 1000


def make_env_config(multiplier: int) -> dict:
    cfg = copy.deepcopy(ENV_CONFIG)
    offset = math.log(float(multiplier))
    cfg["gamma0"] = [
        ENV_CONFIG["gamma0"][0] + offset,
        ENV_CONFIG["gamma0"][1] + offset,
    ]
    return cfg


def evaluate_policy(env_cfg, policy_fn, n_ep=N_EVAL_EPISODES,
                    base_seed=EVAL_BASE_SEED):
    """Generic evaluator returning aggregate metrics (no trajectories
    saved here to keep JSON small — per_episode_rewards retained for CI)."""
    env = NetworkSlicingEnv(config=env_cfg)
    rewards, revenues, penalties = [], [], []
    final_N_U, final_N_E = [], []
    for i in range(n_ep):
        obs, _ = env.reset(seed=base_seed + i)
        tot_r = tot_rev = tot_pen = 0.0
        last_info = None
        for _ in range(env.T):
            a = policy_fn(obs)
            obs, r, term, trunc, info = env.step(a)
            tot_r += r
            tot_rev += info["revenue"]
            tot_pen += info["penalty"]
            last_info = info
            if term or trunc:
                break
        rewards.append(tot_r)
        revenues.append(tot_rev)
        penalties.append(tot_pen)
        final_N_U.append(last_info["N_U"])
        final_N_E.append(last_info["N_E"])
    return {
        "n_eval_episodes": int(n_ep),
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "mean_revenue": float(np.mean(revenues)),
        "mean_penalty": float(np.mean(penalties)),
        "mean_final_N_U": float(np.mean(final_N_U)),
        "mean_final_N_E": float(np.mean(final_N_E)),
        "per_episode_rewards": list(map(float, rewards)),
    }


def train_one_seed(env_cfg, seed: int):
    env = NetworkSlicingEnv(config=env_cfg)
    hp = {k: v for k, v in PPO_CONFIG.items()
          if k not in ("total_timesteps", "seed")}
    model = PPO("MlpPolicy", env, seed=seed, verbose=0, **hp)
    model.learn(total_timesteps=PPO_CONFIG["total_timesteps"])
    return model


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)

    print(f"[init] 7-seed sweep · SEEDS={SEEDS} · m={CHURN_MULTIPLIERS}")
    print(f"[init] MODELS_DIR={MODELS_DIR}")

    all_results = {
        "config": {
            "extended_env": False,
            "seeds": SEEDS,
            "n_seeds": len(SEEDS),
            "multipliers": CHURN_MULTIPLIERS,
            "n_eval_episodes": N_EVAL_EPISODES,
            "eval_base_seed": EVAL_BASE_SEED,
            "ppo_config": {k: (v if not isinstance(v, dict) else str(v))
                            for k, v in PPO_CONFIG.items()
                            if not callable(v)},
            "note": (
                "Paper env (cohort_aware=False) re-trained with 7 seeds "
                "for statistical-power boost over the 3-seed published "
                "configuration. Compare against results/churn_sweep_results"
                ".json under the same m sweep."
            ),
        },
        "multipliers": {},
    }
    t_total = time.time()

    for mult in CHURN_MULTIPLIERS:
        print(f"\n{'='*70}\n  m={mult}x (gamma0 offset +{math.log(mult):.4f})"
              f"\n{'='*70}")
        env_cfg = make_env_config(mult)
        mult_results = {
            "multiplier": mult,
            "gamma0_adjusted": env_cfg["gamma0"],
        }

        # Max-Price baseline (deterministic, ref for uplift %)
        print(f"\n[m={mult}] Max-Price baseline...")
        t0 = time.time()
        mp = evaluate_policy(
            env_cfg,
            lambda obs: np.ones(4, dtype=np.float32),
            n_ep=N_EVAL_EPISODES,
        )
        mult_results["max_price"] = mp
        print(f"  reward={mp['mean_reward']:.1f} ({time.time()-t0:.1f}s)")

        # PPO 7-seed
        print(f"\n[m={mult}] PPO × 7 seeds...")
        per_seed = []
        for seed in SEEDS:
            t0 = time.time()
            model = train_one_seed(env_cfg, seed=seed)
            model_path = os.path.join(
                MODELS_DIR, f"ppo_mult{mult}_seed{seed}.zip"
            )
            model.save(model_path)

            r = evaluate_policy(
                env_cfg,
                lambda o, mdl=model: mdl.predict(o, deterministic=True)[0],
                n_ep=N_EVAL_EPISODES,
            )
            r["seed"] = int(seed)
            per_seed.append(r)
            print(f"  [seed={seed}] reward={r['mean_reward']:.1f} "
                  f"final_N_E={r['mean_final_N_E']:.0f} "
                  f"({(time.time()-t0)/60:.1f}min)")

        seed_means = [r["mean_reward"] for r in per_seed]
        mult_results["ppo_7seeds"] = {
            "mean_reward_across_seeds": float(np.mean(seed_means)),
            "std_reward_across_seeds": float(np.std(seed_means)),
            "per_seed": per_seed,
        }
        all_results["multipliers"][str(mult)] = mult_results

        # Save partial JSON after each m so a crash mid-sweep doesn't lose
        # earlier work
        with open(os.path.join(RESULTS_DIR, "seed_power_paper.json"), "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"  [m={mult}] saved partial results")

    print(f"\n[done] total {(time.time()-t_total)/3600:.2f} h")


if __name__ == "__main__":
    main()
