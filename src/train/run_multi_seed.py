"""
Multi-seed training and evaluation orchestrator.
Trains PPO, SAC, TD3 across 3 seeds and aggregates results.
Also runs baselines and Myopic-PPO.
"""
import os
import json
import numpy as np
from itertools import product as cartprod

from src.train.config import (
    ENV_CONFIG, SAC_CONFIG, PPO_CONFIG, TD3_CONFIG,
    MYOPIC_PPO_CONFIG, EVAL_CONFIG, REFERENCE_ACTION, ORACLE_GRID,
)
from src.train.train_sac import train_sac, evaluate_policy
from src.train.train_ppo import train_ppo
from src.train.train_td3 import train_td3
from src.train.train_myopic import train_myopic_ppo
from src.env.network_slicing_env import NetworkSlicingEnv

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RESULTS_DIR = os.path.join(ROOT, "results")
MODELS_DIR = os.path.join(ROOT, "models", "multi_seed")


def run_static_oracle(env_config, grid_cfg, seed=42):
    """Grid search for best constant action."""
    best_reward = -float("inf")
    best_action = None

    combos = list(cartprod(
        grid_cfg["F_U_range"], grid_cfg["p_U_range"],
        grid_cfg["F_E_range"], grid_cfg["p_E_range"],
    ))
    print(f"  Static-Oracle: searching {len(combos)} combinations...")

    for F_U, p_U, F_E, p_E in combos:
        action = np.array([F_U, p_U, F_E, p_E], dtype=np.float32)
        env = NetworkSlicingEnv(config=env_config)
        ep_rewards = []

        for ep in range(grid_cfg["n_eval_episodes"]):
            obs, _ = env.reset(seed=seed + ep)
            total_r = 0.0
            for _ in range(env.T):
                obs, r, term, trunc, info = env.step(action)
                total_r += r
            ep_rewards.append(total_r)

        mean_r = np.mean(ep_rewards)
        if mean_r > best_reward:
            best_reward = mean_r
            best_action = action.tolist()

    print(f"  Best constant action: {best_action} → reward={best_reward:,.0f}")
    return best_action, best_reward


def run_baseline_policy(env_config, action_fn, name, n_episodes=20, seed=42):
    """Evaluate a fixed policy."""
    env = NetworkSlicingEnv(config=env_config)
    rewards, revenues, penalties = [], [], []
    final_N_U, final_N_E = [], []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        tot_r, tot_rev, tot_pen = 0.0, 0.0, 0.0
        for _ in range(env.T):
            action = action_fn(obs)
            obs, r, term, trunc, info = env.step(action)
            tot_r += r
            tot_rev += info["revenue"]
            tot_pen += info["penalty"]
        rewards.append(tot_r)
        revenues.append(tot_rev)
        penalties.append(tot_pen)
        final_N_U.append(info["N_U"])
        final_N_E.append(info["N_E"])

    return {
        "name": name,
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "mean_revenue": float(np.mean(revenues)),
        "mean_penalty": float(np.mean(penalties)),
        "mean_final_N_U": float(np.mean(final_N_U)),
        "mean_final_N_E": float(np.mean(final_N_E)),
    }


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    seeds = EVAL_CONFIG["train_seeds"]
    all_results = {}

    # ── 1. Fixed baselines (seed-independent) ─────────────────────
    print("=" * 60)
    print("Phase 1: Fixed baselines")
    print("=" * 60)

    rng = np.random.default_rng(42)

    all_results["static_heuristic"] = run_baseline_policy(
        ENV_CONFIG,
        lambda obs: np.array(REFERENCE_ACTION, dtype=np.float32),
        "Static-Heuristic",
    )
    print(f"  Static-Heur.: {all_results['static_heuristic']['mean_reward']:,.0f}")

    all_results["random"] = run_baseline_policy(
        ENV_CONFIG,
        lambda obs: rng.uniform(0, 1, size=4).astype(np.float32),
        "Random",
    )
    print(f"  Random: {all_results['random']['mean_reward']:,.0f}")

    all_results["max_price"] = run_baseline_policy(
        ENV_CONFIG,
        lambda obs: np.ones(4, dtype=np.float32),
        "Max-Price",
    )
    print(f"  Max-Price: {all_results['max_price']['mean_reward']:,.0f}")

    # ── 2. Static-Oracle (grid search) ────────────────────────────
    print("\n" + "=" * 60)
    print("Phase 2: Static-Oracle grid search")
    print("=" * 60)

    best_action, best_reward = run_static_oracle(ENV_CONFIG, ORACLE_GRID)
    oracle_action = np.array(best_action, dtype=np.float32)
    all_results["static_oracle"] = run_baseline_policy(
        ENV_CONFIG,
        lambda obs: oracle_action,
        f"Static-Oracle {best_action}",
        n_episodes=20,
    )
    print(f"  Static-Oracle: {all_results['static_oracle']['mean_reward']:,.0f}")

    # ── 3. Myopic-PPO (γ=0) ──────────────────────────────────────
    print("\n" + "=" * 60)
    print("Phase 3: Myopic-PPO (γ=0)")
    print("=" * 60)

    model_m, cb_m = train_myopic_ppo(seed=42)
    model_m.save(os.path.join(MODELS_DIR, "myopic_ppo_seed42"))
    myopic_eval = evaluate_policy(model_m, ENV_CONFIG, n_episodes=20, seed=42)
    myopic_rewards = [r["total_reward"] for r in myopic_eval]
    all_results["myopic_ppo"] = {
        "name": "Myopic-PPO (γ=0)",
        "mean_reward": float(np.mean(myopic_rewards)),
        "std_reward": float(np.std(myopic_rewards)),
        "mean_revenue": float(np.mean([r["total_revenue"] for r in myopic_eval])),
        "mean_penalty": float(np.mean([r["total_penalty"] for r in myopic_eval])),
        "mean_final_N_U": float(np.mean([r["final_N_U"] for r in myopic_eval])),
        "mean_final_N_E": float(np.mean([r["final_N_E"] for r in myopic_eval])),
    }
    print(f"  Myopic-PPO: {all_results['myopic_ppo']['mean_reward']:,.0f}")

    # ── 4. Multi-seed RL training ─────────────────────────────────
    print("\n" + "=" * 60)
    print(f"Phase 4: Multi-seed RL (seeds={seeds})")
    print("=" * 60)

    algo_results = {"PPO": [], "SAC": [], "TD3": []}

    for seed in seeds:
        print(f"\n--- Seed {seed} ---")

        # PPO
        print(f"  Training PPO (seed={seed})...")
        model_ppo, cb_ppo = train_ppo(seed=seed)
        model_ppo.save(os.path.join(MODELS_DIR, f"ppo_seed{seed}"))
        eval_ppo = evaluate_policy(model_ppo, ENV_CONFIG, n_episodes=20, seed=seed)
        algo_results["PPO"].append({
            "seed": seed,
            "eval_results": eval_ppo,
            "train_rewards": cb_ppo.episode_rewards,
        })
        ppo_r = np.mean([r["total_reward"] for r in eval_ppo])
        print(f"    PPO reward: {ppo_r:,.0f}")

        # SAC
        print(f"  Training SAC (seed={seed})...")
        model_sac, cb_sac = train_sac(seed=seed)
        model_sac.save(os.path.join(MODELS_DIR, f"sac_seed{seed}"))
        eval_sac = evaluate_policy(model_sac, ENV_CONFIG, n_episodes=20, seed=seed)
        algo_results["SAC"].append({
            "seed": seed,
            "eval_results": eval_sac,
            "train_rewards": cb_sac.episode_rewards,
        })
        sac_r = np.mean([r["total_reward"] for r in eval_sac])
        print(f"    SAC reward: {sac_r:,.0f}")

        # TD3
        print(f"  Training TD3 (seed={seed})...")
        model_td3, cb_td3 = train_td3(seed=seed)
        model_td3.save(os.path.join(MODELS_DIR, f"td3_seed{seed}"))
        eval_td3 = evaluate_policy(model_td3, ENV_CONFIG, n_episodes=20, seed=seed)
        algo_results["TD3"].append({
            "seed": seed,
            "eval_results": eval_td3,
            "train_rewards": cb_td3.episode_rewards,
        })
        td3_r = np.mean([r["total_reward"] for r in eval_td3])
        print(f"    TD3 reward: {td3_r:,.0f}")

    # ── 5. Aggregate multi-seed results ───────────────────────────
    print("\n" + "=" * 60)
    print("Phase 5: Aggregation")
    print("=" * 60)

    for algo_name, seed_runs in algo_results.items():
        seed_means = []
        seed_revenues = []
        seed_penalties = []
        seed_N_U = []
        seed_N_E = []
        for run in seed_runs:
            rs = [r["total_reward"] for r in run["eval_results"]]
            revs = [r["total_revenue"] for r in run["eval_results"]]
            pens = [r["total_penalty"] for r in run["eval_results"]]
            nus = [r["final_N_U"] for r in run["eval_results"]]
            nes = [r["final_N_E"] for r in run["eval_results"]]
            seed_means.append(np.mean(rs))
            seed_revenues.append(np.mean(revs))
            seed_penalties.append(np.mean(pens))
            seed_N_U.append(np.mean(nus))
            seed_N_E.append(np.mean(nes))

        all_results[algo_name.lower()] = {
            "name": algo_name,
            "mean_reward": float(np.mean(seed_means)),
            "std_reward_across_seeds": float(np.std(seed_means)),
            "mean_revenue": float(np.mean(seed_revenues)),
            "mean_penalty": float(np.mean(seed_penalties)),
            "mean_final_N_U": float(np.mean(seed_N_U)),
            "mean_final_N_E": float(np.mean(seed_N_E)),
            "per_seed": [
                {"seed": run["seed"], "mean_reward": float(sm)}
                for run, sm in zip(seed_runs, seed_means)
            ],
        }
        print(f"  {algo_name}: {np.mean(seed_means):,.0f} ± {np.std(seed_means):,.0f} "
              f"(seeds: {[f'{s:,.0f}' for s in seed_means]})")

    # ── 6. Save everything ────────────────────────────────────────
    output = {
        "baselines": {
            k: all_results[k] for k in
            ["static_heuristic", "random", "max_price", "static_oracle", "myopic_ppo"]
        },
        "rl_algorithms": {
            k: all_results[k] for k in ["ppo", "sac", "td3"]
        },
        "multi_seed_detail": algo_results,
        "config": {
            "env": ENV_CONFIG,
            "ppo": PPO_CONFIG,
            "sac": SAC_CONFIG,
            "td3": TD3_CONFIG,
            "myopic_ppo": MYOPIC_PPO_CONFIG,
            "eval": EVAL_CONFIG,
        },
    }

    outpath = os.path.join(RESULTS_DIR, "multi_seed_results.json")
    with open(outpath, "w") as f:
        json.dump(output, f, indent=2, default=float)
    print(f"\nAll results saved to {outpath}")

    # ── Summary table ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Policy':<20} {'Reward':>10} {'Revenue(M)':>12} {'Penalty(M)':>12} {'N_E':>8}")
    print("-" * 62)
    for key in ["ppo", "sac", "td3", "myopic_ppo", "static_oracle",
                "max_price", "random", "static_heuristic"]:
        r = all_results[key]
        std_str = ""
        if "std_reward_across_seeds" in r:
            std_str = f" ±{r['std_reward_across_seeds']:,.0f}"
        print(f"{r['name']:<20} {r['mean_reward']:>8,.0f}{std_str:>5} "
              f"{r['mean_revenue']/1e6:>10,.1f} "
              f"{r['mean_penalty']/1e6:>10,.1f} "
              f"{r['mean_final_N_E']:>8,.0f}")


if __name__ == "__main__":
    main()
