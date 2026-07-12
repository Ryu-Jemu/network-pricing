"""
Churn Rate Sweep: Train PPO and evaluate baselines across churn multipliers.
=============================================================================
For each multiplier in {1, 3, 5, 10}, adjusts gamma0 += ln(mult) and:
  - Trains PPO with 3 seeds (42, 123, 456), 500 episodes each
  - Evaluates Max-Price baseline (20 episodes per seed)
  - Runs Static-Oracle grid search
  - Saves results to results/churn_sweep_results.json
  - Saves models to models/churn_sweep/
"""
import os, json, copy, time
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from src.env.network_slicing_env import NetworkSlicingEnv
from src.train.config import ENV_CONFIG, PPO_CONFIG, EVAL_CONFIG, ORACLE_GRID
from src.train.train_sac import EpisodeLogCallback
from src.train.run_multi_seed import run_static_oracle

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

CHURN_MULTIPLIERS = [1, 3, 5, 10]
SEEDS = EVAL_CONFIG["train_seeds"]
N_EVAL_EPISODES = EVAL_CONFIG["n_eval_episodes"]
TOTAL_TIMESTEPS = PPO_CONFIG["total_timesteps"]

RESULTS_DIR = os.path.join(ROOT, "results")
MODELS_DIR = os.path.join(ROOT, "models", "churn_sweep")


def make_env_config(multiplier):
    """Create env config with adjusted gamma0 for given churn multiplier."""
    cfg = copy.deepcopy(ENV_CONFIG)
    offset = float(np.log(multiplier))
    cfg["gamma0"] = [
        ENV_CONFIG["gamma0"][0] + offset,
        ENV_CONFIG["gamma0"][1] + offset,
    ]
    return cfg


def train_ppo_custom(env_config, seed=42, total_timesteps=None):
    """Train PPO with a custom env_config (not the global ENV_CONFIG)."""
    env = Monitor(NetworkSlicingEnv(config=env_config))
    ts = total_timesteps or TOTAL_TIMESTEPS

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=PPO_CONFIG["learning_rate"],
        batch_size=PPO_CONFIG["batch_size"],
        n_epochs=PPO_CONFIG["n_epochs"],
        clip_range=PPO_CONFIG["clip_range"],
        gae_lambda=PPO_CONFIG["gae_lambda"],
        gamma=PPO_CONFIG["gamma"],
        policy_kwargs=PPO_CONFIG["policy_kwargs"],
        seed=seed,
        verbose=0,
    )

    callback = EpisodeLogCallback()
    model.learn(total_timesteps=ts, callback=callback, progress_bar=True)

    return model, callback


def evaluate_baseline_with_trajectory(env_config, action_fn, n_episodes=20, seed=42):
    """Evaluate a fixed policy, returning per-step trajectory data."""
    env = NetworkSlicingEnv(config=env_config)
    all_episodes = []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        tot_r, tot_rev, tot_pen = 0.0, 0.0, 0.0
        traj_N_E = []

        for _ in range(env.T):
            action = action_fn(obs)
            obs, r, term, trunc, info = env.step(action)
            tot_r += r
            tot_rev += info["revenue"]
            tot_pen += info["penalty"]
            traj_N_E.append(float(info["N_E"]))

        all_episodes.append({
            "total_reward": tot_r,
            "total_revenue": tot_rev,
            "total_penalty": tot_pen,
            "final_N_U": float(info["N_U"]),
            "final_N_E": float(info["N_E"]),
            "trajectory_N_E": traj_N_E,
        })

    rewards = [e["total_reward"] for e in all_episodes]
    return {
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "mean_revenue": float(np.mean([e["total_revenue"] for e in all_episodes])),
        "mean_penalty": float(np.mean([e["total_penalty"] for e in all_episodes])),
        "mean_final_N_U": float(np.mean([e["final_N_U"] for e in all_episodes])),
        "mean_final_N_E": float(np.mean([e["final_N_E"] for e in all_episodes])),
        "trajectory_N_E_mean": [
            float(np.mean([ep["trajectory_N_E"][t] for ep in all_episodes]))
            for t in range(env.T)
        ],
        "trajectory_N_E_std": [
            float(np.std([ep["trajectory_N_E"][t] for ep in all_episodes]))
            for t in range(env.T)
        ],
    }


def evaluate_ppo_with_trajectory(model, env_config, n_episodes=20, seed=42):
    """Evaluate trained PPO model, returning per-step trajectory data."""
    env = NetworkSlicingEnv(config=env_config)
    all_episodes = []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        tot_r, tot_rev, tot_pen = 0.0, 0.0, 0.0
        traj_N_E = []

        for _ in range(env.T):
            action, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, info = env.step(action)
            tot_r += r
            tot_rev += info["revenue"]
            tot_pen += info["penalty"]
            traj_N_E.append(float(info["N_E"]))

        all_episodes.append({
            "total_reward": tot_r,
            "total_revenue": tot_rev,
            "total_penalty": tot_pen,
            "final_N_U": float(info["N_U"]),
            "final_N_E": float(info["N_E"]),
            "trajectory_N_E": traj_N_E,
        })

    rewards = [e["total_reward"] for e in all_episodes]
    return {
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "mean_revenue": float(np.mean([e["total_revenue"] for e in all_episodes])),
        "mean_penalty": float(np.mean([e["total_penalty"] for e in all_episodes])),
        "mean_final_N_U": float(np.mean([e["final_N_U"] for e in all_episodes])),
        "mean_final_N_E": float(np.mean([e["final_N_E"] for e in all_episodes])),
        "per_episode_rewards": [e["total_reward"] for e in all_episodes],
        "trajectory_N_E_mean": [
            float(np.mean([ep["trajectory_N_E"][t] for ep in all_episodes]))
            for t in range(NetworkSlicingEnv(config=env_config).T)
        ],
        "trajectory_N_E_std": [
            float(np.std([ep["trajectory_N_E"][t] for ep in all_episodes]))
            for t in range(NetworkSlicingEnv(config=env_config).T)
        ],
    }


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    all_results = {"multipliers": {}, "config": {"base_env": ENV_CONFIG}}
    total_start = time.time()

    for mult in CHURN_MULTIPLIERS:
        print("\n" + "=" * 70)
        print(f"  CHURN MULTIPLIER: {mult}x (gamma0 offset: +{np.log(mult):.4f})")
        print("=" * 70)

        env_cfg = make_env_config(mult)
        mult_key = str(mult)
        mult_results = {
            "multiplier": mult,
            "gamma0_adjusted": env_cfg["gamma0"],
            "monthly_churn_target": f"~{mult}%",
        }

        # ── 1. Max-Price baseline ────────────────────────────────
        print(f"\n  [{mult}x] Evaluating Max-Price...")
        max_price_result = evaluate_baseline_with_trajectory(
            env_cfg,
            lambda obs: np.ones(4, dtype=np.float32),
            n_episodes=N_EVAL_EPISODES,
        )
        mult_results["max_price"] = max_price_result
        print(f"    Max-Price reward: {max_price_result['mean_reward']:,.0f}")

        # ── 2. Static-Oracle grid search ─────────────────────────
        print(f"\n  [{mult}x] Running Static-Oracle grid search...")
        best_action, best_reward = run_static_oracle(env_cfg, ORACLE_GRID)
        oracle_action = np.array(best_action, dtype=np.float32)
        oracle_result = evaluate_baseline_with_trajectory(
            env_cfg,
            lambda obs, a=oracle_action: a,
            n_episodes=N_EVAL_EPISODES,
        )
        mult_results["static_oracle"] = oracle_result
        mult_results["static_oracle"]["best_action"] = best_action
        print(f"    Static-Oracle reward: {oracle_result['mean_reward']:,.0f}")

        # ── 3. PPO training (multi-seed) ─────────────────────────
        print(f"\n  [{mult}x] Training PPO across {len(SEEDS)} seeds...")
        ppo_seed_results = []

        for seed in SEEDS:
            print(f"\n    [{mult}x] PPO seed={seed}...")
            t0 = time.time()

            model, cb = train_ppo_custom(env_cfg, seed=seed)

            # Save model
            model_path = os.path.join(MODELS_DIR, f"ppo_mult{mult}_seed{seed}")
            model.save(model_path)

            # Evaluate with trajectory
            eval_result = evaluate_ppo_with_trajectory(
                model, env_cfg, n_episodes=N_EVAL_EPISODES, seed=seed
            )
            eval_result["seed"] = seed
            eval_result["train_rewards"] = cb.episode_rewards
            ppo_seed_results.append(eval_result)

            elapsed = time.time() - t0
            print(f"      Reward: {eval_result['mean_reward']:,.0f} "
                  f"({elapsed:.0f}s)")

        # Aggregate PPO across seeds
        seed_means = [r["mean_reward"] for r in ppo_seed_results]
        mult_results["ppo"] = {
            "mean_reward": float(np.mean(seed_means)),
            "std_reward": float(np.std(seed_means, ddof=1)),
            "mean_revenue": float(np.mean([r["mean_revenue"] for r in ppo_seed_results])),
            "mean_penalty": float(np.mean([r["mean_penalty"] for r in ppo_seed_results])),
            "mean_final_N_U": float(np.mean([r["mean_final_N_U"] for r in ppo_seed_results])),
            "mean_final_N_E": float(np.mean([r["mean_final_N_E"] for r in ppo_seed_results])),
            "per_seed": ppo_seed_results,
        }
        print(f"\n    [{mult}x] PPO aggregate: "
              f"{np.mean(seed_means):,.0f} +/- {np.std(seed_means, ddof=1):,.0f}")

        # ── 4. Compute improvement ───────────────────────────────
        ppo_r = mult_results["ppo"]["mean_reward"]
        max_r = mult_results["max_price"]["mean_reward"]
        oracle_r = mult_results["static_oracle"]["mean_reward"]

        if max_r != 0:
            mult_results["ppo_vs_max_price_pct"] = float((ppo_r - max_r) / abs(max_r) * 100)
        if oracle_r != 0:
            mult_results["ppo_vs_oracle_pct"] = float((ppo_r - oracle_r) / abs(oracle_r) * 100)

        all_results["multipliers"][mult_key] = mult_results

    # ── Save all results ─────────────────────────────────────────
    outpath = os.path.join(RESULTS_DIR, "churn_sweep_results.json")
    with open(outpath, "w") as f:
        json.dump(all_results, f, indent=2, default=float)

    total_elapsed = time.time() - total_start
    print(f"\n{'=' * 70}")
    print(f"ALL RESULTS SAVED: {outpath}")
    print(f"Total time: {total_elapsed/60:.1f} minutes")
    print(f"{'=' * 70}")

    # ── Summary table ────────────────────────────────────────────
    print(f"\n{'Mult':>5} {'PPO':>12} {'Max-Price':>12} {'Oracle':>12} {'PPO vs Max':>12}")
    print("-" * 55)
    for mult_key, mr in all_results["multipliers"].items():
        ppo_r = mr["ppo"]["mean_reward"]
        max_r = mr["max_price"]["mean_reward"]
        orc_r = mr["static_oracle"]["mean_reward"]
        pct = mr.get("ppo_vs_max_price_pct", 0)
        print(f"{mult_key + 'x':>5} {ppo_r:>12,.0f} {max_r:>12,.0f} {orc_r:>12,.0f} {pct:>+11.1f}%")


if __name__ == "__main__":
    main()
