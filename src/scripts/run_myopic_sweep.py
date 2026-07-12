"""
Myopic-PPO Churn Sweep: Train γ=0 PPO across churn multipliers.
================================================================
For each multiplier in {1, 3, 5, 10}, trains Myopic-PPO (γ=0) with 3 seeds.
Saves results to results/myopic_sweep_results.json.
Saves models to models/myopic_sweep/.

Expected result: Myopic-PPO ≈ Max-Price at all m values,
because γ=0 maximizes immediate reward → always charges max price.
"""
import os, json, time
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from src.env.network_slicing_env import NetworkSlicingEnv
from src.train.config import MYOPIC_PPO_CONFIG, EVAL_CONFIG
from src.train.train_sac import EpisodeLogCallback
from src.scripts.run_churn_sweep import (
    make_env_config,
    evaluate_ppo_with_trajectory,
)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

CHURN_MULTIPLIERS = [1, 3, 5, 10]
SEEDS = EVAL_CONFIG["train_seeds"]
N_EVAL_EPISODES = EVAL_CONFIG["n_eval_episodes"]
TOTAL_TIMESTEPS = MYOPIC_PPO_CONFIG["total_timesteps"]

RESULTS_DIR = os.path.join(ROOT, "results")
MODELS_DIR = os.path.join(ROOT, "models", "myopic_sweep")


def train_myopic_ppo_custom(env_config, seed=42):
    """Train Myopic-PPO (γ=0) with a custom env_config."""
    env = Monitor(NetworkSlicingEnv(config=env_config))

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=MYOPIC_PPO_CONFIG["learning_rate"],
        batch_size=MYOPIC_PPO_CONFIG["batch_size"],
        n_epochs=MYOPIC_PPO_CONFIG["n_epochs"],
        clip_range=MYOPIC_PPO_CONFIG["clip_range"],
        gae_lambda=MYOPIC_PPO_CONFIG["gae_lambda"],
        gamma=MYOPIC_PPO_CONFIG["gamma"],  # 0.0
        policy_kwargs=MYOPIC_PPO_CONFIG["policy_kwargs"],
        seed=seed,
        verbose=0,
    )

    callback = EpisodeLogCallback()
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callback, progress_bar=True)

    return model, callback


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    # Load existing churn sweep for Max-Price comparison
    sweep_path = os.path.join(RESULTS_DIR, "churn_sweep_results.json")
    if os.path.isfile(sweep_path):
        with open(sweep_path) as f:
            sweep_data = json.load(f)
    else:
        sweep_data = None

    all_results = {"multipliers": {}}
    total_start = time.time()

    for mult in CHURN_MULTIPLIERS:
        print("\n" + "=" * 70)
        print(f"  MYOPIC-PPO @ CHURN MULTIPLIER: {mult}x (gamma=0.0)")
        print("=" * 70)

        env_cfg = make_env_config(mult)
        seed_results = []

        for seed in SEEDS:
            print(f"\n    [{mult}x] Myopic-PPO seed={seed}...")
            t0 = time.time()

            model, cb = train_myopic_ppo_custom(env_cfg, seed=seed)

            # Save model
            model_path = os.path.join(MODELS_DIR, f"myopic_mult{mult}_seed{seed}")
            model.save(model_path)

            # Evaluate
            eval_result = evaluate_ppo_with_trajectory(
                model, env_cfg, n_episodes=N_EVAL_EPISODES, seed=seed
            )
            eval_result["seed"] = seed
            eval_result["train_rewards"] = cb.episode_rewards
            seed_results.append(eval_result)

            elapsed = time.time() - t0
            print(f"      Reward: {eval_result['mean_reward']:,.0f} ({elapsed:.0f}s)")

        # Aggregate
        seed_means = [r["mean_reward"] for r in seed_results]
        agg = {
            "mean_reward": float(np.mean(seed_means)),
            "std_reward": float(np.std(seed_means, ddof=1)),
            "mean_revenue": float(np.mean([r["mean_revenue"] for r in seed_results])),
            "mean_penalty": float(np.mean([r["mean_penalty"] for r in seed_results])),
            "mean_final_N_U": float(np.mean([r["mean_final_N_U"] for r in seed_results])),
            "mean_final_N_E": float(np.mean([r["mean_final_N_E"] for r in seed_results])),
            "per_seed": seed_results,
        }
        all_results["multipliers"][str(mult)] = agg

        # Compare with Max-Price from churn sweep
        mp_reward = None
        if sweep_data:
            mp_data = sweep_data.get("multipliers", {}).get(str(mult), {}).get("max_price", {})
            mp_reward = mp_data.get("mean_reward")

        if mp_reward is not None:
            diff_pct = abs(agg["mean_reward"] - mp_reward) / abs(mp_reward) * 100
            status = "OK" if diff_pct < 5 else "WARNING"
            print(f"\n    [{mult}x] Myopic vs Max-Price: "
                  f"{agg['mean_reward']:,.0f} vs {mp_reward:,.0f} "
                  f"(diff={diff_pct:.1f}%) [{status}]")
        else:
            print(f"\n    [{mult}x] Myopic aggregate: {agg['mean_reward']:,.0f}")

    # Save results
    outpath = os.path.join(RESULTS_DIR, "myopic_sweep_results.json")
    with open(outpath, "w") as f:
        json.dump(all_results, f, indent=2, default=float)

    total_elapsed = time.time() - total_start
    print(f"\n{'=' * 70}")
    print(f"SAVED: {outpath}")
    print(f"Total time: {total_elapsed/60:.1f} minutes")
    print(f"{'=' * 70}")

    # Summary
    print(f"\n{'Mult':>5} {'Myopic-PPO':>12} {'Max-Price':>12} {'Diff%':>8}")
    print("-" * 40)
    for mk, mr in all_results["multipliers"].items():
        myopic_r = mr["mean_reward"]
        mp_r = None
        if sweep_data:
            mp_r = sweep_data.get("multipliers", {}).get(mk, {}).get("max_price", {}).get("mean_reward")
        mp_str = f"{mp_r:>12,.0f}" if mp_r else "       N/A"
        diff = f"{abs(myopic_r - mp_r) / abs(mp_r) * 100:>7.1f}%" if mp_r else "    N/A"
        print(f"{mk + 'x':>5} {myopic_r:>12,.0f} {mp_str} {diff}")


if __name__ == "__main__":
    main()
