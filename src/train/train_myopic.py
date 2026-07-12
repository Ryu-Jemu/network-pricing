"""
Myopic PPO (γ=0) baseline — maximizes immediate reward only.
Proves whether RL's long-horizon planning adds value.
"""
import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from src.env.network_slicing_env import NetworkSlicingEnv
from src.train.config import ENV_CONFIG, MYOPIC_PPO_CONFIG
from src.train.train_sac import EpisodeLogCallback, evaluate_policy

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RESULTS_DIR = os.path.join(ROOT, "results")
MODELS_DIR = os.path.join(ROOT, "models", "multi_seed")


def train_myopic_ppo(seed=42, total_timesteps=None):
    env = Monitor(NetworkSlicingEnv(config=ENV_CONFIG))
    ts = total_timesteps or MYOPIC_PPO_CONFIG["total_timesteps"]

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=MYOPIC_PPO_CONFIG["learning_rate"],
        batch_size=MYOPIC_PPO_CONFIG["batch_size"],
        n_epochs=MYOPIC_PPO_CONFIG["n_epochs"],
        clip_range=MYOPIC_PPO_CONFIG["clip_range"],
        gae_lambda=MYOPIC_PPO_CONFIG["gae_lambda"],
        gamma=MYOPIC_PPO_CONFIG["gamma"],      # γ=0 (myopic)
        policy_kwargs=MYOPIC_PPO_CONFIG["policy_kwargs"],
        seed=seed,
        verbose=0,
    )

    callback = EpisodeLogCallback()
    model.learn(total_timesteps=ts, callback=callback, progress_bar=True)
    return model, callback


def main():
    import json
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    seed = MYOPIC_PPO_CONFIG["seed"]
    print(f"Training Myopic-PPO γ=0 (seed={seed})...")
    model, callback = train_myopic_ppo(seed=seed)

    model_path = os.path.join(MODELS_DIR, f"myopic_ppo_seed{seed}")
    model.save(model_path)
    print(f"Model saved to {model_path}.zip")

    print("Evaluating...")
    results = evaluate_policy(model, ENV_CONFIG, n_episodes=20, seed=seed)

    rewards = [r["total_reward"] for r in results]
    revenues = [r["total_revenue"] for r in results]
    penalties = [r["total_penalty"] for r in results]

    print(f"  Mean reward:  {np.mean(rewards):,.0f} ± {np.std(rewards):,.0f}")
    print(f"  Mean revenue: ${np.mean(revenues):,.0f}")
    print(f"  Mean penalty: ${np.mean(penalties):,.0f}")

    output = {
        "train_rewards": callback.episode_rewards,
        "train_final_N_U": callback.episode_final_N_U,
        "train_final_N_E": callback.episode_final_N_E,
        "eval_results": results,
        "config": {"env": ENV_CONFIG, "myopic_ppo": MYOPIC_PPO_CONFIG},
    }
    with open(os.path.join(RESULTS_DIR, f"myopic_ppo_seed{seed}.json"), "w") as f:
        json.dump(output, f, indent=2, default=float)

    print("Done.")


if __name__ == "__main__":
    main()
