"""
TD3 (Twin Delayed DDPG) training for network slicing pricing.
Off-policy + deterministic policy — tests SAC's entropy hypothesis.
"""
import os
import numpy as np
from stable_baselines3 import TD3
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise

from src.env.network_slicing_env import NetworkSlicingEnv
from src.train.config import ENV_CONFIG, TD3_CONFIG
from src.train.train_sac import EpisodeLogCallback, evaluate_policy

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RESULTS_DIR = os.path.join(ROOT, "results")
MODELS_DIR = os.path.join(ROOT, "models", "multi_seed")


def train_td3(seed=42, total_timesteps=None):
    env = Monitor(NetworkSlicingEnv(config=ENV_CONFIG))
    ts = total_timesteps or TD3_CONFIG["total_timesteps"]

    n_actions = env.action_space.shape[0]
    action_noise = NormalActionNoise(
        mean=np.zeros(n_actions),
        sigma=0.1 * np.ones(n_actions),
    )

    model = TD3(
        "MlpPolicy",
        env,
        learning_rate=TD3_CONFIG["learning_rate"],
        batch_size=TD3_CONFIG["batch_size"],
        buffer_size=TD3_CONFIG["buffer_size"],
        tau=TD3_CONFIG["tau"],
        gamma=TD3_CONFIG["gamma"],
        action_noise=action_noise,
        policy_kwargs=TD3_CONFIG["policy_kwargs"],
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

    seed = TD3_CONFIG["seed"]
    print(f"Training TD3 (seed={seed})...")
    model, callback = train_td3(seed=seed)

    model_path = os.path.join(MODELS_DIR, f"td3_seed{seed}")
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
        "config": {"env": ENV_CONFIG, "td3": TD3_CONFIG},
    }
    with open(os.path.join(RESULTS_DIR, f"td3_seed{seed}.json"), "w") as f:
        json.dump(output, f, indent=2, default=float)

    print("Done.")


if __name__ == "__main__":
    main()
