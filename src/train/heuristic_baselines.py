"""Hand-tuned heuristic pricing policies for NetworkSlicingEnv.

Two industry-style baselines without any RL training, useful as "why is RL
necessary" reference points in Q&A:

  B-add-3  Load-threshold rule
      If the normalized active eMBB subscriber count falls below a threshold,
      lower price aggressively to slow churn; otherwise extract revenue at
      higher price. State-dependent (uses obs[1] = N_E / N_init_E).

  B-add-4  Peak / off-peak rule
      Apply higher prices during synthetic "peak" hours of the day
      (h ∈ {9, 10, 11, 18, 19, 20}), lower prices otherwise. Stateless
      with respect to the env observation, but uses an internal step
      counter; a fresh policy must be created per evaluation episode so
      the counter resets correctly.

Both policies return a 4-D action ∈ [0, 1]^4 in the convention of
NetworkSlicingEnv: (F_U_norm, p_U_norm, F_E_norm, p_E_norm).
"""
import numpy as np


# ─────────── Defaults: low (retention) / high (revenue) action vectors ──
# Both inferred from the reference action used in the published paper
# (REFERENCE_ACTION = [0.5, 0.5, 0.3, 0.25]) and from the Max-Price extreme
# ([1, 1, 1, 1]). The low setting is roughly 60% of Reference and the high
# setting is roughly 70-80% of Max-Price, leaving room above/below.
LOW_ACTION = np.array([0.30, 0.30, 0.20, 0.20], dtype=np.float32)
HIGH_ACTION = np.array([0.80, 0.70, 0.60, 0.50], dtype=np.float32)
PEAK_ACTION = np.array([0.80, 0.70, 0.60, 0.50], dtype=np.float32)
OFFPEAK_ACTION = np.array([0.40, 0.40, 0.30, 0.25], dtype=np.float32)

DEFAULT_PEAK_HOURS = frozenset({9, 10, 11, 18, 19, 20})


def make_load_threshold_policy(threshold: float = 0.50,
                                low: np.ndarray = LOW_ACTION,
                                high: np.ndarray = HIGH_ACTION):
    """B-add-3: state-dependent threshold policy.

    Args:
        threshold: switch on N_E_norm = obs[1]. Below threshold → low action.
        low: action when below threshold (gentle pricing to keep subscribers).
        high: action when at/above threshold (more aggressive pricing).
    Returns:
        policy(obs) -> 4-D float32 action.
    """
    low = np.asarray(low, dtype=np.float32)
    high = np.asarray(high, dtype=np.float32)

    def policy(obs):
        return low if obs[1] < threshold else high
    return policy


def make_peak_offpeak_policy(peak_hours=DEFAULT_PEAK_HOURS,
                              peak: np.ndarray = PEAK_ACTION,
                              off: np.ndarray = OFFPEAK_ACTION):
    """B-add-4: time-of-day rule with internal step counter.

    The env does not expose t in obs, so the counter is kept inside this
    closure. Caller MUST instantiate one policy per evaluation episode so
    the counter starts at t = 0.

    Args:
        peak_hours: set of hour-of-day values for which `peak` action is used.
        peak: action during peak hours (higher prices).
        off:  action during off-peak hours (lower prices).
    Returns:
        policy(obs) -> 4-D float32 action.
    """
    peak = np.asarray(peak, dtype=np.float32)
    off = np.asarray(off, dtype=np.float32)
    state = {"t": 0}

    def policy(obs):
        h = state["t"] % 24
        state["t"] += 1
        return peak if h in peak_hours else off
    return policy


def evaluate_policy_factory(env_config, policy_factory, n_episodes=20,
                            base_seed=1000):
    """Run n_episodes evaluation; build a fresh policy each episode.

    This is the recommended evaluation entry for stateful heuristics
    (e.g. peak/off-peak with an internal counter).

    Returns:
        dict with mean_reward / std_reward / mean_revenue / mean_penalty /
        mean_final_N_U / mean_final_N_E / per_episode_rewards /
        trajectory_N_E_mean / trajectory_N_E_std (720-point lists).
    """
    from src.env.network_slicing_env import NetworkSlicingEnv

    env = NetworkSlicingEnv(config=env_config)
    T = env.T
    rewards = np.empty(n_episodes, dtype=np.float64)
    revenues = np.empty(n_episodes, dtype=np.float64)
    penalties = np.empty(n_episodes, dtype=np.float64)
    final_N_U = np.empty(n_episodes, dtype=np.float64)
    final_N_E = np.empty(n_episodes, dtype=np.float64)
    traj_N_E = np.empty((n_episodes, T), dtype=np.float64)

    for i in range(n_episodes):
        obs, _ = env.reset(seed=base_seed + i)
        policy = policy_factory()
        tot_r = tot_rev = tot_pen = 0.0
        last_info = None
        for t in range(T):
            a = policy(obs)
            obs, r, term, trunc, info = env.step(a)
            tot_r += r
            tot_rev += info["revenue"]
            tot_pen += info["penalty"]
            traj_N_E[i, t] = info["N_E"]
            last_info = info
            if term or trunc:
                for tt in range(t + 1, T):
                    traj_N_E[i, tt] = traj_N_E[i, t]
                break
        rewards[i] = tot_r
        revenues[i] = tot_rev
        penalties[i] = tot_pen
        final_N_U[i] = last_info["N_U"]
        final_N_E[i] = last_info["N_E"]

    return {
        "n_eval_episodes": int(n_episodes),
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "mean_revenue": float(np.mean(revenues)),
        "mean_penalty": float(np.mean(penalties)),
        "mean_final_N_U": float(np.mean(final_N_U)),
        "mean_final_N_E": float(np.mean(final_N_E)),
        "per_episode_rewards": rewards.tolist(),
        "trajectory_N_E_mean": traj_N_E.mean(axis=0).tolist(),
        "trajectory_N_E_std": traj_N_E.std(axis=0).tolist(),
    }
