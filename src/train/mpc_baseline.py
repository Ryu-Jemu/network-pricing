"""Oracle Model-Predictive-Control baseline (improvement-5).

At each timestep, the controller has perfect knowledge of the env
parameters and uses Monte-Carlo rollouts of length H to score a small
candidate action set. Picks the action with highest expected
H-step return.

This is an UPPER BOUND on what model-based RL with a learned model
can achieve — it isolates the value of "having a planning horizon"
from "having a perfect model" so PPO's advantage over MPC reflects
genuinely sequential learning gains beyond planning.

Why include this baseline:
- Static-Oracle is the best CONSTANT action; PPO can change action
  over time and so beats it.
- Max-Price is a cliff-policy.
- An MPC oracle is the natural intermediate: it adapts per step
  using a planning horizon but has no learning of multi-period
  dynamics beyond the rollout horizon.

Reference:
- Camacho & Bordons. "Model Predictive Control." Springer, 2007.
  (textbook formulation of receding-horizon control)
- Hafner et al. "Mastering Diverse Control Tasks through World
  Models" (DreamerV3). Nature 2025 / Google DeepMind. — illustrates
  that with a perfect/learned model, planning is competitive with
  model-free RL.
"""
import numpy as np

from src.env.network_slicing_env import NetworkSlicingEnv


def _simulate_rollout(env_template_cfg, init_state, action,
                      H, n_rollouts, gamma, seed):
    """Simulate H-step rollouts holding `action` fixed; return mean
    discounted return.

    init_state: (N_U, N_E, eta_U_prev, eta_E_prev) tuple.
    """
    rng = np.random.default_rng(seed)
    returns = []
    for r in range(n_rollouts):
        env = NetworkSlicingEnv(config=env_template_cfg)
        env.reset(seed=int(rng.integers(0, 1 << 31)))
        # Override internal state with init_state. Dynamics are driven by
        # N_cohort (always allocated), so it must be set too; MPC supports
        # only the single-cohort environment.
        assert env.n_cohorts == 1, (
            "MPC state override supports n_cohorts=1 only"
        )
        env.N_cohort = np.array(
            [[float(init_state[0])], [float(init_state[1])]],
            dtype=np.float64,
        )
        env.N = env.N_cohort.sum(axis=1)
        env.eta_prev = np.array(init_state[2:4], dtype=np.float64)
        env.t = 0
        ret = 0.0
        disc = 1.0
        for h in range(H):
            _, rew, term, trunc, _ = env.step(action.astype(np.float32))
            ret += disc * rew
            disc *= gamma
            if term or trunc:
                break
        returns.append(ret)
    return float(np.mean(returns))


def mpc_step(
    env_template_cfg, init_state,
    candidate_actions, H, n_rollouts=5, gamma=0.99, seed=0,
):
    """Pick best of `candidate_actions` by Monte-Carlo rollout
    score."""
    scores = []
    for i, a in enumerate(candidate_actions):
        s = _simulate_rollout(
            env_template_cfg, init_state, a, H, n_rollouts, gamma,
            seed + i * 1000,
        )
        scores.append(s)
    best = int(np.argmax(scores))
    return candidate_actions[best], scores


def make_action_candidates(n_grid_per_dim=3):
    """Coarse 4-D grid of candidate constant actions for the MPC
    inner optimisation. Default 3^4=81 candidates per planning step.

    The grid spans the FULL action box [0,1] including the corners:
    trained policies routinely saturate at box corners (e.g. [1,0,0,1]),
    and the BO oracle is corner-seeded for the same fairness reason —
    an inner grid capped at 0.9 would handicap MPC against them.
    """
    grid = np.linspace(0.0, 1.0, n_grid_per_dim)
    return np.array(np.meshgrid(grid, grid, grid, grid)).reshape(4, -1).T


def run_mpc_episode_protocol(
    env_config, H=24, n_rollouts=3, n_grid=3, gamma=0.99, seed=1000,
    replan_every=24,
):
    """Run one MPC-controlled episode reset at the given eval seed,
    returning reward AND the CMDP cost J_C (from info['cost'], i.e.
    the env's eta_sla). Replans every `replan_every` steps."""
    env = NetworkSlicingEnv(config=env_config)
    obs, _ = env.reset(seed=seed)
    candidates = make_action_candidates(n_grid)
    rng = np.random.default_rng(seed * 13 + 7)

    total_reward = 0.0
    total_cost = 0.0
    current_action = candidates[0]
    for t in range(env.T):
        if t % replan_every == 0:
            init_state = (
                env.N[0], env.N[1],
                env.eta_prev[0], env.eta_prev[1],
            )
            best_a, _ = mpc_step(
                env_config, init_state, candidates,
                H=H, n_rollouts=n_rollouts, gamma=gamma,
                seed=int(rng.integers(0, 1 << 30)),
            )
            current_action = best_a
        _, rew, term, trunc, info = env.step(
            current_action.astype(np.float32)
        )
        total_reward += rew
        total_cost += info["cost"]
        if term or trunc:
            break
    return {"reward": total_reward, "J_C": total_cost,
            "final_N_U": float(env.N[0]), "final_N_E": float(env.N[1])}