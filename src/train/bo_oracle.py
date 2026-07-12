"""Bayesian-Optimisation Static Oracle (improvement-5).

Replaces the 600-point grid (`run_multi_seed.run_static_oracle`) with
GP-UCB Bayesian optimisation in the same 4-D action box [0,1]^4. Uses
the SAME total evaluation budget (n_episodes per probe) but adapts
sample placement.

Reference:
- Srinivas, Krause, Kakade, Seeger. "Gaussian Process Optimization in
  the Bandit Setting: No Regret and Experimental Design." ICML 2010.
- Frazier. "A Tutorial on Bayesian Optimization." 2018.
  arXiv:1807.02811.
"""
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    Matern, ConstantKernel as C, WhiteKernel,
)

from src.env.network_slicing_env import NetworkSlicingEnv


def _evaluate_constant_action(
    env_config, action, n_episodes=5, seed0=42
):
    """Run a fixed action policy for n_episodes; return mean reward."""
    rewards = []
    for ep in range(n_episodes):
        env = NetworkSlicingEnv(config=env_config)
        env.reset(seed=seed0 + ep)
        r = 0.0
        for _ in range(env.T):
            _, rew, _, _, _ = env.step(action.astype(np.float32))
            r += rew
        rewards.append(r)
    return float(np.mean(rewards))


def bo_static_oracle(
    env_config,
    n_init=8,
    n_iter=22,
    n_episodes=5,
    bounds=None,
    kappa=2.5,
    seed=42,
    verbose=False,
    include_corners=False,
):
    """GP-UCB Bayesian optimisation over constant 4-D action.

    Total objective evaluations = n_init + n_iter (default 30).
    Original grid had 600 evaluations; BO uses 5% of that budget but
    is expected to find as-good-or-better action via adaptive
    placement.

    Args:
        n_init: Number of random initial points.
        n_iter: Number of BO acquisition iterations.
        n_episodes: Episodes averaged per evaluation.
        bounds: list of (lo, hi) per dim; default [0,1]^4.
        kappa: UCB exploration weight.
        include_corners: seed the initial design with all 16 corners of
            the action box. Uniform sampling never hits exact corners,
            yet trained policies frequently saturate there (e.g.
            [1,0,0,1]); seeding them makes the static oracle a fair —
            stronger — comparator.
    """
    bounds = bounds or [(0.0, 1.0)] * 4
    bounds_arr = np.array(bounds)
    rng = np.random.default_rng(seed)

    X = []
    y = []

    if include_corners:
        corners = np.array(
            np.meshgrid(*[[lo, hi] for lo, hi in bounds])
        ).reshape(4, -1).T
        for a in corners:
            r = _evaluate_constant_action(
                env_config, a, n_episodes=n_episodes, seed0=seed,
            )
            X.append(a)
            y.append(r)
            if verbose:
                print(f"  corner action={a.round(3)} reward={r:,.0f}")

    # Initial random sampling
    for _ in range(n_init):
        a = rng.uniform(bounds_arr[:, 0], bounds_arr[:, 1])
        r = _evaluate_constant_action(
            env_config, a, n_episodes=n_episodes, seed0=seed,
        )
        X.append(a)
        y.append(r)
        if verbose:
            print(f"  init action={a.round(3)} reward={r:,.0f}")

    # GP-UCB iterations
    kernel = (
        C(1.0, (1e-3, 1e3))
        * Matern(length_scale=0.3, nu=2.5,
                 length_scale_bounds=(1e-2, 10.0))
        + WhiteKernel(noise_level=1e-2,
                      noise_level_bounds=(1e-6, 10.0))
    )

    for it in range(n_iter):
        X_arr = np.array(X)
        y_arr = np.array(y)
        # Standardise y for GP stability
        y_mean = y_arr.mean()
        y_std = max(y_arr.std(), 1e-3)
        y_norm = (y_arr - y_mean) / y_std

        gp = GaussianProcessRegressor(
            kernel=kernel,
            normalize_y=False,
            n_restarts_optimizer=2,
            random_state=int(rng.integers(0, 1 << 31)),
        )
        gp.fit(X_arr, y_norm)

        # Acquisition: UCB. Sample a candidate cloud and pick the max.
        n_cand = 2000
        cand = rng.uniform(
            bounds_arr[:, 0], bounds_arr[:, 1],
            size=(n_cand, 4),
        )
        mu, sigma = gp.predict(cand, return_std=True)
        ucb = mu + kappa * sigma
        best_idx = int(np.argmax(ucb))
        a_next = cand[best_idx]
        r_next = _evaluate_constant_action(
            env_config, a_next, n_episodes=n_episodes, seed0=seed,
        )
        X.append(a_next)
        y.append(r_next)
        if verbose:
            print(
                f"  iter {it+1}/{n_iter}: "
                f"action={a_next.round(3)} reward={r_next:,.0f} "
                f"best={max(y):,.0f}"
            )

    best_idx = int(np.argmax(y))
    return {
        "best_action": np.array(X[best_idx]).tolist(),
        "best_reward": float(y[best_idx]),
        "n_evaluations": len(X),
        "history": [
            {"action": np.asarray(a).tolist(), "reward": float(r)}
            for a, r in zip(X, y)
        ],
    }
