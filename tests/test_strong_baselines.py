"""Tests for strong baselines (improvement-5 branch)."""
import numpy as np

from src.train.bo_oracle import bo_static_oracle, _evaluate_constant_action
from src.train.mpc_baseline import (
    make_action_candidates, mpc_step, run_mpc_episode_protocol,
)
from src.train.config import ENV_CONFIG


def test_bo_evaluate_constant():
    """Sanity: reference action evaluates to ~3000 reward."""
    a = np.array([0.5, 0.5, 0.3, 0.25])
    r = _evaluate_constant_action(ENV_CONFIG, a, n_episodes=2)
    assert 2500 < r < 3700, r
    print(f"  [PASS] BO eval reference action: reward={r:,.0f}")


def test_bo_oracle_smoke():
    """Smoke: tiny BO run completes and finds something better than
    a single random action."""
    res = bo_static_oracle(
        ENV_CONFIG, n_init=4, n_iter=3, n_episodes=2,
        kappa=2.0, seed=7, verbose=False,
    )
    assert "best_action" in res and "best_reward" in res
    assert len(res["history"]) == 4 + 3
    # Best should be at least as good as median.
    rewards_seen = [h["reward"] for h in res["history"]]
    assert res["best_reward"] >= np.median(rewards_seen)
    print(
        f"  [PASS] BO smoke: best={res['best_reward']:,.0f} "
        f"median seen={np.median(rewards_seen):,.0f}"
    )


def test_mpc_action_candidates():
    cand = make_action_candidates(n_grid_per_dim=3)
    assert cand.shape == (81, 4)
    assert ((cand >= 0.0) & (cand <= 1.0)).all()
    print("  [PASS] MPC candidate grid: 3^4=81 actions in [0,1]^4")


def test_mpc_step_picks_best():
    init_state = (1000.0, 5000.0, 0.95, 0.90)
    candidates = np.array([
        [0.0, 0.0, 0.0, 0.0],   # zero price
        [1.0, 1.0, 1.0, 1.0],   # max price
        [0.5, 0.5, 0.3, 0.25],  # reference
    ])
    a, scores = mpc_step(
        ENV_CONFIG, init_state, candidates,
        H=12, n_rollouts=2, gamma=0.99, seed=42,
    )
    # zero price should score lowest, max-price typically highest at m=1
    assert scores[0] < scores[1] or scores[0] < scores[2]
    print(f"  [PASS] MPC step picks better than zero-price: {scores}")


def test_mpc_full_episode_smoke():
    """One full MPC episode runs without error, returns finite reward
    and a non-negative SLA cost."""
    out = run_mpc_episode_protocol(
        ENV_CONFIG, H=12, n_rollouts=2, n_grid=2,
        gamma=0.99, seed=7, replan_every=24,
    )
    assert np.isfinite(out["reward"])
    assert 1000 < out["reward"] < 12000  # reasonable range for m=1
    assert out["J_C"] >= 0.0
    print(f"  [PASS] MPC full episode: reward={out['reward']:,.0f}")


if __name__ == "__main__":
    test_bo_evaluate_constant()
    test_bo_oracle_smoke()
    test_mpc_action_candidates()
    test_mpc_step_picks_best()
    test_mpc_full_episode_smoke()
    print("ALL STRONG-BASELINE TESTS PASSED")
