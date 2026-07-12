"""
Tests for CMDP/Lagrangian extensions (improvement-1 branch).

Covers:
  - cost signal nonneg, has correct shape, matches penalty/w*N decomposition
  - LagrangianCostWrapper subtracts lam*c (and lam=0 -> identity)
  - Dual ascent updates lam in expected direction
  - cost agreement: penalty == sum_s w_s * cost_s for s in {U,E}
"""
import numpy as np

from src.env.network_slicing_env import NetworkSlicingEnv
from src.train.config import ENV_CONFIG
from src.train.train_ppo_lagrangian import (
    LagrangianCostWrapper, DualAscentCallback,
)


def test_cost_signal_nonneg_and_decomposition():
    env = NetworkSlicingEnv()
    env.reset(seed=42)
    a = np.array([0.5, 0.5, 0.3, 0.25], dtype=np.float32)
    for _ in range(50):
        _, _, _, _, info = env.step(a)
        assert info["cost_U"] >= 0.0
        assert info["cost_E"] >= 0.0
        assert info["cost"] == info["cost_U"] + info["cost_E"]
        # penalty = w_U*cost_U + w_E*cost_E (definition match)
        expected = (
            ENV_CONFIG["w"][0] * info["cost_U"]
            + ENV_CONFIG["w"][1] * info["cost_E"]
        )
        assert abs(info["penalty"] - expected) < 1e-6
    print("  [PASS] cost signal: nonneg + penalty decomposition")


def test_lambda_zero_is_identity():
    inner = NetworkSlicingEnv()
    lam_state = [0.0]
    wrapped = LagrangianCostWrapper(inner, lam_state, cost_scale=1e-5)
    obs1, _ = wrapped.reset(seed=7)
    obs2, _ = NetworkSlicingEnv().reset(seed=7)
    np.testing.assert_array_almost_equal(obs1, obs2)
    a = np.array([0.5, 0.5, 0.3, 0.25], dtype=np.float32)
    _, r_w, _, _, info = wrapped.step(a)
    inner2 = NetworkSlicingEnv()
    inner2.reset(seed=7)
    _, r_b, _, _, info_b = inner2.step(a)
    # cost is read from info; with lam=0 reward must equal base.
    assert abs(r_w - r_b) < 1e-6, (r_w, r_b)
    assert "cost" in info
    print("  [PASS] lam=0 wrapper is identity")


def test_lambda_positive_subtracts_cost():
    inner = NetworkSlicingEnv()
    lam_state = [10.0]  # very large lam to be unambiguous
    wrapped = LagrangianCostWrapper(inner, lam_state, cost_scale=1e-5)
    wrapped.reset(seed=7)
    a = np.array([0.5, 0.5, 0.3, 0.25], dtype=np.float32)
    _, r_w, _, _, info = wrapped.step(a)
    inner2 = NetworkSlicingEnv()
    inner2.reset(seed=7)
    _, r_b, _, _, _ = inner2.step(a)
    # r_w should equal r_b - lam * cost_scaled
    expected = r_b - lam_state[0] * info["cost_scaled"]
    assert abs(r_w - expected) < 1e-6
    print("  [PASS] lam>0 wrapper subtracts lam*cost")


def test_dual_ascent_update_direction():
    lam_state = [1.0]
    cb = DualAscentCallback(
        lam_state=lam_state, cost_limit=100.0, lr_lam=0.1,
    )
    # Simulate violation: J_C = 200 > limit 100 -> lam should grow
    cb._rollout_costs = [200.0, 200.0]
    cb._on_rollout_end()
    assert lam_state[0] > 1.0, lam_state
    # Simulate slack: J_C = 50 < limit 100 -> lam should shrink
    cb._rollout_costs = [50.0, 50.0]
    cb._on_rollout_end()
    # Note: still positive (max with 0), so just check decrease
    assert cb.lam_history[-1] < cb.lam_history[-2]
    print("  [PASS] dual ascent direction correct")


def test_dual_ascent_lam_nonneg():
    lam_state = [0.0]
    cb = DualAscentCallback(
        lam_state=lam_state, cost_limit=1000.0, lr_lam=10.0,
    )
    cb._rollout_costs = [10.0]  # massive slack
    cb._on_rollout_end()
    assert lam_state[0] == 0.0  # projected to 0
    print("  [PASS] dual ascent projects lam to >= 0")


if __name__ == "__main__":
    test_cost_signal_nonneg_and_decomposition()
    test_lambda_zero_is_identity()
    test_lambda_positive_subtracts_cost()
    test_dual_ascent_update_direction()
    test_dual_ascent_lam_nonneg()
    print("ALL CMDP TESTS PASSED")


def test_pid_lambda_responds_and_decays():
    """PID controller: lambda rises under violation, falls when the
    cost drops below the limit (no probe env -> stochastic signal)."""
    from src.train.train_ppo_lagrangian import PIDLagrangianCallback
    lam = [0.0]
    cb = PIDLagrangianCallback(
        lam_state=lam, cost_limit=1.0, kp=20.0, ki=5.0, kd=20.0,
        lam_max=2000.0, probe_env_config=None,
    )
    # three violating rollouts (J_C = 3.0 vs limit 1.0)
    for _ in range(3):
        cb._rollout_costs = [3.0, 3.0]
        cb._on_rollout_end()
    lam_high = lam[0]
    assert lam_high > 50.0, lam_high
    # five satisfied rollouts (J_C = 0.2)
    for _ in range(5):
        cb._rollout_costs = [0.2, 0.2]
        cb._on_rollout_end()
    assert lam[0] < lam_high, (lam[0], lam_high)
    assert lam[0] >= 0.0
    assert cb.lam_history[-1] == lam[0]
