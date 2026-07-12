"""Unit tests for the heuristic baseline policies."""
import numpy as np

from src.train.config import get_env_config
from src.train.heuristic_baselines import (
    DEFAULT_PEAK_HOURS, HIGH_ACTION, LOW_ACTION, OFFPEAK_ACTION, PEAK_ACTION,
    evaluate_policy_factory, make_load_threshold_policy,
    make_peak_offpeak_policy,
)


def test_load_threshold_switches_at_boundary():
    policy = make_load_threshold_policy(threshold=0.5)
    # obs = [N_U_norm, N_E_norm, eta_U, eta_E]
    np.testing.assert_array_equal(policy([1.0, 0.2, 0.9, 0.9]), LOW_ACTION)
    np.testing.assert_array_equal(policy([1.0, 0.6, 0.9, 0.9]), HIGH_ACTION)
    # boundary: < threshold → low; ≥ threshold → high
    np.testing.assert_array_equal(policy([1.0, 0.5, 0.9, 0.9]), HIGH_ACTION)
    np.testing.assert_array_equal(policy([1.0, 0.499, 0.9, 0.9]), LOW_ACTION)


def test_load_threshold_returns_float32_shape_4():
    policy = make_load_threshold_policy()
    a = policy([1.0, 0.3, 0.95, 0.90])
    assert a.shape == (4,)
    assert a.dtype == np.float32


def test_peak_offpeak_cycles_correctly():
    policy = make_peak_offpeak_policy()
    actions = [policy([1.0, 1.0, 0.95, 0.9]) for _ in range(48)]
    # First 24 hours
    for h in range(24):
        expected = PEAK_ACTION if h in DEFAULT_PEAK_HOURS else OFFPEAK_ACTION
        np.testing.assert_array_equal(actions[h], expected, err_msg=f"h={h}")
    # Wraps modulo 24
    for h in range(24):
        expected = PEAK_ACTION if h in DEFAULT_PEAK_HOURS else OFFPEAK_ACTION
        np.testing.assert_array_equal(actions[24 + h], expected,
                                       err_msg=f"second cycle h={h}")


def test_peak_offpeak_fresh_per_factory():
    """Each call to the factory must produce a fresh t-counter."""
    p1 = make_peak_offpeak_policy()
    p2 = make_peak_offpeak_policy()
    a1 = [p1([1, 1, 0.9, 0.9]) for _ in range(10)]
    a2 = [p2([1, 1, 0.9, 0.9]) for _ in range(10)]
    # Both should produce identical sequences starting from t=0
    for x, y in zip(a1, a2):
        np.testing.assert_array_equal(x, y)


def test_peak_action_in_unit_box():
    p = make_peak_offpeak_policy()
    a = p([1, 1, 0.9, 0.9])
    assert (a >= 0).all() and (a <= 1).all()
    p = make_load_threshold_policy()
    for n in (0.1, 0.5, 1.0, 2.0):
        a = p([1, n, 0.9, 0.9])
        assert (a >= 0).all() and (a <= 1).all()


def test_evaluate_policy_factory_runs_full_episode():
    """Smoke test: evaluator completes one episode at m=1, returns sensible
    aggregate metrics. n_episodes=2 for speed."""
    cfg = get_env_config(extended=False, churn_multiplier=1)
    out = evaluate_policy_factory(
        cfg,
        policy_factory=lambda: make_load_threshold_policy(),
        n_episodes=2, base_seed=42,
    )
    assert "mean_reward" in out
    assert out["n_eval_episodes"] == 2
    assert len(out["per_episode_rewards"]) == 2
    assert len(out["trajectory_N_E_mean"]) == 720
    # Initial N_E ≈ 5000 at t=0
    assert 4500 < out["trajectory_N_E_mean"][0] < 5500


def test_peak_offpeak_state_advances_in_eval():
    """When evaluator builds a fresh policy per episode, the peak schedule
    starts at t=0 each episode (verifiable via revenue trajectory)."""
    cfg = get_env_config(extended=False, churn_multiplier=1)
    out = evaluate_policy_factory(
        cfg,
        policy_factory=lambda: make_peak_offpeak_policy(),
        n_episodes=2, base_seed=42,
    )
    # Two episodes give two complete trajectories of length 720
    assert len(out["trajectory_N_E_mean"]) == 720
    # First step trajectory_N_E should be close to N_init_E (5000)
    assert 4500 < out["trajectory_N_E_mean"][0] < 5100


def test_default_action_vectors_well_formed():
    for vec in (LOW_ACTION, HIGH_ACTION, PEAK_ACTION, OFFPEAK_ACTION):
        assert vec.shape == (4,)
        assert (vec >= 0).all() and (vec <= 1).all()
        assert vec.dtype == np.float32
