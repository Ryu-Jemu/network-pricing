"""Integration tests for the journal-extension merge.

The journal codebase combines three previously separate branches into
one environment:
  - improvement-2: endogenous (load-dependent) QoS
  - improvement-1: CMDP cost signal via info["cost"]
  - improvement-9: cohort-aware state (N_cohort always allocated)
These tests cover the seams between them that no single branch's test
suite exercises.
"""
import numpy as np

from src.env.network_slicing_env import NetworkSlicingEnv
from src.train.config import get_env_config
from src.train.mpc_baseline import _simulate_rollout


def test_cost_emitted_on_both_qos_modes():
    for endo in (False, True):
        cfg = get_env_config(endogenous=endo)
        env = NetworkSlicingEnv(config=cfg)
        env.reset(seed=7)
        _, _, _, _, info = env.step(np.array([0.5, 0.5, 0.3, 0.25]))
        assert info["cost"] >= 0.0
        assert np.isclose(info["cost"], info["cost_U"] + info["cost_E"])


def test_eta_sla_changes_cost_not_reward():
    """eta_sla drives the CMDP cost only; reward/penalty use eta_tgt."""
    base = get_env_config(endogenous=True)
    loose = dict(base, eta_sla=[0.5, 0.5])
    r_base, c_base, r_loose, c_loose = [], [], [], []
    for cfg, rs, cs in ((base, r_base, c_base), (loose, r_loose, c_loose)):
        env = NetworkSlicingEnv(config=cfg)
        env.reset(seed=11)
        for _ in range(50):
            _, r, _, _, info = env.step(np.array([0.5, 0.5, 0.3, 0.25]))
            rs.append(r)
            cs.append(info["cost"])
    assert np.allclose(r_base, r_loose), "reward must not depend on eta_sla"
    assert sum(c_loose) < sum(c_base), "looser SLA must lower cost"


def test_mpc_state_override_reaches_cohort_state():
    """The MPC inner rollout must start from the overridden subscriber
    counts, not from N_init (the dynamics read N_cohort)."""
    cfg = get_env_config()  # N_init = [1000, 5000]
    small_state = (10.0, 20.0, 0.95, 0.90)
    # Run a 1-step rollout and inspect via a tiny shim: reuse the
    # function and verify returns scale with the overridden state.
    ret_small = _simulate_rollout(
        cfg, small_state, np.array([0.5, 0.5, 0.3, 0.25]),
        H=1, n_rollouts=3, gamma=0.99, seed=3,
    )
    big_state = (1000.0, 5000.0, 0.95, 0.90)
    ret_big = _simulate_rollout(
        cfg, big_state, np.array([0.5, 0.5, 0.3, 0.25]),
        H=1, n_rollouts=3, gamma=0.99, seed=3,
    )
    # Revenue scales with subscribers: the small-state return must be
    # far below the big-state return (it equals it if the override is
    # silently ignored).
    assert ret_small < 0.05 * ret_big, (ret_small, ret_big)


def test_endogenous_composes_with_cohort_aware():
    """Both extensions on simultaneously: cohort invariant holds and
    endogenous diagnostics are emitted."""
    cfg = get_env_config(extended=True, endogenous=True)
    env = NetworkSlicingEnv(config=cfg)
    env.reset(seed=5)
    for _ in range(30):
        _, _, _, _, info = env.step(env.action_space.sample())
        assert np.allclose(env.N_cohort.sum(axis=1), env.N)
        assert info["util_U"] > 0.0
        assert info["cost"] >= 0.0


def test_default_config_unchanged_vs_published_reference():
    """Default-mode 20-step reward trajectory must match the recorded
    reference produced by the pre-merge improvement-9 environment.

    Reference generated from commit ac8ee55 (branch
    improvement-9-baselines-7seeds) with the same seed/action sequence;
    the merged env claims bit-exact backward compatibility.
    """
    cfg = get_env_config(churn_multiplier=1)
    env = NetworkSlicingEnv(config=cfg)
    env.reset(seed=1000)
    action = np.array([0.5, 0.5, 0.3, 0.25], dtype=np.float32)
    rewards = []
    for _ in range(20):
        _, r, _, _, _ = env.step(action)
        rewards.append(r)
    ref = np.array(REFERENCE_REWARDS_SEED1000)
    assert np.allclose(rewards, ref, rtol=0, atol=1e-12), (
        np.max(np.abs(np.array(rewards) - ref))
    )


# 20-step reference trajectory (reference action, seed=1000, m=1),
# recorded from the pre-merge improvement-9 environment (ac8ee55).
REFERENCE_REWARDS_SEED1000 = [
    4.540347864483676,
    3.8104864607151088,
    4.43594166331918,
    4.588764641577142,
    4.3304794649059595,
    4.034031588402846,
    3.9757865050192343,
    4.073355776560826,
    4.335058602268972,
    4.480744707640355,
    4.0104802842495335,
    4.311582430323357,
    4.329666798892222,
    4.091111651076582,
    4.326109911892351,
    4.851581395863612,
    4.4105975801466615,
    4.221306505326713,
    4.3397669704013255,
    4.3423183589183045,
]
