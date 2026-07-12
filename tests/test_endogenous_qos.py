"""Tests for endogenous-QoS variant (improvement-2 branch).

Verifies:
  - Default env (qos_endogenous=False) is unchanged (backward compat).
  - Endogenous env: load and util are reported, eta is bounded.
  - Endogenous env: higher subscriber count -> higher util -> lower eta.
  - At reference action with default capacities, util ~ 0.70.
"""
import numpy as np

from src.env.network_slicing_env import NetworkSlicingEnv
from src.train.config import ENV_CONFIG_ENDO


REF = np.array([0.5, 0.5, 0.3, 0.25], dtype=np.float32)


def test_default_env_is_exogenous():
    env = NetworkSlicingEnv()
    env.reset(seed=42)
    _, _, _, _, info = env.step(REF)
    # qos_endogenous defaults to False; util fields exist but are 0.
    assert "util_U" in info and info["util_U"] == 0.0
    assert "util_E" in info and info["util_E"] == 0.0
    print("  [PASS] default env remains exogenous (backward compat)")


def test_endogenous_load_reported():
    env = NetworkSlicingEnv(config=ENV_CONFIG_ENDO)
    env.reset(seed=42)
    _, _, _, _, info = env.step(REF)
    # E[load_U] ~ N_U * E[q_U] = 1000 * 3.49 ~ 3490
    # E[load_E] ~ N_E * E[q_E] = 5000 * 29.96 ~ 149800
    assert 2500 < info["load_U"] < 5000, info["load_U"]
    assert 100000 < info["load_E"] < 200000, info["load_E"]
    print(
        f"  [PASS] load reported: U={info['load_U']:.0f}, "
        f"E={info['load_E']:.0f}"
    )


def test_endogenous_util_matches_journal_calibration():
    """Average util across 100 steps at the reference policy matches
    the journal calibration (PARAMETER_JUSTIFICATION.md):
      URLLC: util ~ 0.70 < rho*_U = 0.70  (healthy at base load)
      eMBB:  util ~ 0.85 > onset 0.767    (SLA-risk at full base)
    """
    env = NetworkSlicingEnv(config=ENV_CONFIG_ENDO)
    env.reset(seed=42)
    util_U_list, util_E_list = [], []
    for _ in range(100):
        _, _, _, _, info = env.step(REF)
        util_U_list.append(info["util_U"])
        util_E_list.append(info["util_E"])
    mean_util_U = float(np.mean(util_U_list))
    mean_util_E = float(np.mean(util_E_list))
    # Allow drift since N moves slightly over 100 steps.
    assert 0.60 < mean_util_U < 0.80, mean_util_U
    assert 0.78 < mean_util_E < 0.92, mean_util_E
    print(
        f"  [PASS] journal calibration: "
        f"U={mean_util_U:.3f} (healthy), E={mean_util_E:.3f} (SLA-risk)"
    )


def test_endogenous_eta_bounded():
    env = NetworkSlicingEnv(config=ENV_CONFIG_ENDO)
    env.reset(seed=42)
    for _ in range(50):
        _, _, _, _, info = env.step(REF)
        assert 0.0 <= info["eta_U"] <= 1.0
        assert 0.0 <= info["eta_E"] <= 1.0
    print("  [PASS] eta in [0, 1] over 50 steps")


def test_endogenous_load_eta_anticorrelation():
    """Higher subscriber count should reduce eta on average."""
    cfg_low = {**ENV_CONFIG_ENDO, "N_init": [500.0, 2500.0]}
    cfg_high = {**ENV_CONFIG_ENDO, "N_init": [2000.0, 10000.0]}
    n_steps = 100

    def avg_eta(cfg):
        env = NetworkSlicingEnv(config=cfg)
        env.reset(seed=7)
        eta_U_list, eta_E_list = [], []
        for _ in range(n_steps):
            _, _, _, _, info = env.step(REF)
            eta_U_list.append(info["eta_U"])
            eta_E_list.append(info["eta_E"])
        return float(np.mean(eta_U_list)), float(np.mean(eta_E_list))

    eU_lo, eE_lo = avg_eta(cfg_low)
    eU_hi, eE_hi = avg_eta(cfg_high)
    # High-N must give LOWER eta on average than low-N.
    assert eU_hi < eU_lo, f"URLLC eta did not drop with load: {eU_hi} vs {eU_lo}"
    assert eE_hi < eE_lo, f"eMBB eta did not drop with load: {eE_hi} vs {eE_lo}"
    print(
        f"  [PASS] feedback loop: low-N eta=({eU_lo:.3f},{eE_lo:.3f}) "
        f"vs high-N=({eU_hi:.3f},{eE_hi:.3f})"
    )


if __name__ == "__main__":
    test_default_env_is_exogenous()
    test_endogenous_load_reported()
    test_endogenous_util_matches_journal_calibration()
    test_endogenous_eta_bounded()
    test_endogenous_load_eta_anticorrelation()
    print("ALL ENDOGENOUS-QOS TESTS PASSED")
