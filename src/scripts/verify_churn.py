"""
Churn Rate Verification Script
================================
For each churn multiplier, runs 1 episode with reference prices
and reports observed monthly churn rate to verify gamma0 adjustment.
"""
import numpy as np

from src.env.network_slicing_env import NetworkSlicingEnv
from src.train.config import ENV_CONFIG, REFERENCE_ACTION

MULTIPLIERS = [1, 3, 5, 10]


def verify_churn(multiplier):
    """Run 1 episode with reference prices and measure actual churn."""
    import copy
    cfg = copy.deepcopy(ENV_CONFIG)
    offset = float(np.log(multiplier))
    cfg["gamma0"] = [
        ENV_CONFIG["gamma0"][0] + offset,
        ENV_CONFIG["gamma0"][1] + offset,
    ]

    env = NetworkSlicingEnv(config=cfg)
    obs, _ = env.reset(seed=42)
    action = np.array(REFERENCE_ACTION, dtype=np.float32)

    total_leave_U, total_leave_E = 0, 0

    for t in range(env.T):
        obs, r, term, trunc, info = env.step(action)
        total_leave_U += info["N_leave_U"]
        total_leave_E += info["N_leave_E"]

    # Observed monthly churn = 1 - (N_final / N_initial)
    # (simplified; actual churn involves arrivals too)
    final_N_U = info["N_U"]
    final_N_E = info["N_E"]

    # Per-step departure probability (from logit formula at reference prices)
    sigmoid = lambda x: 1.0 / (1.0 + np.exp(-x))
    F_tilde = np.array([1.0, 1.0])  # reference normalized
    p_tilde = np.array([1.0, 1.0])
    eta_prev = np.array([0.95, 0.90])  # expected QoS

    dep_logit = (
        np.array(cfg["gamma0"])
        + cfg["gamma_F"] * F_tilde
        + cfg["gamma_p"] * p_tilde
        - np.array(cfg["gamma_eta"]) * eta_prev
    )
    P_dep = sigmoid(dep_logit)
    theoretical_monthly = 1.0 - (1.0 - P_dep) ** 720

    return {
        "multiplier": multiplier,
        "gamma0": cfg["gamma0"],
        "P_dep_per_step": P_dep.tolist(),
        "theoretical_monthly_churn_U": float(theoretical_monthly[0]),
        "theoretical_monthly_churn_E": float(theoretical_monthly[1]),
        "total_departures_U": float(total_leave_U),
        "total_departures_E": float(total_leave_E),
        "final_N_U": float(final_N_U),
        "final_N_E": float(final_N_E),
    }


def main():
    print("=" * 70)
    print("Churn Rate Verification")
    print("=" * 70)
    print(f"\nBase gamma0: {ENV_CONFIG['gamma0']}")
    print(f"Reference action: {REFERENCE_ACTION}")
    print()

    print(f"{'Mult':>5} {'gamma0_U':>10} {'gamma0_E':>10} "
          f"{'P_dep_U':>12} {'P_dep_E':>12} "
          f"{'Monthly_U':>10} {'Monthly_E':>10} "
          f"{'Final_N_U':>10} {'Final_N_E':>10}")
    print("-" * 100)

    for mult in MULTIPLIERS:
        r = verify_churn(mult)
        print(f"{str(mult)+'x':>5} "
              f"{r['gamma0'][0]:>10.2f} {r['gamma0'][1]:>10.2f} "
              f"{r['P_dep_per_step'][0]:>12.2e} {r['P_dep_per_step'][1]:>12.2e} "
              f"{r['theoretical_monthly_churn_U']*100:>9.2f}% "
              f"{r['theoretical_monthly_churn_E']*100:>9.2f}% "
              f"{r['final_N_U']:>10.0f} {r['final_N_E']:>10.0f}")

    print()
    print("Churn ratio verification (should approximate multiplier):")
    for mult in MULTIPLIERS:
        r = verify_churn(mult)
        ratio = r["theoretical_monthly_churn_E"] / verify_churn(1)["theoretical_monthly_churn_E"]
        print(f"  {mult}x: actual ratio = {ratio:.2f}x")


if __name__ == "__main__":
    main()
