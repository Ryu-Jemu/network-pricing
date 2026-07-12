"""Subscriber -> traffic load -> SLA structure analysis (Phase B).

Investigates, per user direction, the mechanism "more subscribers =>
more traffic => load => lower SLA" under the endogenous QoS model:

    util_s  = load_s / capacity_s,   load_s = sum of LogNormal usage
    eta_s   = clip(1 - alpha_s * max(0, util_s - rho_star_s) + jitter)

Two questions decide the CMDP calibration:
  Q1. STATIC MAP: at which subscriber level N_s does slice s start to
      violate eta_SLA? Where does the initial base sit relative to it?
  Q2. REACHABILITY: within one episode (T=720), which (N, J_C) region
      can pricing actually reach, per churn multiplier m? The CMDP is
      meaningful only if J_C <= d is reachable (feasible) AND violated
      by the revenue-best policy (binding).

Compares calibrations:
  current : capacity=[5000, 215000], rho_star=[0.5, 0.5]   (branch 2)
  recalib : capacity=[5000, 176000], rho_star=[0.70, 0.50] (proposal)

Output: results/journal/load_sla_analysis.json + console report.
"""
import json
from pathlib import Path

import numpy as np

from src.env.network_slicing_env import NetworkSlicingEnv
from src.train.config import get_env_config

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results" / "journal"

ETA_SLA = [0.995, 0.90]
E_Q = [3.4903, 29.9641]  # E[q] per slice (LogNormal, verified in tests)

CALIBRATIONS = {
    "current": {"capacity": [5000.0, 215000.0], "rho_star": [0.50, 0.50]},
    "recalib": {"capacity": [5000.0, 176000.0], "rho_star": [0.70, 0.50]},
}
ALPHA = [0.45, 0.30]
DELTA = [0.01, 0.02]


def expected_step_cost_rate(n, slice_idx, calib):
    """E[max(0, eta_SLA - eta)] * n for subscriber count n (closed form
    over the jitter; uses expected load n*E[q])."""
    s = slice_idx
    util = n * E_Q[s] / calib["capacity"][s]
    eta_det = 1.0 - ALPHA[s] * max(0.0, util - calib["rho_star"][s])
    lo, hi = eta_det - DELTA[s], eta_det + DELTA[s]
    thr = ETA_SLA[s]
    # eta ~ U(lo, hi) clipped to [0,1]; expected shortfall vs thr
    grid = np.clip(np.linspace(lo, hi, 2001), 0.0, 1.0)
    return float(np.mean(np.maximum(0.0, thr - grid)) * n)


def critical_n(slice_idx, calib):
    """Smallest n at which expected step cost rate becomes >0.5% of n
    (i.e., meaningful violation)."""
    s = slice_idx
    for n in range(0, 20001, 10):
        if expected_step_cost_rate(n, s, calib) > 0.005 * max(n, 1):
            return n
    return None


def episode_jc(env_cfg, action, n_episodes=5, base_seed=2000):
    env = NetworkSlicingEnv(config=env_cfg)
    jcs, rewards, finals = [], [], []
    for i in range(n_episodes):
        obs, _ = env.reset(seed=base_seed + i)
        a = np.asarray(action, dtype=np.float32)
        jc = tot = 0.0
        for _ in range(env.T):
            obs, r, term, trunc, info = env.step(a)
            jc += (
                max(0.0, ETA_SLA[0] - info["eta_U"]) * info["N_U"]
                + max(0.0, ETA_SLA[1] - info["eta_E"]) * info["N_E"]
            )
            tot += r
            if term or trunc:
                break
        jcs.append(jc)
        rewards.append(tot)
        finals.append((info["N_U"], info["N_E"]))
    return {
        "J_C": float(np.mean(jcs)),
        "reward": float(np.mean(rewards)),
        "final_N_U": float(np.mean([f[0] for f in finals])),
        "final_N_E": float(np.mean([f[1] for f in finals])),
    }


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    report = {"eta_sla": ETA_SLA, "calibrations": {}}

    # ---- Q1: static N -> cost-rate map ----
    print("=" * 70)
    print("Q1. Static map: subscriber level -> expected SLA cost rate/step")
    print("=" * 70)
    for name, calib in CALIBRATIONS.items():
        rows = {}
        for s, label, n_grid in (
            (0, "URLLC", [500, 716, 764, 850, 1000, 1100, 1500]),
            (1, "eMBB", [3000, 4000, 4400, 4700, 5000, 5500, 6000]),
        ):
            rows[label] = {
                str(n): round(expected_step_cost_rate(n, s, calib), 2)
                for n in n_grid
            }
            crit = critical_n(s, calib)
            rows[f"{label}_critical_N"] = crit
            print(f"[{name}] {label}: cost-rate/step by N = "
                  f"{rows[label]}  | violation onset N* = {crit}")
        report["calibrations"][name] = {"static_map": rows}

    # ---- Q2: within-episode reachability per m ----
    print("\n" + "=" * 70)
    print("Q2. Reachability: episode J_C for key policies (5 eps each)")
    print("=" * 70)
    policies = {
        "reference": [0.5, 0.5, 0.3, 0.25],
        "max_price": [1.0, 1.0, 1.0, 1.0],
        "shed_eMBB": [0.5, 0.5, 0.8, 0.6],   # raise only eMBB prices
        "grow_eMBB": [0.5, 0.5, 0.1, 0.1],   # cheap eMBB (load-max)
    }
    for name, calib in CALIBRATIONS.items():
        report["calibrations"][name]["episodes"] = {}
        for m in [1, 3]:
            cfg = get_env_config(endogenous=True, churn_multiplier=m)
            cfg["capacity"] = calib["capacity"]
            cfg["rho_star"] = calib["rho_star"]
            cfg["eta_sla"] = ETA_SLA
            res = {pn: episode_jc(cfg, pa) for pn, pa in policies.items()}
            report["calibrations"][name]["episodes"][str(m)] = res
            jcs = {pn: r["J_C"] for pn, r in res.items()}
            spread = max(jcs.values()) - min(jcs.values())
            rel = spread / max(max(jcs.values()), 1.0)
            print(f"[{name}] m={m}: " + "  ".join(
                f"{pn}: J_C={r['J_C']:9,.0f} R={r['reward']:7,.0f} "
                f"N_E={r['final_N_E']:5,.0f}"
                for pn, r in res.items()
            ))
            print(f"          J_C spread = {spread:,.0f} "
                  f"({100 * rel:.0f}% of max) -> "
                  f"{'CONTROLLABLE' if rel > 0.5 else 'WEAK'}")

    path = RESULTS_DIR / "load_sla_analysis.json"
    with open(path, "w") as f:
        json.dump(report, f, indent=1)
    print(f"\nsaved -> {path}")


if __name__ == "__main__":
    main()
