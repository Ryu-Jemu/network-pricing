"""Phase-B feasibility probe for the SLA-constrained journal MDP.

Before any Lagrangian training, this probe establishes — per churn
multiplier m — that the CMDP constraint J_C <= d is *feasible* (some
constant policy satisfies it) and *binding* (the revenue-maximising
static policy violates it). It also quantifies the endogeneity loop
(price -> subscribers -> load -> QoS) and the reachability of SLA
thresholds under measurement jitter.

Cost is computed offline from per-step (eta, N) for BOTH threshold
candidates so the eta_sla decision can be made from one probe run:
    legacy   eta_sla = [0.99999, 0.90]  (improvement-1 behaviour)
    proposed eta_sla = [0.995,   0.90]  (jitter-aware operational SLA)

Output: results/journal/feasibility_probe.json

Usage:
    python -m src.scripts.run_feasibility_probe [--quick]
"""
import argparse
import itertools
import json
import time
from pathlib import Path

import numpy as np

from src.env.network_slicing_env import NetworkSlicingEnv
from src.train.config import get_env_config, REFERENCE_ACTION

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results" / "journal"

MULTIPLIERS = [1, 3, 5, 10]
THRESHOLDS = {
    "legacy": [0.99999, 0.90],
    "proposed": [0.995, 0.90],
}
SPECIAL_ACTIONS = {
    "zero_price": [0.0, 0.0, 0.0, 0.0],
    "reference": list(REFERENCE_ACTION),
    "max_price": [1.0, 1.0, 1.0, 1.0],
    "ppo_corner": [1.0, 0.0, 0.0, 1.0],  # saturated PPO policy (m>=5)
}
GRID_LEVELS = [0.1, 0.5, 0.9]
# d calibration rule (pre-registered): d = floor + 0.3 * (J_C of the
# reward-best static action - floor), under the chosen threshold.
D_RULE_FRACTION = 0.3


def run_episode(env, action, seed):
    """Run one constant-action episode; return per-episode aggregates."""
    obs, _ = env.reset(seed=seed)
    a = np.asarray(action, dtype=np.float32)
    tot_r = 0.0
    costs = {k: 0.0 for k in THRESHOLDS}
    utils_, etas_U, etas_E = [], [], []
    n_u = n_e = 0.0
    for _ in range(env.T):
        obs, r, term, trunc, info = env.step(a)
        tot_r += r
        for k, thr in THRESHOLDS.items():
            costs[k] += (
                max(0.0, thr[0] - info["eta_U"]) * info["N_U"]
                + max(0.0, thr[1] - info["eta_E"]) * info["N_E"]
            )
        utils_.append((info["util_U"], info["util_E"]))
        etas_U.append(info["eta_U"])
        etas_E.append(info["eta_E"])
        n_u, n_e = info["N_U"], info["N_E"]
        if term or trunc:
            break
    utils_ = np.asarray(utils_)
    return {
        "reward": tot_r,
        "cost": costs,
        "mean_util_U": float(utils_[:, 0].mean()),
        "mean_util_E": float(utils_[:, 1].mean()),
        "mean_eta_U": float(np.mean(etas_U)),
        "mean_eta_E": float(np.mean(etas_E)),
        "sla_ok_frac_U_proposed": float(
            np.mean(np.asarray(etas_U) >= THRESHOLDS["proposed"][0])
        ),
        "final_N_U": n_u,
        "final_N_E": n_e,
        # within-episode endogeneity check material
        "corr_utilE_etaE": float(
            np.corrcoef(utils_[:, 1], etas_E)[0, 1]
        ) if utils_[:, 1].std() > 1e-12 and np.std(etas_E) > 1e-12 else 0.0,
    }


def eval_action(env, action, n_episodes, base_seed):
    eps = [run_episode(env, action, base_seed + i) for i in range(n_episodes)]
    out = {
        "action": list(map(float, action)),
        "n_episodes": n_episodes,
        "mean_reward": float(np.mean([e["reward"] for e in eps])),
        "mean_util_E": float(np.mean([e["mean_util_E"] for e in eps])),
        "mean_eta_U": float(np.mean([e["mean_eta_U"] for e in eps])),
        "mean_eta_E": float(np.mean([e["mean_eta_E"] for e in eps])),
        "sla_ok_frac_U_proposed": float(
            np.mean([e["sla_ok_frac_U_proposed"] for e in eps])
        ),
        "mean_final_N_E": float(np.mean([e["final_N_E"] for e in eps])),
        "corr_utilE_etaE": float(np.mean([e["corr_utilE_etaE"] for e in eps])),
    }
    for k in THRESHOLDS:
        out[f"J_C_{k}"] = float(np.mean([e["cost"][k] for e in eps]))
    return out


def probe_multiplier(m, quick=False):
    cfg = get_env_config(endogenous=True, churn_multiplier=m)
    env = NetworkSlicingEnv(config=cfg)
    n_grid_eps = 1 if quick else 2
    n_full_eps = 5 if quick else 20
    # NOTE: probe seeds (2000+) are disjoint from the final-eval
    # protocol seeds (1000-1019) by design.
    base_seed = 2000

    print(f"[m={m}] grid pass ({len(GRID_LEVELS)**4} actions x "
          f"{n_grid_eps} eps)...", flush=True)
    grid_rows = []
    for combo in itertools.product(GRID_LEVELS, repeat=4):
        grid_rows.append(eval_action(env, list(combo), n_grid_eps, base_seed))

    # Confirmation pass: special actions + grid extremes
    by_cost = sorted(grid_rows, key=lambda r: r["J_C_proposed"])[:3]
    by_reward = sorted(
        grid_rows, key=lambda r: -r["mean_reward"]
    )[:3]
    confirm_actions = {**SPECIAL_ACTIONS}
    for i, r in enumerate(by_cost):
        confirm_actions[f"grid_lowcost_{i}"] = r["action"]
    for i, r in enumerate(by_reward):
        confirm_actions[f"grid_highreward_{i}"] = r["action"]

    print(f"[m={m}] confirmation pass ({len(confirm_actions)} actions x "
          f"{n_full_eps} eps)...", flush=True)
    confirmed = {
        name: eval_action(env, a, n_full_eps, base_seed)
        for name, a in confirm_actions.items()
    }

    # d calibration per threshold candidate
    d_calib = {}
    for k in THRESHOLDS:
        floor_name, floor_row = min(
            confirmed.items(), key=lambda kv: kv[1][f"J_C_{k}"]
        )
        best_name, best_row = max(
            confirmed.items(), key=lambda kv: kv[1]["mean_reward"]
        )
        floor = floor_row[f"J_C_{k}"]
        j_best = best_row[f"J_C_{k}"]
        d = floor + D_RULE_FRACTION * max(0.0, j_best - floor)
        d_calib[k] = {
            "floor": floor,
            "floor_policy": floor_name,
            "J_C_of_reward_best_static": j_best,
            "reward_best_policy": best_name,
            "d_rule_0.3": d,
            "feasible_with_margin": bool(floor <= 0.8 * d) if d > 0 else False,
            "binding_on_reward_best": bool(j_best > d),
        }

    # Cross-action endogeneity: mean price level vs mean util_E
    price_levels = [float(np.mean(r["action"])) for r in grid_rows]
    utils_E = [r["mean_util_E"] for r in grid_rows]
    etas_E = [r["mean_eta_E"] for r in grid_rows]
    endo = {
        "corr_price_utilE_across_actions": float(
            np.corrcoef(price_levels, utils_E)[0, 1]
        ),
        "corr_utilE_etaE_across_actions": float(
            np.corrcoef(utils_E, etas_E)[0, 1]
        ),
        "mean_within_episode_corr_utilE_etaE_reference": confirmed[
            "reference"
        ]["corr_utilE_etaE"],
    }

    return {
        "multiplier": m,
        "grid": grid_rows,
        "confirmed": confirmed,
        "d_calibration": d_calib,
        "endogeneity": endo,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="reduced episode counts (smoke run)")
    args = ap.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    out = {
        "thresholds": THRESHOLDS,
        "d_rule_fraction": D_RULE_FRACTION,
        "probe_base_seed": 2000,
        "multipliers": {},
    }
    for m in MULTIPLIERS:
        out["multipliers"][str(m)] = probe_multiplier(m, quick=args.quick)
    out["elapsed_sec"] = time.time() - t0

    path = RESULTS_DIR / "feasibility_probe.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    print(f"saved -> {path} ({out['elapsed_sec']:.0f}s)")

    # Human-readable summary
    for m, r in out["multipliers"].items():
        print(f"\n=== m={m} ===")
        for k, c in r["d_calibration"].items():
            print(
                f"  [{k}] floor={c['floor']:,.0f} ({c['floor_policy']}) "
                f"J_C(best-reward {c['reward_best_policy']})="
                f"{c['J_C_of_reward_best_static']:,.0f} "
                f"d={c['d_rule_0.3']:,.0f} "
                f"feasible={c['feasible_with_margin']} "
                f"binding={c['binding_on_reward_best']}"
            )
        e = r["endogeneity"]
        print(
            f"  endogeneity: corr(price,util_E)="
            f"{e['corr_price_utilE_across_actions']:+.3f} "
            f"corr(util_E,eta_E)="
            f"{e['corr_utilE_etaE_across_actions']:+.3f}"
        )


if __name__ == "__main__":
    main()
