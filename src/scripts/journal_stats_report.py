"""Phase-D statistics report for the journal extension.

One command regenerates every number cited in the journal paper from
the stage JSONs under results/journal/. Seed-level means are the
statistical unit (n=7); per-policy comparisons are paired by eval
episode seed where applicable.

Produces:
  results/journal/stats_report.json   (machine-readable)
  results/journal/stats_report.md     (tables for the manuscript)

Usage:
    python -m src.scripts.journal_stats_report
"""
import json
from pathlib import Path

import numpy as np

from src.train.stats_utils import (
    bootstrap_mean_ci, welch_t_test, permutation_test, cohens_d,
)

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results" / "journal"
MULTIPLIERS = [1, 3, 5, 10]


def load(stage):
    path = RESULTS_DIR / f"{stage}.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def seed_means(stage_data, m, field="mean_reward", algo=None):
    out = []
    for r in stage_data["runs"].values():
        if r.get("multiplier") != m:
            continue
        if algo is not None and r.get("algo") != algo:
            continue
        out.append(r[field])
    return out


def agg(values):
    if not values:
        return None
    mean, lo, hi = bootstrap_mean_ci(values)
    return {
        "n": len(values), "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        "ci95": [lo, hi],
        "values": list(map(float, values)),
    }


def compare(a, b):
    """Two-sided tests on seed means a vs b (b may be a single baseline
    value repeated — then only the uplift is reported)."""
    if not a or not b:
        return None
    out = {"uplift_pct": float(
        100.0 * (np.mean(a) - np.mean(b)) / abs(np.mean(b))
    )}
    if len(b) > 1:
        t, p, _df = welch_t_test(a, b)
        out["welch_p"] = float(p)
        out["perm_p"] = float(permutation_test(a, b))
        out["cohens_d"] = float(cohens_d(a, b))
    return out


def main():
    c1 = load("c1_ppo")
    c2 = load("c2_lagrangian")
    c3 = load("c3_dsweep")
    c5 = load("c5_myopic")
    c6 = load("c6_algos")
    c7 = load("c7_bo")
    c8 = load("c8_mpc")
    c9 = load("c9_static")
    c10 = load("c10_negctrl")
    d_cal = load("d_calibration") or (
        json.load(open(RESULTS_DIR / "d_calibration.json"))
        if (RESULTS_DIR / "d_calibration.json").exists() else None
    )

    report = {"per_m": {}, "negative_control": None,
              "d_sweep": None, "algo_comparison": None}

    for m in MULTIPLIERS:
        row = {}
        ppo = seed_means(c1, m) if c1 else []
        lag = seed_means(c2, m) if c2 else []
        row["ppo_unconstrained"] = agg(ppo)
        row["ppo_lagrangian"] = agg(lag)
        if c2:
            row["ppo_lagrangian_J_C"] = agg(seed_means(c2, m, "mean_J_C"))
            row["ppo_unconstrained_J_C"] = agg(
                seed_means(c1, m, "mean_J_C"))
            row["constraint_satisfaction"] = agg(seed_means(
                c2, m, "constraint_satisfaction_rate"))
            row["price_of_safety_pct"] = compare(lag, ppo)
        if d_cal:
            row["d"] = d_cal["per_m"][str(m)]["d"]

        # honest-comparator table
        comparators = {}
        if c9:
            for name in ("max_price", "reference", "ppo_corner",
                         "static_oracle_grid", "load_threshold",
                         "peak_offpeak", "zero_price"):
                r = c9["runs"].get(f"m{m}_{name}")
                if r:
                    comparators[name] = {
                        "mean_reward": r["mean_reward"],
                        "mean_J_C": r["mean_J_C"],
                    }
        if c7:
            r = c7["runs"].get(f"endo_m{m}")
            if r:
                comparators["bo_oracle"] = {
                    "mean_reward": r["mean_reward"],
                    "mean_J_C": r["mean_J_C"],
                    "best_action": r["best_action"],
                }
        c11 = load("c11_constrained_static")
        if c11:
            r = c11["runs"].get(f"endo_m{m}")
            if r and r.get("feasible"):
                comparators["constrained_static"] = {
                    "mean_reward": r["mean_reward"],
                    "mean_J_C": r["mean_J_C"],
                    "best_action": r["best_action"],
                }
        if c8:
            r = c8["runs"].get(f"endo_m{m}")
            if r:
                comparators["oracle_mpc"] = {
                    "mean_reward": r["mean_reward"],
                    "mean_J_C": r["mean_J_C"],
                }
        if c5:
            myo = seed_means(c5, m)
            if myo:
                comparators["myopic_ppo"] = {
                    "mean_reward": float(np.mean(myo)),
                    "n_seeds": len(myo),
                }
        row["comparators"] = comparators

        # uplifts vs every comparator (PPO and Lagrangian)
        row["uplifts"] = {}
        for pol_name, pol_vals in (("ppo", ppo), ("ppo_lagrangian", lag)):
            for cname, cr in comparators.items():
                if not pol_vals:
                    continue
                base = cr["mean_reward"]
                row["uplifts"][f"{pol_name}_vs_{cname}"] = float(
                    100.0 * (np.mean(pol_vals) - base) / abs(base)
                )
        report["per_m"][str(m)] = row

    # paper-env BO/MPC audit-gap closure
    if c7 or c8:
        report["paper_env_strong_baselines"] = {}
        for m in (3, 5, 10):
            entry = {}
            if c7 and f"paper_m{m}" in c7["runs"]:
                entry["bo_oracle_reward"] = (
                    c7["runs"][f"paper_m{m}"]["mean_reward"])
                entry["bo_best_action"] = (
                    c7["runs"][f"paper_m{m}"]["best_action"])
            if c8 and f"paper_m{m}" in c8["runs"]:
                entry["oracle_mpc_reward"] = (
                    c8["runs"][f"paper_m{m}"]["mean_reward"])
            report["paper_env_strong_baselines"][str(m)] = entry

    # negative control
    if c10:
        runs = list(c10["runs"].values())
        report["negative_control"] = {
            "n_seeds": len(runs),
            "mean_J_C": float(np.mean([r["mean_J_C"] for r in runs])),
            "d": runs[0]["cost_limit_d"] if runs else None,
            "lam_saturated_frac": float(np.mean(
                [r["lam_saturated"] for r in runs])),
            "satisfaction": float(np.mean(
                [r["constraint_satisfaction_rate"] for r in runs])),
        }

    # d sweep
    if c3:
        sweep = {}
        for r in c3["runs"].values():
            f_ = str(r["d_factor"])
            sweep.setdefault(f_, {"rewards": [], "J_Cs": [],
                                  "d": r["cost_limit_d"]})
            sweep[f_]["rewards"].append(r["mean_reward"])
            sweep[f_]["J_Cs"].append(r["mean_J_C"])
        report["d_sweep"] = {
            f_: {"d": v["d"], "reward": agg(v["rewards"]),
                 "J_C": agg(v["J_Cs"])}
            for f_, v in sweep.items()
        }

    # algo comparison on the endo env (unconstrained; reward + SLA cost)
    if c6:
        ac = {}
        for algo in ("sac", "td3"):
            for m in (1, 3):
                vals = seed_means(c6, m, algo=algo)
                if vals:
                    ac[f"{algo}_m{m}"] = agg(vals)
                    ac[f"{algo}_m{m}_J_C"] = agg(
                        seed_means(c6, m, "mean_J_C", algo=algo))
        report["algo_comparison"] = ac

    with open(RESULTS_DIR / "stats_report.json", "w") as f:
        json.dump(report, f, indent=1, default=float)

    # markdown tables
    lines = ["# Journal stats report (auto-generated)", ""]
    lines.append("## Table 1 — reward by policy and m (endogenous env)")
    pols = ["ppo_unconstrained", "ppo_lagrangian"]
    comps = ["bo_oracle", "oracle_mpc", "static_oracle_grid",
             "max_price", "myopic_ppo", "load_threshold"]
    header = "| m | " + " | ".join(pols + comps) + " |"
    lines += [header, "|" + "---|" * (1 + len(pols) + len(comps))]
    for m in MULTIPLIERS:
        row = report["per_m"][str(m)]
        cells = [str(m)]
        for p in pols:
            a = row.get(p)
            cells.append(f"{a['mean']:,.0f}±{a['std']:,.0f}"
                         if a else "—")
        for cname in comps:
            cr = row["comparators"].get(cname)
            cells.append(f"{cr['mean_reward']:,.0f}" if cr else "—")
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("## Table 2 — CMDP constraint outcomes")
    lines += ["| m | d | Lagrangian J_C | satisfaction | "
              "unconstrained J_C | price of safety |",
              "|---|---|---|---|---|---|"]
    for m in MULTIPLIERS:
        row = report["per_m"][str(m)]
        jc_l = row.get("ppo_lagrangian_J_C")
        jc_u = row.get("ppo_unconstrained_J_C")
        sat = row.get("constraint_satisfaction")
        pos = row.get("price_of_safety_pct")
        lines.append(
            f"| {m} | {row.get('d', float('nan')):,.0f} | "
            f"{jc_l['mean']:,.0f} | " if jc_l else f"| {m} | — | — | "
        )
        if jc_l:
            lines[-1] += (
                f"{sat['mean']*100:.0f}% | {jc_u['mean']:,.0f} | "
                f"{pos['uplift_pct']:+.1f}% |"
            )
    ac = report.get("algo_comparison")
    if ac:
        lines.append("")
        lines.append("## Table 3 — algorithm comparison (endogenous env, "
                     "unconstrained)")
        lines += ["| algo | m | reward (mean±std) | J_C (mean) | seeds |",
                  "|---|---|---|---|---|"]
        for algo in ("sac", "td3"):
            for m in (1, 3):
                r = ac.get(f"{algo}_m{m}")
                jc = ac.get(f"{algo}_m{m}_J_C")
                if r:
                    lines.append(
                        f"| {algo.upper()} | {m} | "
                        f"{r['mean']:,.0f}±{r['std']:,.0f} | "
                        f"{jc['mean']:,.0f} | {r['n']} |"
                    )
    with open(RESULTS_DIR / "stats_report.md", "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"saved -> {RESULTS_DIR / 'stats_report.json'} and .md")


if __name__ == "__main__":
    main()
