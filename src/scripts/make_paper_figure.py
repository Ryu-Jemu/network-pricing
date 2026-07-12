"""
make_paper_figure.py — 2-Panel Composite Result Figure for KIPS Paper
=====================================================================
Generates fig_results.png (replaces fig_churn_bar.png):

  Panel (a): Churn sweep — PPO vs Myopic-PPO vs Max-Price vs Static-Oracle
             Covers observations (1) and (2) in the paper.

  Panel (b): Algorithm comparison at m=1 — PPO vs SAC vs TD3 vs baselines
             Covers observation (3) in the paper.

Data sources (all under results/):
  - results/churn_sweep_results.json  (PPO, Max-Price, Static-Oracle @ all m)
  - results/myopic_sweep_results.json (Myopic-PPO @ all m)
  - results/multi_seed_results.json   (PPO, SAC, TD3 @ m=1)

Stats overlay (opt-in via env var STATS_OVERLAY=1):
  - Bootstrap 95% CI error bars on PPO/Myopic-PPO from per-seed means
  - Computed Welch + permutation p-value for PPO vs SAC at m=1
    (replaces the previously hardcoded p=0.938)
  - JSON dump of statistics to paper/fig_results_stats.json for the
    presentation deck / Q&A backing.
"""
from __future__ import annotations

import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

try:
    from src.train.stats_utils import (
        bootstrap_mean_ci, welch_t_test, permutation_test, cohens_d,
    )
    STATS_AVAILABLE = True
except Exception:
    STATS_AVAILABLE = False

STATS_OVERLAY = os.environ.get("STATS_OVERLAY", "0") == "1" and STATS_AVAILABLE

PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RESULTS_DIR = os.path.join(PROJ_ROOT, "results")

FIG_WIDTH = 3.35
DPI = 300


def setup_matplotlib():
    candidates = ["Apple SD Gothic Neo", "AppleGothic",
                  "NanumGothic", "Malgun Gothic", "Noto Sans CJK KR"]
    available = {f.name for f in font_manager.fontManager.ttflist}
    chosen = next((c for c in candidates if c in available), "DejaVu Sans")
    plt.rcParams.update({
        "font.family": chosen,
        "axes.unicode_minus": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 6.5,
        "axes.grid": True,
        "grid.alpha": 0.2,
    })
    print(f"[font] using '{chosen}'")


def load_data():
    """Load all data sources and return structured dicts."""
    with open(os.path.join(RESULTS_DIR, "churn_sweep_results.json")) as f:
        sweep = json.load(f)

    with open(os.path.join(RESULTS_DIR, "myopic_sweep_results.json")) as f:
        myopic = json.load(f)

    with open(os.path.join(RESULTS_DIR, "multi_seed_results.json")) as f:
        multi_seed = json.load(f)

    return sweep, myopic, multi_seed


def _seed_means_from_per_seed(per_seed_list):
    """Extract per-seed mean_reward list from churn_sweep per_seed structure."""
    return [s["mean_reward"] for s in per_seed_list]


def plot_panel_a(ax, sweep, myopic, stats_out=None):
    """Panel (a): Churn sweep bar chart — observations (1) and (2).

    If STATS_OVERLAY is on, replaces yerr on PPO/Myopic-PPO with asymmetric
    bootstrap 95% CI bars computed from per-seed means."""
    multipliers = [1, 3, 5, 10]
    n_groups = len(multipliers)

    policies = ["PPO", "Myopic-PPO", "Max-Price", "Static-Oracle"]
    colors = ["#1f77b4", "#e377c2", "#d62728", "#2ca02c"]
    hatches = ["", "///", "", ""]

    bar_width = 0.19
    n_policies = len(policies)

    if stats_out is not None:
        stats_out.setdefault("panel_a", {})

    for i, (policy, color, hatch) in enumerate(zip(policies, colors, hatches)):
        offset = (i - (n_policies - 1) / 2) * bar_width
        positions = np.arange(n_groups) + offset

        means, errs_lo, errs_hi = [], [], []
        for m in multipliers:
            mk = str(m)
            if policy == "PPO":
                d = sweep["multipliers"][mk]["ppo"]
                mean = d["mean_reward"]
                if STATS_OVERLAY and "per_seed" in d:
                    seed_means = _seed_means_from_per_seed(d["per_seed"])
                    _, lo, hi = bootstrap_mean_ci(seed_means, n_boot=5000, alpha=0.05)
                    e_lo, e_hi = max(0.0, mean - lo), max(0.0, hi - mean)
                    if stats_out is not None:
                        stats_out["panel_a"].setdefault(mk, {})["ppo"] = {
                            "seed_means": seed_means,
                            "mean": mean, "ci_low": lo, "ci_high": hi,
                        }
                else:
                    e_lo = e_hi = d.get("std_reward", 0)
                means.append(mean); errs_lo.append(e_lo); errs_hi.append(e_hi)
            elif policy == "Myopic-PPO":
                d = myopic["multipliers"][mk]
                mean = d["mean_reward"]
                if STATS_OVERLAY and "per_seed" in d:
                    seed_means = _seed_means_from_per_seed(d["per_seed"])
                    _, lo, hi = bootstrap_mean_ci(seed_means, n_boot=5000, alpha=0.05)
                    e_lo, e_hi = max(0.0, mean - lo), max(0.0, hi - mean)
                    if stats_out is not None:
                        stats_out["panel_a"].setdefault(mk, {})["myopic_ppo"] = {
                            "seed_means": seed_means,
                            "mean": mean, "ci_low": lo, "ci_high": hi,
                        }
                else:
                    e_lo = e_hi = d.get("std_reward", 0)
                means.append(mean); errs_lo.append(e_lo); errs_hi.append(e_hi)
            elif policy == "Max-Price":
                d = sweep["multipliers"][mk]["max_price"]
                means.append(d["mean_reward"])
                errs_lo.append(0); errs_hi.append(0)
            elif policy == "Static-Oracle":
                d = sweep["multipliers"][mk]["static_oracle"]
                means.append(d["mean_reward"])
                errs_lo.append(0); errs_hi.append(0)

        show_err = policy in ("PPO", "Myopic-PPO")
        yerr_arg = ([errs_lo, errs_hi] if show_err else None)
        ax.bar(
            positions, means, bar_width,
            yerr=yerr_arg,
            capsize=2, error_kw=dict(lw=0.7),
            color=color, edgecolor="black", linewidth=0.4,
            hatch=hatch, label=policy, zorder=3,
            alpha=0.9 if hatch else 1.0,
        )

    # Percentage labels above PPO bars (m >= 3)
    for j, m in enumerate(multipliers):
        if m == 1:
            continue
        mk = str(m)
        ppo_r = sweep["multipliers"][mk]["ppo"]["mean_reward"]
        mp_r = sweep["multipliers"][mk]["max_price"]["mean_reward"]
        ppo_std = sweep["multipliers"][mk]["ppo"].get("std_reward", 0)
        pct = (ppo_r - mp_r) / abs(mp_r) * 100
        offset_ppo = (0 - (n_policies - 1) / 2) * bar_width
        x_pos = j + offset_ppo
        ax.annotate(
            f"+{pct:.0f}%",
            xy=(x_pos, ppo_r + ppo_std),
            xytext=(0, 4), textcoords="offset points",
            fontsize=5.5, fontweight="bold", color="#1f77b4",
            ha="center", va="bottom",
        )

    ax.set_xticks(np.arange(n_groups))
    ax.set_xticklabels([f"{m}x" for m in multipliers])
    ax.set_xlabel("Churn Multiplier ($m$)")
    ax.set_ylabel("Net Reward")
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper right", framealpha=0.9, ncol=2)
    ax.set_title("(a) Churn Multiplier vs Net Reward", fontsize=8, fontweight="bold")


def plot_panel_b(ax, multi_seed, myopic_sweep, stats_out=None):
    """Panel (b): Algorithm comparison at m=1 — observation (3).

    Myopic-PPO 값은 multi_seed_results.json의 baselines["myopic_ppo"] 엔트리가
    max_price와 중복되어 있어 (2026-04-20 감사에서 확인), 실제 sweep 결과인
    myopic_sweep_results.json["multipliers"]["1"] 에서 읽어오도록 교정됨.

    STATS_OVERLAY 모드에서는 PPO/SAC/TD3의 std (across 60 episodes)를
    seed means의 bootstrap 95% CI로 대체하고, PPO vs SAC의 Welch + permutation
    p-value를 figure에 동시 표시. 그 외에는 기존 결과와 1:1 호환.
    """
    # Collect per-episode rewards for box-style error bars
    detail = multi_seed["multi_seed_detail"]

    policies = ["PPO", "SAC", "TD3"]
    colors = ["#1f77b4", "#17becf", "#8c564b"]

    # Max-Price from multi_seed baselines, Myopic-PPO from myopic sweep
    baselines = multi_seed["baselines"]
    myopic_m1 = myopic_sweep["multipliers"]["1"]
    all_names = policies + ["Myopic-PPO", "Max-Price"]
    all_colors = colors + ["#e377c2", "#d62728"]
    all_hatches = ["", "", "", "///", ""]

    if stats_out is not None:
        stats_out.setdefault("panel_b", {})

    # Per-seed means (3 seeds) for each RL algorithm — used for stats overlay
    seed_means_by_algo = {}
    for name in policies:
        seed_means_by_algo[name] = [
            float(np.mean([ep["total_reward"] for ep in seed_entry["eval_results"]]))
            for seed_entry in detail[name]
        ]

    means, errs_lo, errs_hi = [], [], []
    for name in all_names:
        if name in detail:
            rewards = [ep["total_reward"]
                       for seed_entry in detail[name]
                       for ep in seed_entry["eval_results"]]
            mean = float(np.mean(rewards))
            if STATS_OVERLAY:
                _, lo, hi = bootstrap_mean_ci(seed_means_by_algo[name],
                                              n_boot=5000, alpha=0.05)
                e_lo, e_hi = max(0.0, mean - lo), max(0.0, hi - mean)
                if stats_out is not None:
                    stats_out["panel_b"][name] = {
                        "seed_means": seed_means_by_algo[name],
                        "mean": mean, "ci_low": lo, "ci_high": hi,
                    }
            else:
                e_lo = e_hi = float(np.std(rewards))
            means.append(mean); errs_lo.append(e_lo); errs_hi.append(e_hi)
        elif name == "Myopic-PPO":
            means.append(myopic_m1["mean_reward"])
            s = myopic_m1.get("std_reward", 0)
            errs_lo.append(s); errs_hi.append(s)
        else:
            key_map = {"Max-Price": "max_price"}
            bdata = baselines[key_map[name]]
            means.append(bdata["mean_reward"])
            s = bdata.get("std_reward", 0)
            errs_lo.append(s); errs_hi.append(s)

    x = np.arange(len(all_names))
    bars = ax.bar(
        x, means, 0.55,
        yerr=[errs_lo, errs_hi], capsize=3, error_kw=dict(lw=0.8),
        color=all_colors, edgecolor="black", linewidth=0.4,
        zorder=3,
    )
    for bar, hatch in zip(bars, all_hatches):
        bar.set_hatch(hatch)

    ax.axvline(2.5, color="gray", lw=0.6, ls="--", alpha=0.5, zorder=1)

    # PPO vs SAC: Welch t + permutation. STATS_OVERLAY이면 재계산, 아니면
    # 논문 figure 호환을 위해 기존 하드코딩 표기 유지.
    if STATS_OVERLAY:
        a, b = seed_means_by_algo["PPO"], seed_means_by_algo["SAC"]
        _, p_w, _ = welch_t_test(a, b)
        p_perm = permutation_test(a, b, n_perm=10000)
        d = cohens_d(a, b)
        if stats_out is not None:
            stats_out["panel_b"]["PPO_vs_SAC"] = {
                "welch_p": p_w, "perm_p": p_perm, "cohens_d": d,
                "PPO_seed_means": a, "SAC_seed_means": b,
            }
        ann = f"n.s. (Welch $p$={p_w:.3f}, perm $p$={p_perm:.3f})"
    else:
        ann = "n.s. ($p$=0.938)"

    y_bracket = max(means[0] + errs_hi[0], means[1] + errs_hi[1]) + 100
    ax.plot([0, 0, 1, 1], [y_bracket - 30, y_bracket, y_bracket, y_bracket - 30],
            lw=0.7, color="black", zorder=5)
    ax.text(0.5, y_bracket + 20, ann,
            ha="center", va="bottom", fontsize=5.5, fontstyle="italic", zorder=5)

    for i, (m, s) in enumerate(zip(means, errs_hi)):
        ax.text(i, m + s + 40, f"{m:,.0f}", ha="center", va="bottom",
                fontsize=5, fontweight="bold" if i == 0 else "normal")

    ax.set_xticks(x)
    ax.set_xticklabels(all_names, fontsize=7)
    ax.set_ylabel("Net Reward")
    ax.set_ylim(bottom=6200, top=8400)
    ax.set_title("(b) Algorithm Comparison ($m$=1)", fontsize=8, fontweight="bold")

    # Axis break indicator
    d = 0.015
    kwargs = dict(transform=ax.transAxes, color='black', clip_on=False, lw=0.8)
    ax.plot((-d, +d), (-d, +d), **kwargs)
    ax.plot((-d, +d), (-d + 0.02, +d + 0.02), **kwargs)


def main():
    setup_matplotlib()

    print(f"\n[mode] STATS_OVERLAY={STATS_OVERLAY} "
          f"(available={STATS_AVAILABLE}, "
          f"env={os.environ.get('STATS_OVERLAY', 'unset')})")

    print("\n=== Loading data ===")
    sweep, myopic, multi_seed = load_data()

    stats_out = {} if STATS_OVERLAY else None

    print("\n=== Generating 2-panel composite figure ===")
    fig, (ax_a, ax_b) = plt.subplots(
        2, 1, figsize=(FIG_WIDTH, 3.7),
        gridspec_kw={"height_ratios": [1.3, 1.0]},
    )

    plot_panel_a(ax_a, sweep, myopic, stats_out=stats_out)
    plot_panel_b(ax_b, multi_seed, myopic, stats_out=stats_out)

    fig.tight_layout(h_pad=1.0)

    out_path = os.path.join(PROJ_ROOT, "paper", "fig_results.png")
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    print(f"[saved] {out_path}")

    if stats_out is not None:
        stats_path = os.path.join(PROJ_ROOT, "paper", "fig_results_stats.json")
        with open(stats_path, "w") as f:
            json.dump(stats_out, f, indent=2, default=float)
        print(f"[saved] {stats_path}")

    plt.close(fig)
    print("\n[done]")


if __name__ == "__main__":
    main()
