"""
plot_trajectory.py — PPO 에피소드 trajectory (action/N/η/reward) 4-패널
========================================================================
multi_seed_results.json의 PPO trajectory (간격 100 step 샘플)을 사용하여
1개 에피소드(best, mean, worst 중 택1 — 기본 best)의 시간 진행을 4-패널
시각화. 발표 슬라이드 8 또는 backup으로 사용.

사용:
    PYTHONPATH=. python3 src/scripts/plot_trajectory.py
    PYTHONPATH=. python3 src/scripts/plot_trajectory.py --algo PPO --kind worst
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DPI = 200


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
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.grid": True,
        "grid.alpha": 0.25,
    })


def pick_episode(algo_detail, kind: str):
    """전체 (3 시드 × 20 에피소드)에서 kind ∈ {best, mean, worst} 1건 선택."""
    candidates = []
    for seed_entry in algo_detail:
        for ep in seed_entry["eval_results"]:
            candidates.append((ep["total_reward"], seed_entry["seed"], ep))
    candidates.sort(key=lambda x: x[0])
    if kind == "best":
        return candidates[-1]
    if kind == "worst":
        return candidates[0]
    return candidates[len(candidates) // 2]


def plot_trajectory(ax_F, ax_N, ax_eta, ax_R, ep_traj, label_prefix):
    t = np.array([p["t"] for p in ep_traj])
    F_U = np.array([p["F_U"] for p in ep_traj])
    p_U = np.array([p["p_U"] for p in ep_traj])
    F_E = np.array([p["F_E"] for p in ep_traj])
    p_E = np.array([p["p_E"] for p in ep_traj])
    N_U = np.array([p["N_U"] for p in ep_traj])
    N_E = np.array([p["N_E"] for p in ep_traj])
    eta_U = np.array([p["eta_U"] for p in ep_traj])
    eta_E = np.array([p["eta_E"] for p in ep_traj])
    revenue = np.array([p["revenue"] for p in ep_traj])
    penalty = np.array([p["penalty"] for p in ep_traj])

    ax_F.plot(t, F_U, "-o", color="#1f77b4", lw=1.4, ms=4, label="$F_U$")
    ax_F.plot(t, F_E, "-s", color="#ff7f0e", lw=1.4, ms=4, label="$F_E$")
    ax_F.plot(t, p_U, "--^", color="#1f77b4", lw=1.0, ms=4, label="$p_U$", alpha=0.7)
    ax_F.plot(t, p_E, "--^", color="#ff7f0e", lw=1.0, ms=4, label="$p_E$", alpha=0.7)
    ax_F.set_xlabel("Time $t$ (hour)")
    ax_F.set_ylabel("Price")
    ax_F.set_title("Action (price)")
    ax_F.legend(loc="best", fontsize=7, ncol=2)

    ax_N.plot(t, N_U, "-o", color="#1f77b4", lw=1.6, ms=4, label="$N_U$ (URLLC)")
    ax_N.plot(t, N_E, "-s", color="#ff7f0e", lw=1.6, ms=4, label="$N_E$ (eMBB)")
    ax_N.set_xlabel("Time $t$ (hour)")
    ax_N.set_ylabel("Active subscribers")
    ax_N.set_title("Subscriber dynamics")
    ax_N.legend(loc="best", fontsize=7)

    ax_eta.plot(t, eta_U, "-o", color="#1f77b4", lw=1.4, ms=4, label="$\\eta_U$")
    ax_eta.plot(t, eta_E, "-s", color="#ff7f0e", lw=1.4, ms=4, label="$\\eta_E$")
    ax_eta.axhline(0.99999, color="#1f77b4", ls=":", lw=0.8, alpha=0.6)
    ax_eta.axhline(0.90, color="#ff7f0e", ls=":", lw=0.8, alpha=0.6)
    ax_eta.set_xlabel("Time $t$ (hour)")
    ax_eta.set_ylabel("$\\eta$")
    ax_eta.set_ylim(0.80, 1.01)
    ax_eta.set_title("QoS realization vs target")
    ax_eta.legend(loc="best", fontsize=7)

    ax_R.plot(t, revenue, "-o", color="#2ca02c", lw=1.6, ms=4, label="Revenue")
    ax_R.plot(t, penalty, "-s", color="#d62728", lw=1.6, ms=4, label="Penalty")
    ax_R.set_xlabel("Time $t$ (hour)")
    ax_R.set_ylabel("Per-step amount")
    ax_R.set_title("Revenue vs Penalty")
    ax_R.legend(loc="best", fontsize=7)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", type=str, default="PPO",
                        choices=["PPO", "SAC", "TD3"])
    parser.add_argument("--kind", type=str, default="best",
                        choices=["best", "mean", "worst"])
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    setup_matplotlib()

    with open(os.path.join(PROJ_ROOT, "results",
                           "multi_seed_results.json")) as f:
        ms = json.load(f)

    detail = ms["multi_seed_detail"][args.algo]
    reward, seed, ep = pick_episode(detail, args.kind)
    print(f"[pick] {args.algo} {args.kind}: seed={seed} "
          f"reward={reward:.2f} #points={len(ep['trajectory'])}")

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    plot_trajectory(axes[0][0], axes[0][1], axes[1][0], axes[1][1],
                    ep["trajectory"], label_prefix=args.algo)

    fig.suptitle(
        f"{args.algo} 에피소드 trajectory ({args.kind}, seed={seed}, "
        f"total reward={reward:,.0f})",
        fontsize=11, fontweight="bold",
    )

    out = args.out or os.path.join(
        PROJ_ROOT, "paper",
        f"fig_trajectory_{args.algo.lower()}_{args.kind}_seed{seed}.png",
    )
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    print(f"[saved] {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
