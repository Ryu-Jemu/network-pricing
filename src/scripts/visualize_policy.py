"""
visualize_policy.py — PPO 정책의 (state → action) 매핑 heatmap
================================================================
학습된 PPO 모델을 로딩한 뒤 (N_U/N_init_U, N_E/N_init_E) 격자에서
정책의 4-D 액션 (F_U, p_U, F_E, p_E)을 추론하여 4-패널 heatmap으로 시각화.

발표 보조 figure로 사용: "PPO가 어떻게 가격을 동적으로 설정하는가"의
정성적 가시화. 본 발표 본문 슬라이드 8/10의 보조 자료로 권장.

사용:
    PYTHONPATH=. python3 src/scripts/visualize_policy.py
    PYTHONPATH=. python3 src/scripts/visualize_policy.py --m 3 --seed 42
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

from stable_baselines3 import PPO

from src.env.network_slicing_env import NetworkSlicingEnv
from src.train.config import ENV_CONFIG

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
    })


def model_path(m: int, seed: int) -> str:
    if m == 1:
        return os.path.join(PROJ_ROOT, "models", "multi_seed",
                            f"ppo_seed{seed}.zip")
    return os.path.join(PROJ_ROOT, "models", "churn_sweep",
                        f"ppo_mult{m}_seed{seed}.zip")


def build_grid(env, n_grid: int):
    """관측 공간의 (N_U/N_init_U, N_E/N_init_E) 격자 + eta=eta_init 고정."""
    nu_axis = np.linspace(0.1, 1.5, n_grid)
    ne_axis = np.linspace(0.1, 1.5, n_grid)
    eta_u = float(env.eta_init[0])
    eta_e = float(env.eta_init[1])
    grid_obs = np.zeros((n_grid * n_grid, 4), dtype=np.float32)
    for i, nu in enumerate(nu_axis):
        for j, ne in enumerate(ne_axis):
            idx = i * n_grid + j
            grid_obs[idx] = (nu, ne, eta_u, eta_e)
    return nu_axis, ne_axis, grid_obs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=1,
                        help="churn multiplier (uses corresponding model)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-grid", type=int, default=21)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    setup_matplotlib()

    path = model_path(args.m, args.seed)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found: {path}")
    print(f"[load] {path}")

    env = NetworkSlicingEnv(config=ENV_CONFIG)
    model = PPO.load(path, env=env, device="cpu")

    nu_axis, ne_axis, grid_obs = build_grid(env, args.n_grid)
    actions, _ = model.predict(grid_obs, deterministic=True)

    F_U = actions[:, 0].reshape(args.n_grid, args.n_grid) * env.F_max[0]
    p_U = actions[:, 1].reshape(args.n_grid, args.n_grid) * env.p_max[0]
    F_E = actions[:, 2].reshape(args.n_grid, args.n_grid) * env.F_max[1]
    p_E = actions[:, 3].reshape(args.n_grid, args.n_grid) * env.p_max[1]

    fig, axes = plt.subplots(2, 2, figsize=(8, 7), constrained_layout=True)
    panels = [
        ("$F_U$ (URLLC fixed fee)", F_U, env.F_max[0]),
        ("$p_U$ (URLLC overage)",   p_U, env.p_max[0]),
        ("$F_E$ (eMBB fixed fee)",  F_E, env.F_max[1]),
        ("$p_E$ (eMBB overage)",    p_E, env.p_max[1]),
    ]

    for ax, (title, mat, vmax) in zip(axes.ravel(), panels):
        im = ax.imshow(mat.T, origin="lower", aspect="auto",
                       extent=(nu_axis[0], nu_axis[-1],
                               ne_axis[0], ne_axis[-1]),
                       vmin=0, vmax=vmax, cmap="viridis")
        ax.set_xlabel("$N_U / N_U^{init}$")
        ax.set_ylabel("$N_E / N_E^{init}$")
        ax.set_title(title)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=7)

    fig.suptitle(
        f"PPO 정책 행동 격자 (m={args.m}, seed={args.seed}, "
        f"$\\eta=\\eta_{{init}}={env.eta_init.tolist()}$)",
        fontsize=11, fontweight="bold",
    )

    out = args.out or os.path.join(
        PROJ_ROOT, "paper", f"fig_policy_heatmap_m{args.m}_seed{args.seed}.png"
    )
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    print(f"[saved] {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
