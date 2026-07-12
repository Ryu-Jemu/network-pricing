"""Phase-D figures for the journal extension.

Reads stage JSONs under results/journal/ and writes publication
figures to figures/journal/. Korean-capable font chosen per platform.

    python -m src.scripts.make_journal_figures
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "results" / "journal"
FIG_DIR = ROOT / "figures" / "journal"
MULTIPLIERS = [1, 3, 5, 10]


def set_korean_font():
    candidates = ["Apple SD Gothic Neo", "AppleGothic", "NanumGothic",
                  "Malgun Gothic", "Noto Sans CJK KR"]
    available = {f.name for f in font_manager.fontManager.ttflist}
    for c in candidates:
        if c in available:
            plt.rcParams.update({"font.family": c,
                                 "axes.unicode_minus": False})
            return c
    return None


def load(stage):
    path = RESULTS_DIR / f"{stage}.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def seed_vals(stage_data, m, field="mean_reward"):
    return [r[field] for r in stage_data["runs"].values()
            if r.get("multiplier") == m and "seed" in r]


def fig0_system():
    """Closed-loop endogenous-QoS + CMDP system diagram (그림 1).

    Draws the price -> subscribers -> load -> QoS feedback loop and the
    CMDP reward/cost/lambda signals. Self-contained (no data)."""
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

    fig, ax = plt.subplots(figsize=(7.6, 3.9), dpi=200)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 60)
    ax.axis("off")

    def box(x, y, w, h, text, fc):
        ax.add_patch(FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.6,rounding_size=2",
            fc=fc, ec="#333333", lw=1.1))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=8.2, linespacing=1.35)

    def arrow(p0, p1, text="", style="-|>", ls="-", color="#333333",
              rad=0.0, dx=0, dy=1.8, fs=7.2):
        ax.add_patch(FancyArrowPatch(
            p0, p1, arrowstyle=style, mutation_scale=12, lw=1.1,
            color=color, linestyle=ls,
            connectionstyle=f"arc3,rad={rad}"))
        if text:
            mx, my = (p0[0] + p1[0]) / 2 + dx, (p0[1] + p1[1]) / 2 + dy
            ax.text(mx, my, text, ha="center", va="center", fontsize=fs,
                    color=color)

    # Top row: agent -> tariff -> subscribers -> load -> QoS
    box(2, 40, 20, 14, "RL agent\nπ(a | s)", "#dfe9f5")
    box(40, 42, 20, 12, "Subscriber dynamics\nchurn Binom · arrival Pois",
        "#e8f3ec")
    box(78, 42, 20, 12, "Realized load\nL = Σ q", "#fbeee0")
    # Bottom row: QoS -> reward/cost -> lambda
    box(78, 12, 20, 13, "Endogenous QoS\nη = clip(1 - α·(u - ρ*))", "#fbeee0")
    box(40, 12, 20, 13, "Reward r = R - w·penalty\nSLA cost c", "#e8f3ec")
    box(2, 12, 20, 14, "Dual variable λ\n(PID control)", "#dfe9f5")

    # Forward loop
    arrow((22, 47.5), (40, 48), "action a=(F,p)")
    arrow((60, 48), (78, 48), "price→subscribers")
    arrow((88, 42), (88, 25), "load→quality", dx=8, dy=0)
    arrow((78, 18.5), (60, 18.5), "billing · SLA metering")
    # Feedback to agent: cost -> lambda -> agent
    arrow((40, 18.5), (22, 18.5), "cost c")
    arrow((12, 26), (12, 40), "λ applied")
    # State feedback (N, eta) up to agent
    arrow((50, 25), (14, 40), "state s=(N, η)", ls=":", color="#7a7a7a",
          rad=-0.25, dy=3.0)
    # eta feeds prior-QoS into churn (dashed)
    arrow((80, 21), (52, 42), "previous QoS η(t-1)", ls="--",
          color="#3a6ea5", rad=0.28, dy=-3.0)

    ax.set_title("Endogenous-QoS closed loop and CMDP signal flow",
                 fontsize=9.5)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig0_system_diagram.png")
    plt.close(fig)


def fig1_reward_by_policy(c1, c2, c7, c8, c9):
    """Grouped comparison: PPO / PPO-Lag vs honest comparators per m."""
    fig, ax = plt.subplots(figsize=(7.2, 3.6), dpi=200)
    series = []
    if c1:
        series.append(("PPO (unconstrained)", "#1a7f5a",
                       [np.mean(seed_vals(c1, m)) for m in MULTIPLIERS],
                       [np.std(seed_vals(c1, m)) for m in MULTIPLIERS]))
    if c2:
        series.append(("PPO-Lagrangian", "#114b8a",
                       [np.mean(seed_vals(c2, m)) for m in MULTIPLIERS],
                       [np.std(seed_vals(c2, m)) for m in MULTIPLIERS]))
    if c7:
        series.append(("BO static oracle", "#8a4f11",
                       [c7["runs"][f"endo_m{m}"]["mean_reward"]
                        if f"endo_m{m}" in c7["runs"] else np.nan
                        for m in MULTIPLIERS], None))
    if c8:
        series.append(("Oracle MPC", "#6a3d9a",
                       [c8["runs"][f"endo_m{m}"]["mean_reward"]
                        if f"endo_m{m}" in c8["runs"] else np.nan
                        for m in MULTIPLIERS], None))
    if c9:
        series.append(("Max-Price", "#b22222",
                       [c9["runs"][f"m{m}_max_price"]["mean_reward"]
                        if f"m{m}_max_price" in c9["runs"] else np.nan
                        for m in MULTIPLIERS], None))
    x = np.arange(len(MULTIPLIERS))
    w = 0.8 / max(len(series), 1)
    for i, (label, color, ys, errs) in enumerate(series):
        ax.bar(x + i * w, ys, w, yerr=errs, capsize=2,
               label=label, color=color, alpha=0.88)
    ax.set_xticks(x + w * (len(series) - 1) / 2)
    ax.set_xticklabels([f"m={m}" for m in MULTIPLIERS])
    ax.set_ylabel("Net reward (mean over 20 eval episodes)")
    ax.set_title("Endogenous QoS: net reward by policy (7 seeds, paired eval)")
    ax.legend(fontsize=8, ncol=3)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig1_reward_by_policy.png")
    plt.close(fig)


def fig2_pareto(c1, c2, c3, d_cal):
    """Reward vs J_C at m=3: d-sweep tradeoff curve."""
    fig, ax = plt.subplots(figsize=(4.8, 3.6), dpi=200)
    pts = []
    if c3:
        for r in c3["runs"].values():
            pts.append((r["mean_J_C"], r["mean_reward"],
                        f"d×{r['d_factor']}"))
    if c2:
        for r in c2["runs"].values():
            if r.get("multiplier") == 3:
                pts.append((r["mean_J_C"], r["mean_reward"], "d×0.3"))
    if c1:
        for r in c1["runs"].values():
            if r.get("multiplier") == 3:
                pts.append((r["mean_J_C"], r["mean_reward"],
                            "unconstrained"))
    groups = {}
    for jc, rw, g in pts:
        groups.setdefault(g, []).append((jc, rw))
    colors = {"d×0.15": "#114b8a", "d×0.3": "#1a7f5a",
              "d×0.5": "#8a4f11", "unconstrained": "#b22222"}
    for g, arr in groups.items():
        arr = np.array(arr)
        ax.scatter(arr[:, 0], arr[:, 1], s=22,
                   color=colors.get(g, "gray"), label=g, alpha=0.8)
    if d_cal:
        d = d_cal["per_m"]["3"]["d"]
        ax.axvline(d, ls="--", color="#1a7f5a", lw=1,
                   label=f"d = {d:,.0f}")
    ax.set_xlabel("SLA shortfall cost J_C (episode cumulative)")
    ax.set_ylabel("Net reward")
    ax.set_title("m=3: revenue–SLA tradeoff (per-seed points)")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig2_pareto_m3.png")
    plt.close(fig)


def fig3_lambda(c2, c10):
    """Dual-variable trajectories: endogenous (controllable, lambda
    stabilizes) vs paper env negative control (infeasible -> lambda
    grows monotonically without ever satisfying the constraint)."""
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.0), dpi=200,
                             sharey=False)
    if c2 and "lam_histories" in c2:
        ax = axes[0]
        for key, hist in c2["lam_histories"].items():
            if key.startswith("m3_") and hist:
                ax.plot(hist, lw=0.9, alpha=0.8)
        ax.set_title("Endogenous QoS (m=3): λ stabilizes", fontsize=9)
        ax.set_xlabel("rollout")
        ax.set_ylabel("λ")
        ax.grid(alpha=0.3)
    if c10 and "lam_histories" in c10:
        ax = axes[1]
        for key, hist in c10["lam_histories"].items():
            if hist:
                ax.plot(hist, lw=0.9, alpha=0.8)
        ax.set_title("Exogenous QoS (negative control): "
                     "λ grows, constraint unmet", fontsize=9)
        ax.set_xlabel("rollout")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig3_lambda_trajectories.png")
    plt.close(fig)


def fig4_endogeneity():
    """Price level -> utilization -> QoS across the probe grid."""
    with open(RESULTS_DIR / "feasibility_probe.json") as f:
        probe = json.load(f)
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.2), dpi=200)
    g = probe["multipliers"]["3"]["grid"]
    price = [np.mean(r["action"]) for r in g]
    util = [r["mean_util_E"] for r in g]
    eta = [r["mean_eta_E"] for r in g]
    axes[0].scatter(price, util, s=14, color="#114b8a", alpha=0.75)
    axes[0].set_xlabel("Mean price level (normalized action)")
    axes[0].set_ylabel("Mean eMBB utilization")
    axes[0].set_title("Price → Load")
    axes[1].scatter(util, eta, s=14, color="#1a7f5a", alpha=0.75)
    axes[1].axhline(0.90, ls="--", color="#b22222", lw=1,
                    label="η_SLA,E = 0.90")
    axes[1].set_xlabel("Mean eMBB utilization")
    axes[1].set_ylabel("Mean eMBB QoS η")
    axes[1].set_title("Load → QoS")
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig4_endogeneity_m3.png")
    plt.close(fig)


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    set_korean_font()
    c1, c2, c3 = load("c1_ppo"), load("c2_lagrangian"), load("c3_dsweep")
    c7, c8, c9, c10 = (load("c7_bo"), load("c8_mpc"),
                       load("c9_static"), load("c10_negctrl"))
    d_cal = None
    if (RESULTS_DIR / "d_calibration.json").exists():
        with open(RESULTS_DIR / "d_calibration.json") as f:
            d_cal = json.load(f)
    fig0_system()
    if c1 or c2:
        fig1_reward_by_policy(c1, c2, c7, c8, c9)
    if c2 or c3:
        fig2_pareto(c1, c2, c3, d_cal)
    if c2 or c10:
        fig3_lambda(c2, c10)
    if (RESULTS_DIR / "feasibility_probe.json").exists():
        fig4_endogeneity()
    print(f"figures -> {FIG_DIR}")


if __name__ == "__main__":
    main()
