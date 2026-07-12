"""
Centralized configuration for all training and evaluation.
Parameter ratios calibrated to Korean 5G market data (SKT/KT/LG U+).

Korean Market Calibration Sources:
  - Simulated monthly churn ~1.0% (higher than Korean avg for
    dynamic emphasis; Korean actual: SKT 0.825%/yr, KT/LG U+
    ~1.0-1.4%/semi-annual; 시사저널e 2025)
  - 5G ARPU ~29,000-34,000 KRW/mo (SKT IR 2024 Q1-Q3)
  - 5G avg data usage ~27-30 GB/mo (MSIT 무선데이터트래픽 통계 2024)
  - URLLC QoS 99.999% / eMBB QoS ~90% (3GPP TS 22.261)
  - Price ratio URLLC:eMBB ≈ 1.67 (enterprise vs consumer tier)
  - Data add-on ~3,000 KRW/GB (SKT 추가 데이터팩 2025)
"""

# ── Environment Parameters ───────────────────────────────────────
# Monetary units are abstract simulation units; ratios reflect
# Korean 5G market structure (premium/consumer tier pricing).
# gamma0 calibrated so that reference-price monthly churn ≈ 1.0%,
# matching Korean telecom industry average (MSIT/KTOA statistics).
ENV_CONFIG = {
    # Tariff reference — ratio 50:30 ≈ 1.67 reflects Korean
    # enterprise(URLLC) vs consumer(eMBB) 5G plan price gap
    "F_ref": [50.0, 30.0],
    "p_ref": [10.0, 5.0],
    "Q_bar": [5.0, 30.0],
    # Action bounds
    "F_max": [100.0, 100.0],
    "p_max": [20.0, 20.0],
    # Traffic (LogNormal) — eMBB E[q]=29.96 reflects Korean 5G
    # avg usage ~30 GB/mo (MSIT 2024); URLLC is lower-volume
    "mu": [1.0, 3.0],
    "sigma2": [0.5, 0.8],
    # Departure — gamma0 tuned for monthly churn ≈ 1.0% at ref prices
    # (intentionally higher than Korean avg to emphasize dynamic effects;
    #  Korean actual: SKT ~0.07%/mo, KT/LG U+ ~0.17-0.23%/mo)
    # σ(-11.18) ≈ 1.395e-5/hr → (1-1.395e-5)^720 ≈ 0.99 → 1.0%/mo
    "gamma0": [-10.13, -12.53],
    "gamma_F": 1.0,
    "gamma_p": 0.8,
    # URLLC QoS sensitivity 6× eMBB (3GPP TS 22.261 reliability gap)
    "gamma_eta": [3.0, 0.5],
    # Arrival
    "beta0": [2.0, 2.5],
    "beta_F": 0.8,
    "beta_p": 0.6,
    "lambda_max": [0.05, 0.15],
    # QoS (3GPP TS 22.261)
    "eta_low": [0.90, 0.80],
    "eta_high": [1.0, 1.0],
    "eta_tgt": [0.99999, 0.90],
    # Penalty — URLLC 10× eMBB reflecting SLA stringency
    "w": [500.0, 50.0],
    # Reward scaling (stabilize critic learning)
    "reward_scale": 1e-5,
    # MDP — 720 steps = 1 month (hourly granularity)
    "T": 720,
    "gamma": 0.99,
    # Initial state
    "N_init": [1000.0, 5000.0],
    "eta_init": [0.95, 0.90],
}

# ── Phase 11: Telecom+Econ extension config (OFF by default) ─────
# Activate by deep-copying ENV_CONFIG and overlaying these keys, or by
# calling get_env_config(extended=True). Default ENV_CONFIG above gives
# numerically identical behavior to the published paper.
#
# Citations (all verified):
#   Bolton 1998 Marketing Science 17(1):45-65 — duration & satisfaction
#   Cox 1972 JRSS B 34(2):187-220 — proportional hazards
#   Klemperer 1987 Economic Journal 97(Supp):99-117 — switching costs
#   Gerpott, Rams & Schindler 2001 Telecom. Policy 25(4):249-269 — asym γ
#   Tirole 1988 *The Theory of Industrial Organization* — 3rd-degree PD
ENV_CONFIG_TELECOM_ECON = {
    # Master flag — must be True to enable cohort logic
    "cohort_aware": True,
    # Gerpott 2001 + Tirole 1988: B2B (URLLC) less price-sensitive than B2C (eMBB)
    "gamma_F_per_slice": [0.4, 1.5],  # original symmetric value was 1.0
    "gamma_p_per_slice": [0.3, 0.8],  # original symmetric value was 0.8
    # Cox 1972 + Bolton 1998: tenure cohort hazard reduction
    "n_cohorts": 6,
    "cohort_bins_months": [0.5, 2.0, 4.5, 9.0, 18.0, 36.0],
    "cohort_init": [
        # URLLC 1,000 with enterprise long-tenure skew
        [50.0, 100.0, 200.0, 300.0, 250.0, 100.0],
        # eMBB 5,000 with consumer-flatter distribution
        [500.0, 1000.0, 1500.0, 1000.0, 700.0, 300.0],
    ],
    "alpha_tenure": 0.3,  # Cox PH log-hazard coefficient
    # Klemperer 1987: switching cost lock-in proportional to mean tenure
    "beta_sc": 0.5,
}


# ── Journal extension: endogenous QoS env overlay ────────────────
# Closes the price → subscribers → load → QoS loop (improvement-2).
#
# Calibration (analyze_load_sla.py, results/journal/load_sla_analysis
# .json): branch-2's original capacity [5000, 215000] with rho_star
# 0.5 left URLLC chronically overloaded (violation onset N*=750 <
# N_init=1000, uncontrollable by pricing) and eMBB never violating
# (N*=5980, unreachable) — episode-cost controllability only 17–35%.
# Recalibrated so the SLA-load coupling is *operative*:
#   rho_star_U = 0.70  → URLLC healthy at its initial base (N*=1040)
#   capacity_E = 176k  → eMBB enters the SLA-risk region at full base
#                        (N*=4900 < N_init=5000); shedding ~12% of
#                        subscribers restores compliance.
# Result: episode-cost controllability 96–97%; constraint is feasible
# (shed-eMBB policies reach J_C ≈ 4–8k) and binds on retention-heavy
# policies (J_C ≈ 26–38k at full base).
#
# eta_sla is the CMDP constraint threshold: the URLLC physical target
# 0.99999 is unreachable under measurement jitter delta_qos_U = 0.01,
# so the operational SLA threshold [0.995, 0.90] is used for the
# constraint; the reward penalty keeps eta_tgt unchanged.
ENV_CONFIG_ENDO = {
    "qos_endogenous": True,
    "capacity": [5000.0, 176000.0],
    "rho_star": [0.70, 0.50],
    "alpha_qos": [0.45, 0.30],
    "delta_qos": [0.01, 0.02],
    "eta_sla": [0.995, 0.90],
}


def get_env_config(
    extended: bool = False,
    endogenous: bool = False,
    churn_multiplier: float | None = None,
):
    """Return env config dict.

    Args:
        extended: If True, overlay Phase 11 telecom+econ extension keys.
        endogenous: If True, overlay endogenous-QoS keys (journal env).
        churn_multiplier: If given (>0), add ln(m) to both gamma0 entries
            following src/scripts/run_churn_sweep.py:make_env_config.

    Returns:
        New dict (deep-copied) suitable for `NetworkSlicingEnv(config=...)`.
    """
    import copy
    cfg = copy.deepcopy(ENV_CONFIG)
    if extended:
        cfg.update(copy.deepcopy(ENV_CONFIG_TELECOM_ECON))
    if endogenous:
        cfg.update(copy.deepcopy(ENV_CONFIG_ENDO))
    if churn_multiplier is not None and churn_multiplier > 0:
        import math
        offset = math.log(float(churn_multiplier))
        cfg["gamma0"] = [
            ENV_CONFIG["gamma0"][0] + offset,
            ENV_CONFIG["gamma0"][1] + offset,
        ]
    return cfg

# ── SAC Hyperparameters ──────────────────────────────────────────
SAC_CONFIG = {
    "learning_rate": 3e-4,
    "batch_size": 256,
    "buffer_size": 1_000_000,
    "tau": 0.005,
    "gamma": 0.99,
    "ent_coef": "auto",
    "policy_kwargs": dict(net_arch=[256, 256, 256]),
    "total_timesteps": 720 * 500,   # 500 episodes
    "seed": 42,
}

# ── PPO Hyperparameters ──────────────────────────────────────────
PPO_CONFIG = {
    "learning_rate": 3e-4,
    "batch_size": 64,
    "n_epochs": 10,
    "clip_range": 0.2,
    "gae_lambda": 0.95,
    "gamma": 0.99,
    "policy_kwargs": dict(net_arch=[256, 256, 256]),
    "total_timesteps": 720 * 500,   # 500 episodes
    "seed": 42,
}

# ── TD3 Hyperparameters ─────────────────────────────────────────
TD3_CONFIG = {
    "learning_rate": 3e-4,
    "batch_size": 256,
    "buffer_size": 1_000_000,
    "tau": 0.005,
    "gamma": 0.99,
    "policy_kwargs": dict(net_arch=[256, 256, 256]),
    "total_timesteps": 720 * 500,
    "seed": 42,
}

# ── Myopic (γ=0) PPO — no long-horizon planning ────────────────
MYOPIC_PPO_CONFIG = {
    "learning_rate": 3e-4,
    "batch_size": 64,
    "n_epochs": 10,
    "clip_range": 0.2,
    "gae_lambda": 0.95,
    "gamma": 0.0,           # ← myopic: no discounting
    "policy_kwargs": dict(net_arch=[256, 256, 256]),
    "total_timesteps": 720 * 500,
    "seed": 42,
}

# ── Evaluation ───────────────────────────────────────────────────
EVAL_CONFIG = {
    "n_eval_episodes": 20,
    "train_seeds": [42, 123, 456],   # multi-seed training
}

# ── Journal extension: experiment protocol ───────────────────────
# 7 training seeds (matches improvement-9 seed_power_paper runs).
JOURNAL_SEEDS = [42, 123, 456, 789, 1011, 1213, 1415]
JOURNAL_MULTIPLIERS = [1, 3, 5, 10]

# Unified final-evaluation protocol: EVERY policy (RL, Lagrangian, BO,
# MPC, heuristics, static actions) is scored on episode seeds
# eval_base_seed .. eval_base_seed + n_eval_episodes - 1. Identical
# episode seeds across policies → paired comparisons are valid.
# Selection/search procedures (BO acquisition, oracle grids, MPC inner
# rollouts) must NOT use these seeds — they keep their own RNG streams.
EVAL_PROTOCOL = {
    "eval_base_seed": 1000,
    "n_eval_episodes": 20,
}

# ── PPO-Lagrangian (CMDP dual ascent) ────────────────────────────
# cost_limit_raw is per-(m) and set from results/journal/
# d_calibration.json; the value below is only a smoke-test fallback.
#
# Dual scaling (journal recalibration): costs are normalised by the
# budget, cost_scale = 1/d, so the scaled episode constraint is always
# J_C/d <= 1 and lambda is dimensionless. The enforcing multiplier
# follows from the measured revenue-cost tradeoff at m=3
# (unconstrained PPO -> constrained static: dR = 975 reward units,
# dJ_C/d = 3.94), giving lambda* ~ 247; lam_max = 4000 leaves ~16x
# headroom. Branch-1's fixed cost_scale=1e-5 was calibrated for raw
# costs ~1e5 and caps enforcement two orders of magnitude short on
# the recalibrated env (lambda* ~ 1.2e4 vs lam_max 100).
# PID gains (Stooke et al. 2020): with budget-normalised error
# e = J_C/d - 1 in O(0.1-3), Kp=20 gives immediate lambda ~ O(60) at
# onset, Ki=5 accumulates the enforcing level lambda* ~ 250 within
# ~15-50 rollouts, Kd=20 damps while cost rises. A deterministic probe
# episode every 5 rollouts closes the stochastic-train vs
# deterministic-eval cost gap (observed: lambda -> 0 while the
# deterministic policy violated by 1.5x).
LAGRANGIAN_CONFIG = {
    "cost_limit_raw": 70000.0,
    "lr_lam": 5.0,          # legacy dual-ascent path only
    "lam_max": 4000.0,
    "cost_scale_rule": "1/d",
    "lam_init": 0.0,
    "kp": 20.0,
    "ki": 5.0,
    "kd": 20.0,
}

# ── BO static oracle (GP-UCB) ────────────────────────────────────
BO_CONFIG = {
    "n_init": 8,
    "n_iter": 22,
    "n_episodes": 5,
    "kappa": 2.5,
    "include_corners": True,   # seed the 16 action-box corners
}

# ── Oracle MPC ───────────────────────────────────────────────────
MPC_CONFIG = {
    "H": 24,
    "n_rollouts": 3,
    "n_grid": 3,
    "replan_every": 24,
}

# ── Reference action for Static-Heuristic baseline ───────────────
# F_U=50/100=0.5, p_U=10/20=0.5, F_E=30/100=0.3, p_E=5/20=0.25
REFERENCE_ACTION = [0.5, 0.5, 0.3, 0.25]

# ── Static-Oracle grid search parameters ─────────────────────────
ORACLE_GRID = {
    "F_U_range": [0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
    "p_U_range": [0.3, 0.4, 0.5, 0.6, 0.7],
    "F_E_range": [0.2, 0.3, 0.4, 0.5, 0.6],
    "p_E_range": [0.2, 0.3, 0.4, 0.5],
    "n_eval_episodes": 5,
}
