"""Journal-extension experiment orchestrator (Phase C).

Pure-Python, cross-platform CLI. Every experiment unit writes its
result incrementally into results/journal/<stage>.json and is skipped
on re-run if already present, so the whole campaign is resumable.

Stages (run order matters only where noted):
    c1_ppo        PPO unconstrained (conference reward), endo env,
                  7 seeds x m in {1,3,5,10}
    d_calib      d per m from C1 J_C + probe floor  (needs c1_ppo)
    c7_bo        BO static oracle: endo all m + paper env m in {3,5,10}
    c9_static    static/heuristic/grid-oracle baselines, endo all m
    c2_lagrangian PPO-Lagrangian, 7 seeds x m        (needs d_calib)
    c3_dsweep    d sensitivity at m=3, factors {0.15, 0.5}, 7 seeds
    c10_negctrl  Lagrangian on exogenous paper env (negative control)
    c5_myopic    Myopic-PPO (gamma=0), endo, 3 seeds x m
    c6_algos     SAC + TD3, endo, m in {1,3}, 3 seeds
    c8_mpc       Oracle MPC: endo all m + paper m in {3,5,10}

Usage:
    python -m src.scripts.run_journal_experiments --stage c1_ppo
    python -m src.scripts.run_journal_experiments --stage all
"""
import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "results" / "journal"
MODELS_DIR = ROOT / "models" / "journal"

from src.env.network_slicing_env import NetworkSlicingEnv  # noqa: E402
from src.train.config import (  # noqa: E402
    get_env_config, PPO_CONFIG, SAC_CONFIG, TD3_CONFIG,
    MYOPIC_PPO_CONFIG, JOURNAL_SEEDS, JOURNAL_MULTIPLIERS,
    EVAL_PROTOCOL, LAGRANGIAN_CONFIG, BO_CONFIG, MPC_CONFIG,
    ORACLE_GRID, REFERENCE_ACTION,
)

MYOPIC_SEEDS = JOURNAL_SEEDS[:3]
ALGO_SEEDS = JOURNAL_SEEDS[:3]
PAPER_ENV_MULTIPLIERS = [3, 5, 10]   # close audit gap #2
D_SWEEP_FACTORS = [0.15, 0.5]        # main rule uses 0.3
D_RULE_FRACTION = 0.3


# ──────────────────────────────────────────────────────────────────
# Shared infrastructure
# ──────────────────────────────────────────────────────────────────

def git_sha():
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


def load_stage(stage):
    path = RESULTS_DIR / f"{stage}.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {"stage": stage, "runs": {}}


def save_stage(stage, data):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"{stage}.json"
    with open(path, "w") as f:
        json.dump(data, f, indent=1, default=float)


def manifest_add(stage, key, elapsed):
    path = RESULTS_DIR / "MANIFEST.json"
    entries = []
    if path.exists():
        with open(path) as f:
            entries = json.load(f)
    entries.append({
        "stage": stage, "key": key, "elapsed_sec": round(elapsed, 1),
        "git_sha": git_sha(), "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
    })
    with open(path, "w") as f:
        json.dump(entries, f, indent=1)


def evaluate_protocol(env_cfg, policy_fn, n_ep=None, base_seed=None):
    """Unified final evaluation: identical episode seeds across all
    policies (paired design). Returns aggregates + per-episode arrays.
    J_C uses the env's configured eta_sla via info['cost']."""
    n_ep = n_ep or EVAL_PROTOCOL["n_eval_episodes"]
    base_seed = base_seed or EVAL_PROTOCOL["eval_base_seed"]
    env = NetworkSlicingEnv(config=env_cfg)
    rewards, costs, revenues, penalties = [], [], [], []
    final_N_U, final_N_E = [], []
    for i in range(n_ep):
        obs, _ = env.reset(seed=base_seed + i)
        tot_r = tot_c = tot_rev = tot_pen = 0.0
        info = {}
        for _ in range(env.T):
            a = policy_fn(obs)
            obs, r, term, trunc, info = env.step(a)
            tot_r += r
            tot_c += info["cost"]
            tot_rev += info["revenue"]
            tot_pen += info["penalty"]
            if term or trunc:
                break
        rewards.append(tot_r)
        costs.append(tot_c)
        revenues.append(tot_rev)
        penalties.append(tot_pen)
        final_N_U.append(info["N_U"])
        final_N_E.append(info["N_E"])
    return {
        "n_eval_episodes": n_ep,
        "eval_base_seed": base_seed,
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "mean_J_C": float(np.mean(costs)),
        "std_J_C": float(np.std(costs)),
        "mean_revenue": float(np.mean(revenues)),
        "mean_penalty": float(np.mean(penalties)),
        "mean_final_N_U": float(np.mean(final_N_U)),
        "mean_final_N_E": float(np.mean(final_N_E)),
        "per_episode_rewards": list(map(float, rewards)),
        "per_episode_J_C": list(map(float, costs)),
    }


def model_policy(model):
    return lambda obs: model.predict(obs, deterministic=True)[0]


def const_policy(action):
    a = np.asarray(action, dtype=np.float32)
    return lambda obs: a


def endo_cfg(m):
    return get_env_config(endogenous=True, churn_multiplier=m)


def paper_cfg(m):
    return get_env_config(churn_multiplier=m)


# ──────────────────────────────────────────────────────────────────
# Training helpers
# ──────────────────────────────────────────────────────────────────

def train_sb3(algo, env_cfg, seed, save_path):
    """Train an SB3 model (skips if checkpoint exists); returns model."""
    from stable_baselines3 import PPO, SAC, TD3
    from stable_baselines3.common.monitor import Monitor

    cls = {"ppo": PPO, "sac": SAC, "td3": TD3, "myopic": PPO}[algo]
    hp = {
        "ppo": PPO_CONFIG, "sac": SAC_CONFIG,
        "td3": TD3_CONFIG, "myopic": MYOPIC_PPO_CONFIG,
    }[algo]

    if save_path.exists():
        return cls.load(save_path, device="cpu")

    env = Monitor(NetworkSlicingEnv(config=env_cfg))
    env.reset(seed=seed)
    kwargs = {
        k: v for k, v in hp.items()
        if k not in ("total_timesteps", "seed")
    }
    model = cls("MlpPolicy", env, seed=seed, verbose=0, device="cpu",
                **kwargs)
    model.learn(total_timesteps=hp["total_timesteps"], progress_bar=False)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(save_path)
    return model


# ──────────────────────────────────────────────────────────────────
# Stages
# ──────────────────────────────────────────────────────────────────

def stage_c1_ppo():
    stage = "c1_ppo"
    data = load_stage(stage)
    for m in JOURNAL_MULTIPLIERS:
        for seed in JOURNAL_SEEDS:
            key = f"m{m}_seed{seed}"
            if key in data["runs"]:
                continue
            t0 = time.time()
            path = MODELS_DIR / "c1_ppo" / f"ppo_endo_m{m}_seed{seed}.zip"
            model = train_sb3("ppo", endo_cfg(m), seed, path)
            res = evaluate_protocol(endo_cfg(m), model_policy(model))
            res["seed"], res["multiplier"] = seed, m
            data["runs"][key] = res
            save_stage(stage, data)
            manifest_add(stage, key, time.time() - t0)
            print(f"[{stage}] {key}: R={res['mean_reward']:,.0f} "
                  f"J_C={res['mean_J_C']:,.0f}", flush=True)


def stage_d_calib():
    """d per m: floor + 0.3*(J_C of the revenue-best policy - floor).

    Floor = min confirmed J_C from the feasibility probe (proposed
    threshold). Anchor = the realized J_C of whichever policy earns
    the highest mean reward among {BO static oracle, unconstrained
    PPO (seed-median)} — i.e. d sits below what unconstrained revenue
    maximisation costs, so the constraint is meaningfully binding.
    (Amended from the PPO-median anchor with user approval after the
    BO oracle revealed PPO under-explores the revenue-optimal region.)
    """
    c1 = load_stage("c1_ppo")["runs"]
    c7 = load_stage("c7_bo")["runs"]
    with open(RESULTS_DIR / "feasibility_probe.json") as f:
        probe = json.load(f)
    out = {"rule": "d = floor + 0.3*(J_C(revenue-best of "
                   "{BO, PPO-median}) - floor)",
           "threshold": "proposed eta_sla=[0.995,0.90]", "per_m": {}}
    for m in JOURNAL_MULTIPLIERS:
        runs_m = [r for r in c1.values() if r["multiplier"] == m]
        if not runs_m:
            raise RuntimeError(f"c1_ppo missing for m={m}")
        ppo_med_reward = float(np.median(
            [r["mean_reward"] for r in runs_m]))
        ppo_med_jc = float(np.median([r["mean_J_C"] for r in runs_m]))
        bo = c7.get(f"endo_m{m}")
        if bo and bo["mean_reward"] > ppo_med_reward:
            anchor_name, anchor_jc = "bo_oracle", bo["mean_J_C"]
            anchor_reward = bo["mean_reward"]
        else:
            anchor_name, anchor_jc = "ppo_median", ppo_med_jc
            anchor_reward = ppo_med_reward
        floor = min(
            r["J_C_proposed"] for r in
            probe["multipliers"][str(m)]["confirmed"].values()
        )
        d = floor + D_RULE_FRACTION * max(0.0, anchor_jc - floor)
        out["per_m"][str(m)] = {
            "floor": floor,
            "anchor_policy": anchor_name,
            "anchor_reward": anchor_reward,
            "anchor_J_C": anchor_jc,
            "median_ppo_J_C": ppo_med_jc,
            "d": d,
            "feasible_margin_ok": bool(floor <= 0.8 * d),
            "binding_on_anchor": bool(anchor_jc > d),
        }
        print(f"[d_calib] m={m}: floor={floor:,.0f} "
              f"anchor={anchor_name} J_C={anchor_jc:,.0f} "
              f"-> d={d:,.0f}")
    with open(RESULTS_DIR / "d_calibration.json", "w") as f:
        json.dump(out, f, indent=1)


def _load_d():
    with open(RESULTS_DIR / "d_calibration.json") as f:
        return json.load(f)["per_m"]


def _lambda_star(m):
    """Warm-start level for the PID integral: the enforcing multiplier
    derived from the measured unconstrained-PPO vs constrained-static
    tradeoff, lambda* = dR / d(J_C/d). Zero when unconstrained PPO is
    already feasible (e.g. m=1). Pre-registered design rule; the
    integral self-corrects downward if over-set."""
    c1 = load_stage("c1_ppo")["runs"]
    c11 = load_stage("c11_constrained_static")["runs"]
    d = _load_d()[str(m)]["d"]
    rs = [r for r in c1.values() if r["multiplier"] == m]
    r_ppo = float(np.median([r["mean_reward"] for r in rs]))
    jc_ppo = float(np.median([r["mean_J_C"] for r in rs]))
    cs = c11[f"endo_m{m}"]
    if jc_ppo <= d:
        return 0.0
    dr = max(0.0, r_ppo - cs["mean_reward"])
    djc = max(1e-9, (jc_ppo - cs["mean_J_C"]) / d)
    return dr / djc


def stage_c2_lagrangian():
    from src.train.train_ppo_lagrangian import (
        train_ppo_lagrangian, )
    stage = "c2_lagrangian"
    data = load_stage(stage)
    d_per_m = _load_d()
    for m in JOURNAL_MULTIPLIERS:
        d = d_per_m[str(m)]["d"]
        lam_star = _lambda_star(m)
        for seed in JOURNAL_SEEDS:
            key = f"m{m}_seed{seed}"
            if key in data["runs"]:
                continue
            t0 = time.time()
            path = (MODELS_DIR / "c2_lagrangian" /
                    f"ppolag_endo_m{m}_seed{seed}.zip")
            if path.exists():
                from stable_baselines3 import PPO
                model = PPO.load(path, device="cpu")
                lam_hist = data.get("lam_histories", {}).get(key, [])
            else:
                model, cb, lam_state = train_ppo_lagrangian(
                    seed=seed, cost_limit_raw=d,
                    lr_lam=LAGRANGIAN_CONFIG["lr_lam"],
                    cost_scale=1.0 / d,
                    lam_init=lam_star,
                    lam_max=LAGRANGIAN_CONFIG["lam_max"],
                    pid=True,
                    kp=LAGRANGIAN_CONFIG["kp"],
                    ki=LAGRANGIAN_CONFIG["ki"],
                    kd=LAGRANGIAN_CONFIG["kd"],
                    env_config=endo_cfg(m), verbose=0,
                )
                path.parent.mkdir(parents=True, exist_ok=True)
                model.save(path)
                lam_hist = [float(x) for x in cb.lam_history]
            res = evaluate_protocol(endo_cfg(m), model_policy(model))
            res["seed"], res["multiplier"] = seed, m
            res["cost_limit_d"] = d
            res["lam_warm_start"] = lam_star
            res["constraint_satisfaction_rate"] = float(np.mean(
                [c <= d for c in res["per_episode_J_C"]]
            ))
            res["lam_final"] = lam_hist[-1] if lam_hist else None
            data["runs"][key] = res
            data.setdefault("lam_histories", {})[key] = lam_hist
            save_stage(stage, data)
            manifest_add(stage, key, time.time() - t0)
            print(f"[{stage}] {key}: R={res['mean_reward']:,.0f} "
                  f"J_C={res['mean_J_C']:,.0f} (d={d:,.0f}) "
                  f"sat={res['constraint_satisfaction_rate']:.2f} "
                  f"lam={res['lam_final']}", flush=True)


def stage_c3_dsweep():
    from src.train.train_ppo_lagrangian import train_ppo_lagrangian
    stage = "c3_dsweep"
    data = load_stage(stage)
    d_info = _load_d()["3"]
    floor, med = d_info["floor"], d_info["median_ppo_J_C"]
    lam_star3 = _lambda_star(3)
    for f_ in D_SWEEP_FACTORS:
        d = floor + f_ * max(0.0, med - floor)
        for seed in JOURNAL_SEEDS:
            key = f"m3_f{f_}_seed{seed}"
            if key in data["runs"]:
                continue
            t0 = time.time()
            path = (MODELS_DIR / "c3_dsweep" /
                    f"ppolag_endo_m3_f{f_}_seed{seed}.zip")
            if path.exists():
                from stable_baselines3 import PPO
                model = PPO.load(path, device="cpu")
            else:
                model, cb, _ = train_ppo_lagrangian(
                    seed=seed, cost_limit_raw=d,
                    lr_lam=LAGRANGIAN_CONFIG["lr_lam"],
                    cost_scale=1.0 / d,
                    lam_init=lam_star3,
                    lam_max=LAGRANGIAN_CONFIG["lam_max"],
                    pid=True,
                    kp=LAGRANGIAN_CONFIG["kp"],
                    ki=LAGRANGIAN_CONFIG["ki"],
                    kd=LAGRANGIAN_CONFIG["kd"],
                    env_config=endo_cfg(3), verbose=0,
                )
                path.parent.mkdir(parents=True, exist_ok=True)
                model.save(path)
            res = evaluate_protocol(endo_cfg(3), model_policy(model))
            res["seed"], res["d_factor"], res["cost_limit_d"] = (
                seed, f_, d)
            res["constraint_satisfaction_rate"] = float(np.mean(
                [c <= d for c in res["per_episode_J_C"]]
            ))
            data["runs"][key] = res
            save_stage(stage, data)
            manifest_add(stage, key, time.time() - t0)
            print(f"[{stage}] {key}: R={res['mean_reward']:,.0f} "
                  f"J_C={res['mean_J_C']:,.0f} (d={d:,.0f})", flush=True)


def stage_c10_negctrl():
    """Negative control: Lagrangian on the exogenous paper env, where
    QoS is uncontrollable -> expect lambda saturation + violation
    (reproduces branch-1 infeasibility)."""
    from src.train.train_ppo_lagrangian import train_ppo_lagrangian
    stage = "c10_negctrl"
    data = load_stage(stage)
    d = LAGRANGIAN_CONFIG["cost_limit_raw"]  # branch-1's 70k
    for seed in MYOPIC_SEEDS:
        key = f"paper_m1_seed{seed}"
        if key in data["runs"]:
            continue
        t0 = time.time()
        model, cb, lam_state = train_ppo_lagrangian(
            seed=seed, cost_limit_raw=d,
            lr_lam=LAGRANGIAN_CONFIG["lr_lam"],
            cost_scale=1.0 / d,
            lam_max=LAGRANGIAN_CONFIG["lam_max"],
            pid=True,
            kp=LAGRANGIAN_CONFIG["kp"],
            ki=LAGRANGIAN_CONFIG["ki"],
            kd=LAGRANGIAN_CONFIG["kd"],
            env_config=paper_cfg(1), verbose=0,
        )
        res = evaluate_protocol(paper_cfg(1), model_policy(model))
        res["seed"], res["cost_limit_d"] = seed, d
        res["lam_final"] = float(lam_state[0])
        res["lam_saturated"] = bool(
            lam_state[0] >= LAGRANGIAN_CONFIG["lam_max"] * 0.99
        )
        res["constraint_satisfaction_rate"] = float(np.mean(
            [c <= d for c in res["per_episode_J_C"]]
        ))
        data["runs"][key] = res
        data.setdefault("lam_histories", {})[key] = [
            float(x) for x in cb.lam_history
        ]
        save_stage(stage, data)
        manifest_add(stage, key, time.time() - t0)
        print(f"[{stage}] {key}: J_C={res['mean_J_C']:,.0f} vs d={d:,.0f} "
              f"lam={res['lam_final']:.1f} "
              f"saturated={res['lam_saturated']}", flush=True)


def stage_c7_bo():
    from src.train.bo_oracle import bo_static_oracle
    stage = "c7_bo"
    data = load_stage(stage)
    cases = (
        [("endo", m, endo_cfg(m)) for m in JOURNAL_MULTIPLIERS]
        + [("paper", m, paper_cfg(m)) for m in PAPER_ENV_MULTIPLIERS]
    )
    for env_name, m, cfg in cases:
        key = f"{env_name}_m{m}"
        if key in data["runs"]:
            continue
        t0 = time.time()
        bo = bo_static_oracle(
            cfg, n_init=BO_CONFIG["n_init"], n_iter=BO_CONFIG["n_iter"],
            n_episodes=BO_CONFIG["n_episodes"], kappa=BO_CONFIG["kappa"],
            seed=42, include_corners=BO_CONFIG["include_corners"],
        )
        res = evaluate_protocol(
            cfg, const_policy(bo["best_action"])
        )
        res["env"], res["multiplier"] = env_name, m
        res["best_action"] = bo["best_action"]
        res["search_best_reward"] = bo["best_reward"]
        res["n_evaluations"] = bo["n_evaluations"]
        data["runs"][key] = res
        save_stage(stage, data)
        manifest_add(stage, key, time.time() - t0)
        print(f"[{stage}] {key}: action={np.round(bo['best_action'], 3)} "
              f"R={res['mean_reward']:,.0f} J_C={res['mean_J_C']:,.0f}",
              flush=True)


def _grid_static_oracle(cfg):
    """Conference-style grid search; selection on probe seeds (3 eps),
    confirmation on the eval protocol."""
    import itertools
    best_a, best_r = None, -np.inf
    for combo in itertools.product(
        ORACLE_GRID["F_U_range"], ORACLE_GRID["p_U_range"],
        ORACLE_GRID["F_E_range"], ORACLE_GRID["p_E_range"],
    ):
        r = 0.0
        env = NetworkSlicingEnv(config=cfg)
        for i in range(3):
            obs, _ = env.reset(seed=42 + i)
            a = np.asarray(combo, dtype=np.float32)
            for _ in range(env.T):
                obs, rew, term, trunc, _ = env.step(a)
                r += rew
                if term or trunc:
                    break
        if r > best_r:
            best_r, best_a = r, list(combo)
    return best_a


def stage_c9_static():
    from src.train.heuristic_baselines import (
        make_load_threshold_policy, make_peak_offpeak_policy,
    )
    stage = "c9_static"
    data = load_stage(stage)
    for m in JOURNAL_MULTIPLIERS:
        cfg = endo_cfg(m)
        static_policies = {
            "max_price": const_policy([1, 1, 1, 1]),
            "reference": const_policy(REFERENCE_ACTION),
            "zero_price": const_policy([0, 0, 0, 0]),
            "ppo_corner": const_policy([1, 0, 0, 1]),
        }
        for name, pol in static_policies.items():
            key = f"m{m}_{name}"
            if key in data["runs"]:
                continue
            t0 = time.time()
            res = evaluate_protocol(cfg, pol)
            res["multiplier"], res["policy"] = m, name
            data["runs"][key] = res
            save_stage(stage, data)
            manifest_add(stage, key, time.time() - t0)
            print(f"[{stage}] {key}: R={res['mean_reward']:,.0f} "
                  f"J_C={res['mean_J_C']:,.0f}", flush=True)
        # fresh factory per episode for the stateful heuristics
        for name, factory in (
            ("load_threshold", make_load_threshold_policy),
            ("peak_offpeak", make_peak_offpeak_policy),
        ):
            key = f"m{m}_{name}"
            if key in data["runs"]:
                continue
            t0 = time.time()

            class _FreshPolicy:
                def __init__(self, fct):
                    self.fct = fct
                    self.pol = fct()
                    self.steps = 0

                def __call__(self, obs):
                    if self.steps % 720 == 0:
                        self.pol = self.fct()
                    self.steps += 1
                    return self.pol(obs)

            res = evaluate_protocol(cfg, _FreshPolicy(factory))
            res["multiplier"], res["policy"] = m, name
            data["runs"][key] = res
            save_stage(stage, data)
            manifest_add(stage, key, time.time() - t0)
            print(f"[{stage}] {key}: R={res['mean_reward']:,.0f} "
                  f"J_C={res['mean_J_C']:,.0f}", flush=True)
        # grid static-oracle (conference continuity)
        key = f"m{m}_static_oracle_grid"
        if key not in data["runs"]:
            t0 = time.time()
            best_a = _grid_static_oracle(cfg)
            res = evaluate_protocol(cfg, const_policy(best_a))
            res["multiplier"], res["policy"] = m, "static_oracle_grid"
            res["best_action"] = best_a
            data["runs"][key] = res
            save_stage(stage, data)
            manifest_add(stage, key, time.time() - t0)
            print(f"[{stage}] {key}: a={best_a} "
                  f"R={res['mean_reward']:,.0f} "
                  f"J_C={res['mean_J_C']:,.0f}", flush=True)


def stage_c5_myopic():
    stage = "c5_myopic"
    data = load_stage(stage)
    for m in JOURNAL_MULTIPLIERS:
        for seed in MYOPIC_SEEDS:
            key = f"m{m}_seed{seed}"
            if key in data["runs"]:
                continue
            t0 = time.time()
            path = (MODELS_DIR / "c5_myopic" /
                    f"myopic_endo_m{m}_seed{seed}.zip")
            model = train_sb3("myopic", endo_cfg(m), seed, path)
            res = evaluate_protocol(endo_cfg(m), model_policy(model))
            res["seed"], res["multiplier"] = seed, m
            data["runs"][key] = res
            save_stage(stage, data)
            manifest_add(stage, key, time.time() - t0)
            print(f"[{stage}] {key}: R={res['mean_reward']:,.0f}",
                  flush=True)


def stage_c6_algos():
    stage = "c6_algos"
    data = load_stage(stage)
    for algo in ("sac", "td3"):
        for m in (1, 3):
            for seed in ALGO_SEEDS:
                key = f"{algo}_m{m}_seed{seed}"
                if key in data["runs"]:
                    continue
                t0 = time.time()
                path = (MODELS_DIR / "c6_algos" /
                        f"{algo}_endo_m{m}_seed{seed}.zip")
                model = train_sb3(algo, endo_cfg(m), seed, path)
                res = evaluate_protocol(endo_cfg(m),
                                        model_policy(model))
                res["seed"], res["multiplier"] = seed, m
                res["algo"] = algo
                data["runs"][key] = res
                save_stage(stage, data)
                manifest_add(stage, key, time.time() - t0)
                print(f"[{stage}] {key}: R={res['mean_reward']:,.0f}",
                      flush=True)


def stage_c8_mpc():
    from src.train.mpc_baseline import (
        run_mpc_episode_protocol,
    )
    stage = "c8_mpc"
    data = load_stage(stage)
    cases = (
        [("endo", m, endo_cfg(m)) for m in JOURNAL_MULTIPLIERS]
        + [("paper", m, paper_cfg(m)) for m in PAPER_ENV_MULTIPLIERS]
    )
    n_ep = EVAL_PROTOCOL["n_eval_episodes"]
    base = EVAL_PROTOCOL["eval_base_seed"]
    for env_name, m, cfg in cases:
        key = f"{env_name}_m{m}"
        if key in data["runs"]:
            continue
        t0 = time.time()
        rows = [
            run_mpc_episode_protocol(
                cfg, H=MPC_CONFIG["H"],
                n_rollouts=MPC_CONFIG["n_rollouts"],
                n_grid=MPC_CONFIG["n_grid"],
                replan_every=MPC_CONFIG["replan_every"],
                seed=base + i,
            )
            for i in range(n_ep)
        ]
        res = {
            "env": env_name, "multiplier": m,
            "n_eval_episodes": n_ep, "eval_base_seed": base,
            "mean_reward": float(np.mean([r["reward"] for r in rows])),
            "std_reward": float(np.std([r["reward"] for r in rows])),
            "mean_J_C": float(np.mean([r["J_C"] for r in rows])),
            "per_episode_rewards": [r["reward"] for r in rows],
            "per_episode_J_C": [r["J_C"] for r in rows],
        }
        data["runs"][key] = res
        save_stage(stage, data)
        manifest_add(stage, key, time.time() - t0)
        print(f"[{stage}] {key}: R={res['mean_reward']:,.0f} "
              f"J_C={res['mean_J_C']:,.0f}", flush=True)


def _measure_action(cfg, action, n_episodes=5, base_seed=2000):
    """Selection-time measurement of (reward, J_C) for a constant
    action on probe seeds (disjoint from the final-eval protocol)."""
    env = NetworkSlicingEnv(config=cfg)
    a = np.asarray(action, dtype=np.float32)
    rs, cs = [], []
    for i in range(n_episodes):
        obs, _ = env.reset(seed=base_seed + i)
        tot_r = tot_c = 0.0
        for _ in range(env.T):
            obs, r, term, trunc, info = env.step(a)
            tot_r += r
            tot_c += info["cost"]
            if term or trunc:
                break
        rs.append(tot_r)
        cs.append(tot_c)
    return float(np.mean(rs)), float(np.mean(cs))


def stage_c11_constrained_static():
    """Constrained static oracle: the best CONSTANT action subject to
    J_C <= d, selected on probe seeds and confirmed on the eval
    protocol. The honest static comparator for PPO-Lagrangian."""
    import itertools
    stage = "c11_constrained_static"
    data = load_stage(stage)
    d_per_m = _load_d()
    c7 = load_stage("c7_bo")["runs"]
    for m in JOURNAL_MULTIPLIERS:
        key = f"endo_m{m}"
        if key in data["runs"]:
            continue
        t0 = time.time()
        cfg = endo_cfg(m)
        d = d_per_m[str(m)]["d"]
        candidates = [list(c)
                      for c in itertools.product(GRID_LEVELS_C11,
                                                 repeat=4)]
        candidates += [[1, 1, 1, 1], [1, 0, 0, 1], [0, 0, 0, 0],
                       list(REFERENCE_ACTION)]
        bo = c7.get(f"endo_m{m}")
        if bo:
            candidates.append(list(bo["best_action"]))
        best_a, best_r = None, -np.inf
        feasible_count = 0
        for a in candidates:
            r, jc = _measure_action(cfg, a, n_episodes=3)
            if jc <= d:
                feasible_count += 1
                if r > best_r:
                    best_r, best_a = r, a
        if best_a is None:
            print(f"[{stage}] {key}: NO feasible constant action "
                  f"among {len(candidates)} candidates", flush=True)
            data["runs"][key] = {
                "multiplier": m, "cost_limit_d": d,
                "feasible": False,
                "n_candidates": len(candidates),
            }
        else:
            res = evaluate_protocol(cfg, const_policy(best_a))
            res["multiplier"], res["cost_limit_d"] = m, d
            res["feasible"] = True
            res["best_action"] = best_a
            res["n_candidates"] = len(candidates)
            res["n_feasible_candidates"] = feasible_count
            res["constraint_satisfaction_rate"] = float(np.mean(
                [c <= d for c in res["per_episode_J_C"]]
            ))
            data["runs"][key] = res
            print(f"[{stage}] {key}: a={np.round(best_a, 3)} "
                  f"R={res['mean_reward']:,.0f} "
                  f"J_C={res['mean_J_C']:,.0f} (d={d:,.0f})",
                  flush=True)
        save_stage(stage, data)
        manifest_add(stage, key, time.time() - t0)


GRID_LEVELS_C11 = [0.1, 0.3, 0.5, 0.7, 0.9]


STAGES = {
    "c1_ppo": stage_c1_ppo,
    "d_calib": stage_d_calib,
    "c7_bo": stage_c7_bo,
    "c9_static": stage_c9_static,
    "c2_lagrangian": stage_c2_lagrangian,
    "c3_dsweep": stage_c3_dsweep,
    "c10_negctrl": stage_c10_negctrl,
    "c5_myopic": stage_c5_myopic,
    "c6_algos": stage_c6_algos,
    "c8_mpc": stage_c8_mpc,
    "c11_constrained_static": stage_c11_constrained_static,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    help="comma-separated stage names or 'all'")
    args = ap.parse_args()
    names = (list(STAGES) if args.stage == "all"
             else [s.strip() for s in args.stage.split(",")])
    for name in names:
        if name not in STAGES:
            raise SystemExit(f"unknown stage {name}; "
                             f"choose from {list(STAGES)}")
        print(f"\n===== stage {name} =====", flush=True)
        STAGES[name]()


if __name__ == "__main__":
    main()
