"""
PPO-Lagrangian for SLA-constrained dynamic pricing.

Constrained MDP (CMDP) formulation:
    max_pi  E[sum gamma^t r_t]
    s.t.    E[sum gamma^t c_t] <= d  (cumulative SLA constraint)

Lagrangian:
    L(pi, lam) = E[sum gamma^t (r_t - lam * c_t)] + lam * d

Dual ascent (projected):
    lam_{k+1} = max(0, lam_k + eta_lam * (J_C(pi_k) - d))

Implementation references:
- Achiam et al., "Constrained Policy Optimization" (CPO), ICML 2017.
- Stooke, Achiam, Abbeel, "Responsive Safety in RL by PID
  Lagrangian Methods" (PPO-Lagrangian), ICML 2020.
- Nagib, Abou-Zeid, Hassanein, "SafeSlice", ICMLCN 2025
  (arXiv:2503.12753).

We use the (a) approach: subtract lam * c from the per-step reward
via a Gymnasium wrapper, then update lam after each rollout based on
the empirical cost return measured on the unmodified env signal
(info["cost"]).
"""
import json
from typing import List

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from src.env.network_slicing_env import NetworkSlicingEnv
from src.train.progress import default_progress_bar
from src.train.config import ENV_CONFIG, PPO_CONFIG


class LagrangianCostWrapper(gym.Wrapper):
    """Subtracts lam * cost from reward (does NOT modify env state).

    Lambda is a shared mutable container so the training callback can
    update it in place without re-wrapping the env each rollout.
    """

    def __init__(self, env, lam_state, cost_scale=1.0):
        super().__init__(env)
        self.lam_state = lam_state
        self.cost_scale = float(cost_scale)
        self._ep_cost = 0.0

    def reset(self, **kwargs):
        self._ep_cost = 0.0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        c_raw = float(info.get("cost", 0.0))
        c_scaled = c_raw * self.cost_scale
        # SCALED units used throughout: ep_cost, cost_limit, lam are
        # all in O(1) so dual ascent has well-conditioned dynamics.
        self._ep_cost += c_scaled
        info["cost_raw"] = c_raw
        info["cost_scaled"] = c_scaled
        info["ep_cost_running"] = self._ep_cost
        if terminated or truncated:
            info["ep_cost"] = self._ep_cost  # in scaled units
            info["ep_cost_raw"] = self._ep_cost / self.cost_scale
        lam = float(self.lam_state[0])
        # Reward subtraction in scaled units (matches base reward).
        return obs, reward - lam * c_scaled, terminated, truncated, info


class PIDLagrangianCallback(BaseCallback):
    """PID-controlled Lagrange multiplier (Stooke, Achiam & Abbeel,
    "Responsive Safety in RL by PID Lagrangian Methods," ICML 2020).

    Pure dual ascent is integral-only control and oscillates: lambda
    overshoots, the policy over-corrects, lambda collapses, the policy
    reverts (observed empirically on this env — see
    PARAMETER_JUSTIFICATION.md). The PID form adds a proportional term
    (immediate response to current violation) and a derivative term
    (damping while cost is rising):

        e_t  = J_C - d              (in scaled units; here J_C/d - 1)
        I_t  = max(0, I_{t-1} + Ki * e_t)
        lam  = clip(Kp * max(0, e_smooth) + I_t + Kd * max(0, de), 0,
                    lam_max)

    Two additions over the vanilla form:
    - EMA smoothing of e_t (rollouts contain only ~3 episodes).
    - A periodic DETERMINISTIC probe episode: the deployed policy is
      deterministic, and its cost can exceed the exploratory training
      cost (train/eval gap observed on this env). The error signal
      uses max(stochastic EMA, last deterministic probe).
    """

    def __init__(
        self,
        lam_state: List[float],
        cost_limit: float,
        kp: float = 20.0,
        ki: float = 5.0,
        kd: float = 20.0,
        lam_max: float = 2000.0,
        ema_alpha: float = 0.3,
        probe_env_config: dict = None,
        probe_every: int = 5,
        probe_cost_scale: float = 1.0,
        probe_seed0: int = 3000,
        integral_init: float = 0.0,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.lam_state = lam_state
        self.cost_limit = float(cost_limit)
        self.kp, self.ki, self.kd = float(kp), float(ki), float(kd)
        self.lam_max = float(lam_max)
        self.ema_alpha = float(ema_alpha)
        self.probe_env_config = probe_env_config
        self.probe_every = int(probe_every)
        self.probe_cost_scale = float(probe_cost_scale)
        self.probe_seed0 = int(probe_seed0)
        self.episode_costs: List[float] = []
        self.episode_rewards: List[float] = []
        self.lam_history: List[float] = []
        self.constraint_satisfied: List[bool] = []
        self._rollout_costs: List[float] = []
        # Warm start: initialise the integral term at the derived
        # enforcing level lambda*(m) so high-m runs do not spend most
        # of the training budget ramping lambda up from zero. The
        # integral self-corrects downward whenever e < 0.
        self._integral = max(0.0, float(integral_init))
        self._e_ema = None
        self._e_prev = 0.0
        self._e_det = 0.0
        self._n_rollouts = 0

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "ep_cost" in info:
                self.episode_costs.append(float(info["ep_cost"]))
                self._rollout_costs.append(float(info["ep_cost"]))
            if "episode" in info:
                self.episode_rewards.append(float(info["episode"]["r"]))
        return True

    def _deterministic_probe(self):
        """One deterministic episode on a held-out probe seed; returns
        episode cost in scaled units."""
        env = NetworkSlicingEnv(config=self.probe_env_config)
        obs, _ = env.reset(seed=self.probe_seed0 + self._n_rollouts)
        cost = 0.0
        for _ in range(env.T):
            a, _ = self.model.predict(obs, deterministic=True)
            obs, _, term, trunc, info = env.step(a)
            cost += float(info["cost"]) * self.probe_cost_scale
            if term or trunc:
                break
        return cost

    def _on_rollout_end(self) -> None:
        self._n_rollouts += 1
        if self._rollout_costs:
            j_c = float(np.mean(self._rollout_costs))
            e_t = j_c - self.cost_limit
        else:
            e_t = self._e_ema if self._e_ema is not None else 0.0
        self._rollout_costs = []
        self._e_ema = (
            e_t if self._e_ema is None
            else (1 - self.ema_alpha) * self._e_ema + self.ema_alpha * e_t
        )
        if (self.probe_env_config is not None
                and self._n_rollouts % self.probe_every == 0):
            self._e_det = self._deterministic_probe() - self.cost_limit
        e_used = max(self._e_ema, self._e_det)
        self._integral = max(0.0, self._integral + self.ki * e_used)
        d_term = self.kd * max(0.0, e_used - self._e_prev)
        self._e_prev = e_used
        new_lam = min(
            self.lam_max,
            max(0.0, self.kp * max(0.0, e_used))
            + self._integral + d_term,
        )
        self.lam_state[0] = new_lam
        self.lam_history.append(new_lam)
        self.constraint_satisfied.append(e_used <= 0.0)
        if self.verbose:
            print(
                f"  rollout {self._n_rollouts} | "
                f"e_stoch={self._e_ema:+.3f} e_det={self._e_det:+.3f} "
                f"I={self._integral:.1f} lam={new_lam:.1f}"
            )


class DualAscentCallback(BaseCallback):
    """Updates lambda after each PPO rollout using empirical cost return.

    Lambda update:
        lam <- max(0, lam + lr_lam * (J_C - d))
    where J_C is the average per-episode cumulative cost in the most
    recent rollout. Uses log-space to avoid sign issues.
    """

    def __init__(
        self,
        lam_state: List[float],
        cost_limit: float,
        lr_lam: float = 0.05,
        lam_max: float = 100.0,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.lam_state = lam_state
        self.cost_limit = float(cost_limit)
        self.lr_lam = float(lr_lam)
        self.lam_max = float(lam_max)
        self.episode_costs: List[float] = []
        self.episode_rewards: List[float] = []
        self.episode_final_N_U: List[float] = []
        self.episode_final_N_E: List[float] = []
        self.lam_history: List[float] = []
        self.constraint_satisfied: List[bool] = []
        self._rollout_costs: List[float] = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if "ep_cost" in info:
                self.episode_costs.append(float(info["ep_cost"]))
                self._rollout_costs.append(float(info["ep_cost"]))
            if "episode" in info:
                self.episode_rewards.append(float(info["episode"]["r"]))
                self.episode_final_N_U.append(float(info.get("N_U", 0)))
                self.episode_final_N_E.append(float(info.get("N_E", 0)))
        return True

    def _on_rollout_end(self) -> None:
        if not self._rollout_costs:
            self.lam_history.append(float(self.lam_state[0]))
            return
        j_c = float(np.mean(self._rollout_costs))
        violation = j_c - self.cost_limit
        new_lam = max(0.0, self.lam_state[0] + self.lr_lam * violation)
        new_lam = min(new_lam, self.lam_max)
        self.lam_state[0] = new_lam
        self.lam_history.append(new_lam)
        self.constraint_satisfied.append(violation <= 0.0)
        self._rollout_costs = []
        if self.verbose:
            ep_n = len(self.episode_rewards)
            r_recent = (
                np.mean(self.episode_rewards[-50:])
                if self.episode_rewards else 0.0
            )
            print(
                f"  rollout end | ep={ep_n} "
                f"reward~{r_recent:.0f} "
                f"J_C={j_c:.2f} "
                f"limit={self.cost_limit:.2f} "
                f"lam={new_lam:.4f}"
            )


def _make_env(env_config, lam_state, cost_scale, seed):
    inner = NetworkSlicingEnv(config=env_config)
    wrapped = LagrangianCostWrapper(
        inner, lam_state=lam_state, cost_scale=cost_scale
    )
    monitored = Monitor(wrapped)
    monitored.reset(seed=seed)
    return monitored


def train_ppo_lagrangian(
    seed: int = 42,
    cost_limit_raw: float = 70000.0,
    lr_lam: float = 5.0,
    cost_scale: float = 1e-5,
    lam_init: float = 0.0,
    lam_max: float = 100.0,
    total_timesteps: int = None,
    env_config: dict = None,
    verbose: int = 0,
    pid: bool = False,
    kp: float = 20.0,
    ki: float = 5.0,
    kd: float = 20.0,
    probe_every: int = 5,
):
    """Train PPO with primal-dual Lagrangian relaxation.

    All internal quantities (lam, ep_cost, cost_limit_internal,
    lr_lam) are in SCALED units (cost_scale=1e-5 multiplied) so the
    dual ascent dynamics are well-conditioned with lam, J_C, d ~ O(1).

    Args:
        cost_limit_raw: per-episode cumulative cost budget d in raw
            units. Empirical calibration (m=1, ENV_CONFIG):
              Static-Heuristic J_C_raw ~ 127k
              Max-Price J_C_raw ~ 88k
              Zero-Price J_C_raw ~ 128k
            Default 70k is below all three — constraint is binding.
            Scaled equivalent: 0.7.
        lr_lam: dual ascent step size in scaled units.
            Typical violation_scaled per rollout ~ 0.1-0.5; with
            lr_lam=5.0, lam grows by O(1-2) per rollout, reaching
            useful magnitudes (~tens) within tens of rollouts.
        cost_scale: rescales raw cost (~10^5 per ep) to O(1) range
            matching reward_scale.
        lam_init: initial Lagrange multiplier (scaled units).
    """
    cfg = env_config or ENV_CONFIG
    ts = total_timesteps or PPO_CONFIG["total_timesteps"]
    cost_limit_scaled = cost_limit_raw * cost_scale

    lam_state = [float(lam_init)]
    env = _make_env(cfg, lam_state, cost_scale, seed)

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=PPO_CONFIG["learning_rate"],
        batch_size=PPO_CONFIG["batch_size"],
        n_epochs=PPO_CONFIG["n_epochs"],
        clip_range=PPO_CONFIG["clip_range"],
        gae_lambda=PPO_CONFIG["gae_lambda"],
        gamma=PPO_CONFIG["gamma"],
        policy_kwargs=PPO_CONFIG["policy_kwargs"],
        seed=seed,
        verbose=0,
    )

    if pid:
        callback = PIDLagrangianCallback(
            lam_state=lam_state,
            cost_limit=cost_limit_scaled,
            kp=kp, ki=ki, kd=kd,
            lam_max=lam_max,
            probe_env_config=cfg,
            probe_every=probe_every,
            probe_cost_scale=cost_scale,
            integral_init=lam_init,
            verbose=verbose,
        )
    else:
        callback = DualAscentCallback(
            lam_state=lam_state,
            cost_limit=cost_limit_scaled,
            lr_lam=lr_lam,
            lam_max=lam_max,
            verbose=verbose,
        )

    print(
        f"\n{'='*60}\n"
        f"PPO-Lagrangian seed={seed} "
        f"cost_limit_raw={cost_limit_raw} (scaled={cost_limit_scaled})\n"
        f"  lr_lam={lr_lam} cost_scale={cost_scale} ts={ts}\n"
        f"{'='*60}"
    )
    model.learn(
        total_timesteps=ts, callback=callback,
        progress_bar=default_progress_bar()
    )
    return model, callback, lam_state


def evaluate_constrained(
    model, env_config, lam_state, cost_scale,
    n_episodes=20, seed=42,
):
    """Evaluate trained policy on the underlying env (lam=0, no cost
    subtraction in reward) so total_reward matches the unconstrained
    metric used elsewhere. Cost is read from info."""
    inner = NetworkSlicingEnv(config=env_config)
    eval_lam = [0.0]
    wrapped = LagrangianCostWrapper(
        inner, lam_state=eval_lam, cost_scale=cost_scale
    )
    rows = []
    for ep in range(n_episodes):
        obs, _ = wrapped.reset(seed=seed + ep)
        tot_r, tot_c = 0.0, 0.0
        tot_rev, tot_pen = 0.0, 0.0
        for _ in range(wrapped.env.T):
            a, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, info = wrapped.step(a)
            tot_r += r
            tot_c += info["cost"]
            tot_rev += info["revenue"]
            tot_pen += info["penalty"]
            if term or trunc:
                break
        rows.append({
            "ep": ep,
            "total_reward": tot_r,
            "total_cost": tot_c,
            "total_revenue": tot_rev,
            "total_penalty": tot_pen,
            "final_N_U": info["N_U"],
            "final_N_E": info["N_E"],
        })
    return rows


def main():
    """Smoke-test entry point. Production runs use run_lagrangian.py."""
    model, cb, lam_state = train_ppo_lagrangian(
        seed=42, total_timesteps=720 * 5, verbose=1,
    )
    rows = evaluate_constrained(model, ENV_CONFIG, lam_state, 1e-5, 2, 42)
    print(json.dumps(rows, indent=2, default=float))


if __name__ == "__main__":
    main()
