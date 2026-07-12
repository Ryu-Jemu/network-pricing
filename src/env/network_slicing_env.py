"""
5G Network Slicing Dynamic Pricing Environment
================================================
Gymnasium environment implementing the MDP from Model Spec (F1-F11).

Slices: URLLC (s=0) and eMBB (s=1)
State:  (N_U, N_E, eta_U_prev, eta_E_prev) -- 4-dim (kept unchanged)
Action: (F_U, p_U, F_E, p_E)               -- 4-dim continuous [0,1] scaled
Reward: r_t = Revenue - QoS Penalty

Theoretical foundations
-----------------------
- Departure/arrival logits implement a binary discrete-choice model with random
  utility, equivalent to McFadden (1974) "Conditional Logit Analysis of
  Qualitative Choice Behavior" (Frontiers in Econometrics; Nobel 2000).
- The 3PT charging structure (F + max(0, q-Q̄)*p) extends Oi (1971)
  "A Disneyland Dilemma: Two-Part Tariffs for a Mickey Mouse Monopoly"
  (QJE 85(1):77-96) by adding a free allowance Q̄.
- The dynamic pricing MDP formulation is the discrete-time analogue of
  Stokey (1979) "Intertemporal Price Discrimination" (QJE 93(3):355-371).
- LogNormal usage follows Alasmar et al. (2021) IEEE/ACM Trans. Netw. 29(3).
- Logistic churn follows Ahn, Han & Lee (2006) Telecom. Policy 30:552-568.

Telecom + Economics Optional Extensions (Phase 11, OFF by default)
------------------------------------------------------------------
When `cohort_aware=True`, the departure logit is augmented with:
  - Asymmetric per-slice price sensitivities γ_{F,s}, γ_{p,s}
    (Gerpott, Rams & Schindler 2001 Telecom. Policy 25(4):249-269;
     Tirole 1988 *The Theory of Industrial Organization* ch. 3.3 on
     third-degree price discrimination).
  - Tenure cohort Cox proportional-hazard term −α·ln(k+1) over K buckets
    (Cox 1972 JRSS B 34(2):187-220; Bolton 1998 *Marketing Science*
     17(1):45-65 — longer relationship → lower departure hazard via
     accumulated satisfaction).
  - Klemperer (1987) *Economic Journal* 97(Supp):99-117 switching-cost
    term −β·ln(τ̄_s+1) using the cohort-weighted mean tenure as a
    proxy for the operator's installed lock-in.
With defaults (cohort_aware=False, n_cohorts=1, alpha_tenure=0, beta_sc=0,
γ_F_per_slice=[1.0,1.0], γ_p_per_slice=[0.8,0.8]) the env is numerically
identical to the published-paper environment.
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class NetworkSlicingEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, config=None):
        super().__init__()
        cfg = config or {}

        # --- Slices: index 0 = URLLC, index 1 = eMBB ---
        self.n_slices = 2

        # Tariff reference (normalization)
        self.F_ref = np.array(cfg.get("F_ref", [50.0, 30.0]))
        self.p_ref = np.array(cfg.get("p_ref", [10.0, 5.0]))
        self.Q_bar = np.array(cfg.get("Q_bar", [5.0, 30.0]))

        # Action bounds
        self.F_max = np.array(cfg.get("F_max", [100.0, 100.0]))
        self.p_max = np.array(cfg.get("p_max", [20.0, 20.0]))

        # Traffic (LogNormal parameters)
        self.mu = np.array(cfg.get("mu", [1.0, 3.0]))
        self.sigma2 = np.array(cfg.get("sigma2", [0.5, 0.8]))
        self.sigma = np.sqrt(self.sigma2)

        # Departure coefficients (F3)
        self.gamma0 = np.array(cfg.get("gamma0", [-10.13, -12.53]))
        # Slice-scalar (original published paper) — kept for backward compat
        self.gamma_F_scalar = cfg.get("gamma_F", 1.0)
        self.gamma_p_scalar = cfg.get("gamma_p", 0.8)
        self.gamma_eta = np.array(cfg.get("gamma_eta", [3.0, 0.5]))

        # Arrival coefficients (F4, F5)
        self.beta0 = np.array(cfg.get("beta0", [2.0, 2.5]))
        self.beta_F = cfg.get("beta_F", 0.8)
        self.beta_p = cfg.get("beta_p", 0.6)
        self.lambda_max = np.array(cfg.get("lambda_max", [0.05, 0.15]))

        # QoS parameters (F7)
        self.eta_low = np.array(cfg.get("eta_low", [0.90, 0.80]))
        self.eta_high = np.array(cfg.get("eta_high", [1.0, 1.0]))
        self.eta_tgt = np.array(cfg.get("eta_tgt", [0.99999, 0.90]))

        # Endogenous QoS (improvement-2 branch, OFF by default).
        # eta_s = clip(1 - alpha_s * max(0, util_s - rho_star_s)
        #              + Uniform(-delta_s, +delta_s), 0, 1)
        # util_s = total realised load / capacity_s
        # See Khani et al. 2024 (Slice admission control: a survey;
        # 10.1002/dac.5857) and Lin et al. 2025 (HWEL Rule;
        # 10.1038/s41598-025-17385-4) for load->QoS linkages used in
        # 5G slice resource models.
        self.qos_endogenous = bool(cfg.get("qos_endogenous", False))
        # Capacity calibrated so reference policy gives util ~ 0.70:
        #   E[load_U]=N_U_init*E[q_U]=1000*3.49=3490 -> C_U=5000
        #   E[load_E]=N_E_init*E[q_E]=5000*29.96=149800 -> C_E=215000
        self.capacity = np.array(cfg.get("capacity", [5000.0, 215000.0]))
        self.rho_star = np.array(cfg.get("rho_star", [0.50, 0.50]))
        # Degradation slope: alpha_U > alpha_E (URLLC more sensitive).
        self.alpha_qos = np.array(cfg.get("alpha_qos", [0.45, 0.30]))
        # Residual noise (measurement jitter), preserves stochastic
        # character of original Uniform formulation.
        self.delta_qos = np.array(cfg.get("delta_qos", [0.01, 0.02]))

        # CMDP cost signal threshold (improvement-1 branch). The constraint
        # cost uses eta_sla, which defaults to eta_tgt (legacy behaviour) but
        # may be calibrated separately so the constraint is feasible under
        # measurement jitter without touching the reward-penalty target.
        self.eta_sla = np.array(cfg.get("eta_sla", list(self.eta_tgt)))

        # Penalty weights (F9)
        self.w = np.array(cfg.get("w", [500.0, 50.0]))

        # MDP parameters
        self.T = cfg.get("T", 720)
        self.discount = cfg.get("gamma", 0.99)

        # Initial state
        self.N_init = np.array(cfg.get("N_init", [1000.0, 5000.0]))
        self.eta_init = np.array(cfg.get("eta_init", [0.95, 0.90]))

        # Normalization constants (eliminate scale imbalance).
        self.reward_scale = cfg.get("reward_scale", 1e-5)

        # ---- Phase 11: telecom + econ extension flags (OFF by default) ----
        self.cohort_aware = bool(cfg.get("cohort_aware", False))
        # Slice-level price sensitivity arrays (Gerpott 2001, Tirole 1988)
        self.gamma_F_per_slice = np.array(cfg.get(
            "gamma_F_per_slice",
            [self.gamma_F_scalar, self.gamma_F_scalar],
        ), dtype=np.float64)
        self.gamma_p_per_slice = np.array(cfg.get(
            "gamma_p_per_slice",
            [self.gamma_p_scalar, self.gamma_p_scalar],
        ), dtype=np.float64)
        # Tenure cohort settings (Cox 1972, Bolton 1998)
        self.n_cohorts = int(cfg.get("n_cohorts", 1))
        # cohort_bins_months[k] = nominal midpoint of bucket k (months).
        # length must equal n_cohorts. Used only for τ̄ computation; the
        # cohort index k itself enters the Cox term via ln(k+1).
        default_bins = [0.5, 2.0, 4.5, 9.0, 18.0, 36.0][:max(self.n_cohorts, 1)]
        self.cohort_bins_months = np.array(cfg.get(
            "cohort_bins_months",
            default_bins,
        ), dtype=np.float64)
        assert len(self.cohort_bins_months) == self.n_cohorts, (
            f"cohort_bins_months length {len(self.cohort_bins_months)} "
            f"must equal n_cohorts {self.n_cohorts}"
        )
        # Default cohort initial distribution: all in cohort 0 if n_cohorts==1,
        # otherwise the synthetic distribution from Plan 11.2.
        if self.n_cohorts == 1:
            default_cohort_init_arr = np.array(
                [[float(self.N_init[0])], [float(self.N_init[1])]],
                dtype=np.float64,
            )
        else:
            default_cohort_init_arr = np.array(
                [
                    [50.0, 100.0, 200.0, 300.0, 250.0, 100.0][:self.n_cohorts],
                    [500.0, 1000.0, 1500.0, 1000.0, 700.0, 300.0][:self.n_cohorts],
                ],
                dtype=np.float64,
            )
        self.cohort_init = np.array(
            cfg.get("cohort_init", default_cohort_init_arr.tolist()),
            dtype=np.float64,
        )
        assert self.cohort_init.shape == (2, self.n_cohorts), (
            f"cohort_init shape {self.cohort_init.shape} "
            f"must be (2, {self.n_cohorts})"
        )
        # Hazard coefficients (Cox PH tenure, Klemperer SC)
        self.alpha_tenure = float(cfg.get("alpha_tenure", 0.0))
        self.beta_sc = float(cfg.get("beta_sc", 0.0))

        # --- Spaces ---
        # Action: 4-dim continuous [0, 1], scaled to actual values in step()
        self.action_space = spaces.Box(
            low=0.0, high=1.0, shape=(4,), dtype=np.float32
        )

        # State: (N_U/N_init_U, N_E/N_init_E, eta_U_prev, eta_E_prev)
        # All dimensions normalized to ~[0, 2] range
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([10.0, 10.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        self.reset()

    def _sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

    def _scale_action(self, action):
        """Scale [0,1] action to actual F and p values."""
        F = np.array([action[0] * self.F_max[0], action[2] * self.F_max[1]])
        p = np.array([action[1] * self.p_max[0], action[3] * self.p_max[1]])
        return F, p

    def _mean_tenure_months(self):
        """Cohort-weighted mean tenure per slice (months). Returns shape (2,).

        Used as a proxy for installed switching cost (Klemperer 1987): longer
        average tenure → larger lock-in → lower departure hazard.
        """
        totals = self.N_cohort.sum(axis=1)
        out = np.zeros(2)
        for s in range(2):
            if totals[s] > 0:
                out[s] = float(np.dot(
                    self.N_cohort[s], self.cohort_bins_months
                ) / totals[s])
        return out

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Cohort state (always allocated; n_cohorts=1 reduces to scalar N)
        self.N_cohort = self.cohort_init.copy().astype(np.float64)
        self.N = self.N_cohort.sum(axis=1)
        self.eta_prev = self.eta_init.copy().astype(np.float64)
        self.t = 0
        self.info_log = {}
        return self._get_obs(), {}

    def _get_obs(self):
        # Normalize N by initial values to eliminate scale imbalance
        return np.array(
            [self.N[0] / self.N_init[0],
             self.N[1] / self.N_init[1],
             self.eta_prev[0],
             self.eta_prev[1]],
            dtype=np.float32,
        )

    def step(self, action):
        self.t += 1
        F, p = self._scale_action(action)

        # Normalized prices (for departure/arrival formulas)
        F_tilde = F / self.F_ref
        p_tilde = p / self.p_ref

        # ============ Phase B: Subscriber Dynamics ============

        # Step 3: Departure (F3) — extended with Cox cohort + Klemperer SC
        # in cohort_aware mode; reduces to the published-paper logit when
        # gamma_F_per_slice = [γ_F, γ_F], alpha_tenure = 0, beta_sc = 0,
        # n_cohorts = 1.
        mean_tau = self._mean_tenure_months()  # shape (2,)
        # Base per-slice logit (without tenure cohort term)
        base_logit = (
            self.gamma0
            + self.gamma_F_per_slice * F_tilde
            + self.gamma_p_per_slice * p_tilde
            - self.gamma_eta * self.eta_prev
            - self.beta_sc * np.log(mean_tau + 1.0)
        )  # shape (2,)

        # Cohort departures
        N_leave_cohort = np.zeros_like(self.N_cohort)
        for s in range(2):
            for k in range(self.n_cohorts):
                if self.N_cohort[s, k] <= 0:
                    continue
                logit_sk = base_logit[s] - self.alpha_tenure * np.log(k + 1.0)
                P_dep_sk = float(self._sigmoid(logit_sk))
                N_leave_cohort[s, k] = self.np_random.binomial(
                    int(self.N_cohort[s, k]), P_dep_sk
                )
        N_leave = N_leave_cohort.sum(axis=1)  # aggregate per slice
        N_surv_cohort = self.N_cohort - N_leave_cohort

        # Aggregate per-slice P_dep (for logging only; weighted by cohort pop)
        with np.errstate(invalid="ignore", divide="ignore"):
            P_dep = np.where(
                self.N > 0,
                N_leave / np.maximum(self.N, 1.0),
                self._sigmoid(base_logit),
            )

        # Step 4: Arrival (F4, F5)
        arr_logit = self.beta0 - self.beta_F * F_tilde - self.beta_p * p_tilde
        P_arr = self._sigmoid(arr_logit)
        lam = self.lambda_max * P_arr
        N_new = np.array(
            [self.np_random.poisson(float(lam[s])) for s in range(2)],
            dtype=np.float64,
        )

        # New arrivals enter cohort 0
        self.N_cohort = N_surv_cohort.copy()
        self.N_cohort[:, 0] += N_new

        # Active users (F6)
        N_active = self.N_cohort.sum(axis=1)

        # ============ Phase C: Usage and QoS ============

        # Step 5: Usage generation and billing (F1, F2)
        total_revenue = 0.0
        load = np.zeros(2)
        for s in range(2):
            n = int(N_active[s])
            if n > 0:
                # Generate usage for all active users
                usage = self.np_random.lognormal(
                    mean=float(self.mu[s]),
                    sigma=float(self.sigma[s]),
                    size=n,
                )
                # Compute bills (F1)
                overage = np.maximum(0.0, usage - self.Q_bar[s])
                bills = F[s] + overage * p[s]
                total_revenue += np.sum(bills)
                load[s] = float(np.sum(usage))

        # Step 6: QoS realization (F7).
        # Endogenous mode closes the price -> subscribers -> load -> QoS
        # loop; exogenous mode (default) reproduces the published-paper
        # Uniform draw with identical RNG consumption order.
        if self.qos_endogenous:
            util = load / self.capacity
            shortfall_load = np.maximum(0.0, util - self.rho_star)
            eta_det = 1.0 - self.alpha_qos * shortfall_load
            jitter = np.array([
                self.np_random.uniform(
                    -float(self.delta_qos[s]),
                    +float(self.delta_qos[s]),
                )
                for s in range(2)
            ])
            eta = np.clip(eta_det + jitter, 0.0, 1.0)
        else:
            eta = np.array([
                self.np_random.uniform(
                    float(self.eta_low[s]), float(self.eta_high[s])
                )
                for s in range(2)
            ])

        # ============ Phase D: Learning Signal ============

        # Step 7: Reward (F8-F10) — unchanged; CLV / shaping is audit-only.
        penalty = 0.0
        for s in range(2):
            shortfall = max(0.0, self.eta_tgt[s] - eta[s])
            penalty += self.w[s] * N_active[s] * shortfall

        reward = (total_revenue - penalty) * self.reward_scale

        # CMDP cost signal (improvement-1 branch): unweighted SLA-shortfall
        # user-hours against eta_sla. Always emitted via info; the reward
        # above is untouched. See SafeSlice-style constrained slicing.
        cost_U = max(0.0, self.eta_sla[0] - eta[0]) * float(N_active[0])
        cost_E = max(0.0, self.eta_sla[1] - eta[1]) * float(N_active[1])

        # Step 8: State transition (F11)
        self.N = N_active.copy()
        self.eta_prev = eta.copy()

        terminated = self.t >= self.T
        truncated = False

        info = {
            "revenue": total_revenue,
            "penalty": penalty,
            "N_U": N_active[0],
            "N_E": N_active[1],
            "eta_U": eta[0],
            "eta_E": eta[1],
            "F_U": F[0],
            "p_U": p[0],
            "F_E": F[1],
            "p_E": p[1],
            "P_dep_U": P_dep[0],
            "P_dep_E": P_dep[1],
            "N_leave_U": N_leave[0],
            "N_leave_E": N_leave[1],
            "N_new_U": N_new[0],
            "N_new_E": N_new[1],
            "mean_tenure_U_months": mean_tau[0],
            "mean_tenure_E_months": mean_tau[1],
            # Endogenous-QoS diagnostics (improvement-2 branch)
            "load_U": load[0],
            "load_E": load[1],
            "util_U": (
                load[0] / self.capacity[0] if self.qos_endogenous else 0.0
            ),
            "util_E": (
                load[1] / self.capacity[1] if self.qos_endogenous else 0.0
            ),
            # CMDP cost (improvement-1 branch). Original reward unchanged.
            "cost_U": cost_U,
            "cost_E": cost_E,
            "cost": cost_U + cost_E,
        }

        return self._get_obs(), float(reward), terminated, truncated, info
