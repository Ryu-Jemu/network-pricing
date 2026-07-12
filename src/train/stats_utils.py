"""Statistical utilities for multi-seed RL evaluation.

Implements:
  - Bootstrap percentile CI for the mean
  - Welch's t-test (unequal variances) — works on small samples
  - Permutation test (Pitman / randomization) — distribution-free
  - Cohen's d (effect size)

References:
  Henderson, Islam, Bachman, Pineau, Precup, Meger.
    "Deep Reinforcement Learning that Matters." AAAI 2018.
    arXiv:1709.06560.
  Colas, Sigaud, Oudeyer. "How Many Random Seeds? Statistical Power
    Analysis in Deep Reinforcement Learning Experiments." 2018.
  Agarwal, Schwarzer, Castro, Courville, Bellemare. "Deep
    Reinforcement Learning at the Edge of the Statistical
    Precipice." NeurIPS 2021. arXiv:2108.13264.

Ported from improvement-3-seed-power branch (commit 313a618) for use
in the main churn-sweep figure pipeline.
"""
import numpy as np
from scipy import stats


def bootstrap_mean_ci(samples, n_boot=5000, alpha=0.05, rng=None):
    """Returns (mean, ci_low, ci_high) with percentile bootstrap."""
    rng = rng or np.random.default_rng(0)
    samples = np.asarray(samples, dtype=float)
    n = len(samples)
    if n < 2:
        m = float(np.mean(samples)) if n else 0.0
        return m, m, m
    boots = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots[i] = float(np.mean(samples[idx]))
    lo = float(np.percentile(boots, 100 * alpha / 2))
    hi = float(np.percentile(boots, 100 * (1 - alpha / 2)))
    return float(np.mean(samples)), lo, hi


def welch_t_test(a, b):
    """Welch's t-test (unequal-variance two-sample). Returns
    (t_stat, p_value, df_satterthwaite)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) < 2 or len(b) < 2:
        return float("nan"), float("nan"), float("nan")
    res = stats.ttest_ind(a, b, equal_var=False)
    sa2, sb2 = a.var(ddof=1), b.var(ddof=1)
    na, nb = len(a), len(b)
    num = (sa2 / na + sb2 / nb) ** 2
    den = (
        (sa2 / na) ** 2 / max(na - 1, 1)
        + (sb2 / nb) ** 2 / max(nb - 1, 1)
    )
    df = num / den if den > 0 else float("nan")
    return float(res.statistic), float(res.pvalue), float(df)


def permutation_test(a, b, n_perm=10000, rng=None):
    """Pitman permutation test on the difference of means.
    Two-sided p-value with add-one smoothing."""
    rng = rng or np.random.default_rng(0)
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    pooled = np.concatenate([a, b])
    n_a = len(a)
    obs = abs(np.mean(a) - np.mean(b))
    count = 0
    for _ in range(n_perm):
        rng.shuffle(pooled)
        m_a = pooled[:n_a].mean()
        m_b = pooled[n_a:].mean()
        if abs(m_a - m_b) >= obs:
            count += 1
    return float((count + 1) / (n_perm + 1))


def cohens_d(a, b):
    """Cohen's d with pooled SD."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    pooled_sd = np.sqrt(
        ((len(a) - 1) * a.var(ddof=1)
         + (len(b) - 1) * b.var(ddof=1))
        / (len(a) + len(b) - 2)
    )
    if pooled_sd == 0:
        return float("nan")
    return float((np.mean(a) - np.mean(b)) / pooled_sd)


def summarize_seed_means(seed_means, baseline_means=None,
                         n_boot=5000, alpha=0.05, label=""):
    """Returns a dict with mean/sd/CI for `seed_means` plus comparison
    statistics vs `baseline_means` if given."""
    rng = np.random.default_rng(0)
    mean, lo, hi = bootstrap_mean_ci(
        seed_means, n_boot=n_boot, alpha=alpha, rng=rng,
    )
    out = {
        "label": label,
        "n_seeds": len(seed_means),
        "mean": mean,
        "sd": float(np.std(seed_means, ddof=1)) if len(seed_means) > 1 else 0.0,
        "se": (
            float(np.std(seed_means, ddof=1) / np.sqrt(len(seed_means)))
            if len(seed_means) > 1 else 0.0
        ),
        "ci_low": lo,
        "ci_high": hi,
        "alpha": alpha,
    }
    if baseline_means is not None:
        t_stat, p_t, df_t = welch_t_test(seed_means, baseline_means)
        p_perm = permutation_test(seed_means, baseline_means)
        d = cohens_d(seed_means, baseline_means)
        out["vs_baseline"] = {
            "welch_t": t_stat,
            "welch_p": p_t,
            "welch_df": df_t,
            "perm_p": p_perm,
            "cohens_d": d,
            "diff_mean": float(np.mean(seed_means) - np.mean(baseline_means)),
        }
    return out
