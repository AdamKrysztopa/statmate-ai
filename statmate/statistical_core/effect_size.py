"""Effect size helpers for all statistical test families."""

from __future__ import annotations

import numpy as np
from scipy import stats


def cohen_d(a: np.ndarray, b: np.ndarray) -> float:
    """Pooled-SD Cohen's d for two independent samples."""
    n1, n2 = len(a), len(b)
    pooled = np.sqrt(((n1 - 1) * np.var(a, ddof=1) + (n2 - 1) * np.var(b, ddof=1)) / (n1 + n2 - 2))
    if pooled == 0:
        return 0.0
    return float((np.mean(a) - np.mean(b)) / pooled)


def hedges_g(a: np.ndarray, b: np.ndarray) -> float:
    """Bias-corrected Cohen's d (Hedges' g)."""
    d = cohen_d(a, b)
    n = len(a) + len(b)
    correction = 1 - (3 / (4 * n - 9))
    return float(d * correction)


def eta_squared(f_stat: float, df_between: int, df_within: int) -> float:
    """Eta-squared from one-way ANOVA F-statistic."""
    ss_between = f_stat * df_between
    return float(ss_between / (ss_between + df_within))


def rank_biserial(u_stat: float, n1: int, n2: int) -> float:
    """Rank-biserial correlation for Mann-Whitney U."""
    return float(1 - (2 * u_stat) / (n1 * n2))


def wilcoxon_rank_biserial(statistic: float, n: int) -> float:
    """Rank-biserial correlation for Wilcoxon signed-rank (z approximation)."""
    # T+ statistic ranges from 0 to n*(n+1)/2
    max_t = n * (n + 1) / 2
    if max_t == 0:
        return 0.0
    return float(1 - (2 * statistic) / max_t)


def ci_mean_diff(a: np.ndarray, b: np.ndarray, alpha: float = 0.05) -> list[float]:
    """95% CI for the difference in means (Welch's method)."""
    result = stats.ttest_ind(a, b, equal_var=False)
    se = np.sqrt(np.var(a, ddof=1) / len(a) + np.var(b, ddof=1) / len(b))
    df = result.df  # type: ignore[union-attr]
    t_crit = stats.t.ppf(1 - alpha / 2, df=df)
    diff = float(np.mean(a) - np.mean(b))
    return [diff - float(t_crit * se), diff + float(t_crit * se)]


def ci_pearson(r: float, n: int, alpha: float = 0.05) -> list[float] | None:
    """Fisher Z-transform CI for Pearson r. Returns None when n <= 3."""
    if n <= 3:
        return None
    z = np.arctanh(r)
    se = 1 / np.sqrt(n - 3)
    z_crit = stats.norm.ppf(1 - alpha / 2)
    return [float(np.tanh(z - z_crit * se)), float(np.tanh(z + z_crit * se))]


def epsilon_squared_kruskal(h_stat: float, n_total: int) -> float:
    """Epsilon-squared effect size for Kruskal-Wallis H-test."""
    if n_total <= 1:
        return 0.0
    return float(h_stat / (n_total - 1))


def kendall_w_friedman(chi2_stat: float, n_subjects: int, k_conditions: int) -> float:
    """Kendall's W (concordance) from Friedman chi2 statistic."""
    denom = n_subjects * (k_conditions - 1)
    if denom == 0:
        return 0.0
    return float(chi2_stat / denom)
