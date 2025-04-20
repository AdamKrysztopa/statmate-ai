"""Shapiro-Wilk normality test."""

import warnings
from collections.abc import Callable
from itertools import product
from typing import Any

import numpy as np
import pandas as pd
import scipy.stats
from joblib import Parallel, delayed
from scipy.stats import anderson, cramervonmises, jarque_bera, kstest, normaltest
from statsmodels.stats.diagnostic import lilliefors
from tqdm import tqdm

from statmate.statistical_core.base import StatTestResult

# Suppress specific warnings
warnings.filterwarnings('ignore', category=UserWarning, module='scipy.stats._axis_nan_policy')
warnings.filterwarnings('ignore', category=RuntimeWarning, module='scipy.stats._morestats')


def shapiro_wilk_test(data: np.ndarray, alpha: float = 0.05, nan_policy: str = 'propagate') -> StatTestResult:
    """Performs the Shapiro-Wilk normality test.

    Null hypothesis:
        The sample is drawn from a normal distribution.

    Alternative hypothesis:
        The sample is not drawn from a normal distribution.

    Args:
        data (np.ndarray): Input data to test.
        alpha (float, optional): Significance level. Defaults to 0.05.
        nan_policy (str, optional): Defines how to handle NaN values.
            Options are 'propagate', 'raise', or 'omit'. Defaults to 'propagate'.

    Returns:
        StatTestResult: Result of the Shapiro-Wilk normality test.
    """
    statistic, p_value = scipy.stats.shapiro(data, nan_policy=nan_policy)  # type: ignore not true
    if p_value < alpha:
        result_text = f'We must reject the null hypothesis (p = {p_value:.4f} < alpha = {alpha}).'
    else:
        result_text = f'We cannot reject the null hypothesis (p = {p_value:.4f} >= alpha = {alpha}).'

    return StatTestResult(
        test_name='Shapiro-Wilk Normality Test',
        statistics=statistic,
        p_value=p_value,
        null_hypothesis='The sample is drawn from a normal distribution.',
        alternative='The sample is not drawn from a normal distribution.',
        statistical_test_results=result_text,
        test_specifics={
            'alpha': alpha,
            'nan_policy': nan_policy,
            'sample_size': len(data),
        },
    )


def normality_of_difference(
    data1: np.ndarray,
    data2: np.ndarray,
    alpha: float = 0.05,
    nan_policy: str = 'propagate',
) -> StatTestResult:
    """Performs the Shapiro-Wilk normality test on the difference between two samples.

    Null hypothesis:
        The difference between the two samples is drawn from a normal distribution.

    Alternative hypothesis:
        The difference between the two samples is not drawn from a normal distribution.
    """
    difference = data1 - data2
    results = shapiro_wilk_test(difference, alpha=alpha, nan_policy=nan_policy)
    results.test_name = 'Normality of Difference Test'
    results.null_hypothesis = 'The difference between the two samples is drawn from a normal distribution.'
    results.alternative = 'The difference between the two samples is not drawn from a normal distribution.'

    return results


def anderson_darling_test(data: np.ndarray, alpha: float = 0.05) -> StatTestResult:
    """Anderson-Darling test for normality.

    Null hypothesis:
        The sample is drawn from a normal distribution.
    Alternative hypothesis:
        The sample is not drawn from a normal distribution.

    Args:
        data (np.ndarray): Input data to test.
        alpha (float, optional): Significance level. Defaults to 0.05.

    Returns:
        StatTestResult: Result of the Anderson-Darling normality test.
    """
    res = anderson(data, dist='norm')
    stat = res.statistic  # type: ignore # not true
    # pick critical value for the given α (levels are in percent)
    cv = next(
        cv
        for perc, cv in zip(res.significance_level, res.critical_values, strict=False)  # type: ignore # not true
        if np.isclose(perc / 100, alpha)
    )
    decision = stat > cv
    text = (
        f'Reject H0 (A squared={stat:.3f} > CV={cv:.3f} at alpha={alpha})'
        if decision
        else f'Fail to reject H0 (A squared={stat:.3f} ≤ CV={cv:.3f})'
    )
    # estimate p-value
    # find the upper index of the critical value
    idx = np.searchsorted(res.critical_values, stat, side='right')  # type: ignore # not true
    p_value = res.significance_level[idx] / 100 if idx < len(res.critical_values) else 0.0  # type: ignore # not true

    return StatTestResult(
        test_name='Anderson-Darling Test',
        statistics=stat,
        p_value=p_value,
        null_hypothesis='Sample from a normal distribution.',
        alternative='Sample not from a normal distribution.',
        statistical_test_results=text,
        test_specifics={'alpha': alpha, 'critical_value': cv, 'sample_size': len(data)},
    )


def dagostino_pearson_test(data: np.ndarray, alpha: float = 0.05) -> StatTestResult:
    """D`Agostino-Pearson Omnibus Test for normality.

    Null hypothesis:
        The sample is drawn from a normal distribution.
    Alternative hypothesis:
        The sample is not drawn from a normal distribution.

    Args:
        data (np.ndarray): Input data to test.
        alpha (float, optional): Significance level. Defaults to 0.05.

    Returns:
        StatTestResult: Result of the D`Agostino-Pearson Omnibus Test.
    """
    stat, p = normaltest(data, nan_policy='propagate')
    decision = p < alpha
    text = f'Reject H0 (p={p:.4f}<alpha={alpha})' if decision else f'Fail to reject H0 (p={p:.4f}≥alpha={alpha})'
    return StatTestResult(
        test_name='D`Agostino-Pearson K squared Test',
        statistics=stat,
        p_value=p,
        null_hypothesis='Sample is normally distributed.',
        alternative='Sample deviates from normality.',
        statistical_test_results=text,
        test_specifics={'alpha': alpha, 'sample_size': len(data)},
    )


def lilliefors_test(data: np.ndarray, alpha: float = 0.05) -> StatTestResult:
    """Lilliefors test for normality.

    Null hypothesis:
        The sample is drawn from a normal distribution.
    Alternative hypothesis:
        The sample is not drawn from a normal distribution.

    Args:
        data (np.ndarray): Input data to test.
        alpha (float, optional): Significance level. Defaults to 0.05.

    Returns:
        StatTestResult: Result of the Lilliefors test.
    """
    stat, p = lilliefors(data, dist='norm')
    decision = p < alpha
    text = f'Reject H0 (p={p:.4f}<alpha={alpha})' if decision else f'Fail to reject H0 (p={p:.4f}≥alpha={alpha})'
    return StatTestResult(
        test_name='Lilliefors Test',
        statistics=stat,
        p_value=p,  # type: ignore # not true
        null_hypothesis='Data follow a normal distribution (parameters estimated).',
        alternative='Data do not follow a normal distribution.',
        statistical_test_results=text,
        test_specifics={'alpha': alpha, 'sample_size': len(data)},
    )


def jarque_bera_test(data: np.ndarray, alpha: float = 0.05) -> StatTestResult:
    """Jarque-Bera test for normality.

    Null hypothesis:
        The sample is drawn from a normal distribution.
    Alternative hypothesis:
        The sample is not drawn from a normal distribution.

    Args:
        data (np.ndarray): Input data to test.
        alpha (float, optional): Significance level. Defaults to 0.05.

    Returns:
        StatTestResult: Result of the Jarque-Bera test.
    """
    stat, p = jarque_bera(data)
    p = float(p)  # type: ignore # not true
    decision = p < alpha
    text = f'Reject H0 (p={p:.4f}<alpha={alpha})' if decision else f'Fail to reject H0 (p={p:.4f}≥alpha={alpha})'
    return StatTestResult(
        test_name='Jarque-Bera Test',
        statistics=stat,  # type: ignore # not true
        p_value=p,
        null_hypothesis='Data have skewness=0 and kurtosis=3 (normal).',
        alternative='Data skewness≠0 or kurtosis≠3.',
        statistical_test_results=text,
        test_specifics={'alpha': alpha, 'sample_size': len(data)},
    )


def cramer_von_mises_test(data: np.ndarray, alpha: float = 0.05) -> StatTestResult:
    """Cramér-von Mises test for normality.

    Null hypothesis:
        The sample is drawn from a normal distribution.
    Alternative hypothesis:
        The sample is not drawn from a normal distribution.

    Args:
        data (np.ndarray): Input data to test.
        alpha (float, optional): Significance level. Defaults to 0.05.

    Returns:
        StatTestResult: Result of the Cramér-von Mises test.
    """
    res = cramervonmises(data, cdf='norm')
    stat, p = res.statistic, res.pvalue
    decision = p < alpha
    text = f'Reject H0 (p={p:.4f}<aplha={alpha})' if decision else f'Fail to reject H0 (p={p:.4f}≥alpha={alpha})'
    return StatTestResult(
        test_name='Cramér-von Mises Test',
        statistics=stat,
        p_value=p,  # type: ignore # not true
        null_hypothesis='Data from a normal distribution.',
        alternative='Data not from a normal distribution.',
        statistical_test_results=text,
        test_specifics={'alpha': alpha, 'sample_size': len(data)},
    )


def ks_normal_test(data: np.ndarray, alpha: float = 0.05) -> StatTestResult:
    """Kolmogorov-Smirnov test for normality.

    Null hypothesis:
        The sample is drawn from a normal distribution.
    Alternative hypothesis:
        The sample is not drawn from a normal distribution.

    Args:
        data (np.ndarray): Input data to test.
        alpha (float, optional): Significance level. Defaults to 0.05.

    Returns:
        StatTestResult: Result of the Kolmogorov-Smirnov test.
    """
    mu, sigma = np.mean(data), np.std(data, ddof=1)
    stat, p = kstest(data, 'norm', args=(mu, sigma))
    decision = p < alpha
    text = f'Reject H0 (p={p:.4f}<alpha={alpha})' if decision else f'Fail to reject H0 (p={p:.4f}≥alpha={alpha})'
    return StatTestResult(
        test_name='Kolmogorov-Smirnov One-Sample Test',
        statistics=stat,
        p_value=p,
        null_hypothesis='Data come from N(mu,sigma_squared) with mu,sigma estimated.',
        alternative='Data not from that normal model.',
        statistical_test_results=text,
        test_specifics={'alpha': alpha, 'mu_est': mu, 'sigma_est': sigma, 'sample_size': len(data)},
    )


def run_monte_carlo(
    tests: dict[str, Callable[[np.ndarray, float, str], Any]],
    distributions: dict[str, Callable[..., np.ndarray]],
    dist_params: dict[str, dict[str, Any]],
    sample_sizes: int | list[int],
    n_trials: int = 1000,
    alpha: float = 0.05,
    random_state: int | None = None,
    n_jobs: int = 1,
) -> pd.DataFrame:
    """Generic Monte Carlo simulation for benchmarking statistical tests.

    Args:
        tests: Mapping of test names to functions accepting (data, alpha, nan_policy) and returning
               an object with attribute `p_value` indicating the test p-value.
        distributions: Mapping of distribution names to functions creating samples.
                       Each function signature: (**params, size) -> np.ndarray.
        dist_params: Parameters for each distribution (keys match distributions dict).
        sample_sizes: Single int or list of sample sizes to evaluate.
        n_trials: Number of Monte Carlo replications per configuration.
        alpha: Significance level.
        nan_policy: Passed to test functions (e.g. 'omit').
        random_state: Seed for reproducibility.
        n_jobs: Number of parallel jobs (requires joblib).

    Returns:
        DataFrame with columns:
            - test: name of the statistical test
            - distribution: distribution name
            - sample_size: sample size
            - rejection_rate: fraction of trials rejecting H0
            - trials: total trials run
    """
    rng = np.random.default_rng(random_state)
    # Ensure sample_sizes is list
    sizes = [sample_sizes] if isinstance(sample_sizes, int) else sample_sizes

    results = []

    def _single_run(test_name: str, dist_name: str, size: int) -> bool:
        """Run a single Monte Carlo trial."""
        # Generate data
        data = distributions[dist_name](**dist_params.get(dist_name, {}), size=size)
        # Run test
        result = tests[test_name](data, alpha=alpha)  # type: ignore # to be verified
        return result.p_value < alpha if not np.isnan(result.p_value) else False

    for test_name, dist_name, size in tqdm(
        product(tests, distributions, sizes), total=len(tests) * len(distributions) * len(sizes)
    ):
        # Pre-generate seeds for reproducibility across parallel jobs
        seeds = rng.integers(0, 1_000_000_000, size=n_trials)
        # Run trials
        if n_jobs == 1:
            rejects = [_single_run(test_name, dist_name, size) for _ in seeds]
            # cast elements to int
            rejects = [int(reject) for reject in rejects if reject is not None]
        else:
            rejects = Parallel(n_jobs=n_jobs)(delayed(_single_run)(test_name, dist_name, size) for _ in seeds)
            # cast to list
            rejects = [int(reject) if reject is not None else 0 for reject in rejects]
        rejection_rate = np.mean(rejects)
        results.append(
            {
                'test': test_name,
                'distribution': dist_name,
                'sample_size': size,
                'rejection_rate': rejection_rate,
                'trials': n_trials,
            }
        )

    return pd.DataFrame(results)


if __name__ == '__main__':
    # Example usage:
    # from monte_carlo_framework import run_monte_carlo
    tests = {'shapiro': shapiro_wilk_test, 'anderson': anderson_darling_test}
    distributions = {
        'normal': lambda loc, scale, size: np.random.default_rng().normal(loc, scale, size),
        'uniform': lambda low, high, size: np.random.default_rng().uniform(low, high, size),
        'exponential': lambda scale, size: np.random.default_rng().exponential(scale, size),
        'poisson': lambda lam, size: np.random.default_rng().poisson(lam, size),
        'binomial': lambda n, p, size: np.random.default_rng().binomial(n, p, size),
    }
    dist_params = {
        'normal': {'loc': 0, 'scale': 1},
        'uniform': {'low': -1, 'high': 1},
        'exponential': {'scale': 1},
        'poisson': {'lam': 5},
        'binomial': {'n': 10, 'p': 0.5},
    }
    df = run_monte_carlo(
        tests,
        distributions,
        dist_params,
        sample_sizes=[5, 10, 20, 50, 100, 10000],
        n_trials=10000,
        alpha=0.05,
        random_state=42,
        n_jobs=-1,
    )

    print(df)
