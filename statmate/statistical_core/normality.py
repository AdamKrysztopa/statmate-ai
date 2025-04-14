"""Shapiro-Wilk normality test."""

import numpy as np
import scipy.stats
from base import StatTestResult


def shapiro_wilk_test(
    data: np.ndarray, alpha: float = 0.05, nan_policy: str = 'propagate'
) -> StatTestResult:
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
        result_text = (
            f'We must reject the null hypothesis (p = {p_value:.4f} < alpha = {alpha}).'
        )
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
