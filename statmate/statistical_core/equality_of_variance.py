"""Equality of variance module."""

import numpy as np
import scipy.stats

from statmate.statistical_core.base import StatTestResult


# 7. Bartlett’s Test (Two Independent Samples, Parametric)
def bartlett_test(data1: np.ndarray, data2: np.ndarray, alpha: float = 0.05) -> StatTestResult:
    """Performs Bartlett’s test for equality of variances between two samples.

    Null hypothesis:
        The two independent samples have equal variances.

    Alternative hypothesis:
        The two independent samples have different variances.
    """
    statistic, p_value = scipy.stats.bartlett(data1, data2)
    if p_value < alpha:
        result_text = f'We must reject the null hypothesis (p = {p_value:.4f} < alpha = {alpha}).'
    else:
        result_text = f'We cannot reject the null hypothesis (p = {p_value:.4f} >= alpha = {alpha}).'

    return StatTestResult(
        test_name="Bartlett's test",
        statistics=statistic,
        p_value=p_value,
        null_hypothesis='The two independent samples have equal variances.',
        alternative='The two independent samples have different variances.',
        statistical_test_results=result_text,
        test_specifics={
            'alpha': alpha,
            'sample_size_1': len(data1),
            'sample_size_2': len(data2),
        },
    )


# 8. Levene’s Test (Two Independent Samples, Non-parametric)
def levene_test(
    data1: np.ndarray,
    data2: np.ndarray,
    center: str = 'median',
    alpha: float = 0.05,
) -> StatTestResult:
    """Performs Levene’s test for equality of variances between two samples.

    Null hypothesis:
        The two independent samples have equal variances.

    Alternative hypothesis:
        The two independent samples have different variances.
    """
    statistic, p_value = scipy.stats.levene(data1, data2, center=center)
    if p_value < alpha:
        result_text = f'We must reject the null hypothesis (p = {p_value:.4f} < alpha = {alpha}).'
    else:
        result_text = f'We cannot reject the null hypothesis (p = {p_value:.4f} >= alpha = {alpha}).'

    return StatTestResult(
        test_name="Levene's test",
        statistics=statistic,
        p_value=p_value,
        null_hypothesis='The two independent samples have equal variances.',
        alternative='The two independent samples have different variances.',
        statistical_test_results=result_text,
        test_specifics={
            'alpha': alpha,
            'center': center,
            'sample_size_1': len(data1),
            'sample_size_2': len(data2),
        },
    )
