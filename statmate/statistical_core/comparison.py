"""Comparison module for statistical tests."""

import numpy as np
import scipy.stats

from statmate.statistical_core.base import StatTestResult


# 2. Paired t-test (Dependent Samples)
def ttest_rel_test(data1: np.ndarray, data2: np.ndarray, alpha: float = 0.05) -> StatTestResult:
    """Performs a paired t-test on two related samples.

    Null hypothesis:
        The mean difference between paired samples is zero.

    Alternative hypothesis:
        The mean difference between paired samples is not zero.
    """
    statistic, p_value = scipy.stats.ttest_rel(data1, data2, nan_policy='propagate')
    if p_value < alpha:
        result_text = f'We must reject the null hypothesis (p = {p_value:.4f} < alpha = {alpha}).'
    else:
        result_text = f'We cannot reject the null hypothesis (p = {p_value:.4f} >= alpha = {alpha}).'

    return StatTestResult(
        test_name='Paired t-test',
        statistics=statistic,
        p_value=p_value,
        null_hypothesis='The mean difference between paired samples is zero.',
        alternative='The mean difference between paired samples is not zero.',
        statistical_test_results=result_text,
        test_specifics={'alpha': alpha, 'sample_size': len(data1)},
    )


# 3. Wilcoxon Signed-Rank Test (Paired, Non-parametric)
def wilcoxon_test(data1: np.ndarray, data2: np.ndarray, alpha: float = 0.05) -> StatTestResult:
    """Performs the Wilcoxon signed-rank test for paired samples.

    Null hypothesis:
        The distribution of the differences between paired samples is symmetric about zero.

    Alternative hypothesis:
        The distribution of the differences is not symmetric about zero.
    """
    results = scipy.stats.wilcoxon(data1, data2, zero_method='wilcox', correction=False)
    statistic = results.statistic  # type: ignore # not true
    p_value = results.pvalue  # type: ignore # not true
    if p_value < alpha:
        result_text = f'We must reject the null hypothesis (p = {p_value:.4f} < alpha = {alpha}).'
    else:
        result_text = f'We cannot reject the null hypothesis (p = {p_value:.4f} >= alpha = {alpha}).'

    return StatTestResult(
        test_name='Wilcoxon Signed-Rank Test',
        statistics=statistic,
        p_value=p_value,
        null_hypothesis='The distribution of the differences between paired samples is symmetric about zero.',
        alternative='The distribution of the differences between paired samples is not symmetric about zero.',
        statistical_test_results=result_text,
        test_specifics={'alpha': alpha, 'sample_size': len(data1)},
    )
