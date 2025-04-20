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


# 4. Independent t-test (Two Independent Samples)
def ttest_ind_test(data1: np.ndarray, data2: np.ndarray, alpha: float = 0.05, equal_var: bool = True) -> StatTestResult:
    """Performs an independent t-test on two independent samples.

    Null hypothesis:
        The two independent samples have equal means.

    Alternative hypothesis:
        The two independent samples have different means.
    """
    results = scipy.stats.ttest_ind(data1, data2, equal_var=equal_var, nan_policy='propagate')
    statistic = results.statistic  # type: ignore # not true
    p_value = results.pvalue  # type: ignore # not true

    test_type = "Student's t-test" if equal_var else "Welch's t-test"
    if p_value < alpha:
        result_text = f'We must reject the null hypothesis (p = {p_value:.4f} < alpha = {alpha}) using {test_type}.'
    else:
        result_text = f'We cannot reject the null hypothesis (p = {p_value:.4f} >= alpha = {alpha}) using {test_type}.'

    return StatTestResult(
        test_name=test_type,
        statistics=statistic,
        p_value=p_value,
        null_hypothesis='The two independent samples have equal means.',
        alternative='The two independent samples have different means.',
        statistical_test_results=result_text,
        test_specifics={
            'alpha': alpha,
            'equal_var': equal_var,
            'sample_size_1': len(data1),
            'sample_size_2': len(data2),
        },
    )


# 5. Mann–Whitney U Test (Two Independent Samples, Non-parametric)
def mannwhitneyu_test(
    data1: np.ndarray, data2: np.ndarray, alpha: float = 0.05, alternative: str = 'two-sided'
) -> StatTestResult:
    """Performs the Mann–Whitney U test for two independent samples.

    Null hypothesis:
        The distributions of the two independent samples are equal.

    Alternative hypothesis:
        The distributions of the two independent samples are not equal.
    """
    statistic, p_value = scipy.stats.mannwhitneyu(data1, data2, alternative=alternative)
    if p_value < alpha:
        result_text = f'We must reject the null hypothesis (p = {p_value:.4f} < alpha = {alpha}).'
    else:
        result_text = f'We cannot reject the null hypothesis (p = {p_value:.4f} >= alpha = {alpha}).'

    return StatTestResult(
        test_name='Mann-Whitney U Test',
        statistics=statistic,
        p_value=p_value,
        null_hypothesis='The two independent samples come from identical distributions.',
        alternative='The two independent samples come from different distributions.',
        statistical_test_results=result_text,
        test_specifics={
            'alpha': alpha,
            'alternative': alternative,
            'sample_size_1': len(data1),
            'sample_size_2': len(data2),
        },
    )


# 6 Welch's t-test (Two Independent Samples, Non-parametric)
def welch_t_test(data1: np.ndarray, data2: np.ndarray, alpha: float = 0.05) -> StatTestResult:
    """Performs Welch's t-test for two independent samples with unequal variances.

    Args:
        data1 (np.ndarray): First independent sample.
        data2 (np.ndarray): Second independent sample.
        alpha (float, optional): Significance level. Defaults to 0.05.

    Returns:
        StatTestResult: A standardized result object containing:
            - test_name: "Welch's t-test"
            - statistics: The test statistic.
            - p_value: The p-value for the test.
            - null_hypothesis: "The two independent samples have equal means."
            - alternative: "The two independent samples have different means."
            - statistical_test_results: A human-readable decision string.
            - test_specifics: A dictionary of extra details like alpha and sample sizes.
    """
    results = scipy.stats.ttest_ind(data1, data2, equal_var=False, nan_policy='propagate')
    statistic = results.statistic  # type: ignore # not true
    p_value = results.pvalue  # type: ignore # not true

    if p_value < alpha:
        result_text = f'We must reject the null hypothesis (p = {p_value:.4f} < alpha = {alpha}); '
        'means are significantly different.'
    else:
        result_text = f'We cannot reject the null hypothesis (p = {p_value:.4f} >= alpha = {alpha}); '
        'means are not significantly different.'

    return StatTestResult(
        test_name="Welch's t-test",
        statistics=statistic,
        p_value=p_value,
        null_hypothesis='The two independent samples have equal means.',
        alternative='The two independent samples have different means.',
        statistical_test_results=result_text,
        test_specifics={
            'alpha': alpha,
            'sample_size_1': len(data1),
            'sample_size_2': len(data2),
            'equal_variance_assumed': False,
        },
    )
