import numpy as np
from scipy.stats import pearsonr, spearmanr

from statmate.statistical_core.base import StatTestResult


def pearson_corr(data: np.ndarray, data2: np.ndarray, alpha: float = 0.05) -> StatTestResult:
    """Performs Pearson’s correlation test between two variables.

    Null hypothesis:
        No linear correlation between the two variables.

    Alternative hypothesis:
        There is a linear correlation between the two variables.

    Args:
        data: First sample of observations.
        data2: Second sample of observations.
        alpha: Significance level for the test.

    Returns:
        A StatTestResult containing statistic, p-value, decision text, and details.
    """
    corr_coef, p_value = pearsonr(data, data2)
    if p_value < alpha:  # type: ignore # not true
        result_text = (
            f'Reject H₀ (p = {p_value:.4f} < α = {alpha}); '
            f'r = {corr_coef:.4f} indicates significant linear correlation.'
        )
    else:
        result_text = (
            f'Fail to reject H₀ (p = {p_value:.4f} ≥ α = {alpha}); '
            f'r = {corr_coef:.4f} indicates no significant linear correlation.'
        )

    return StatTestResult(
        test_name='Pearson’s correlation',
        statistics=corr_coef,  # type: ignore # not true
        p_value=p_value,  # type: ignore # not true
        null_hypothesis='No linear correlation between the two variables.',
        alternative='There is a linear correlation between the two variables.',
        statistical_test_results=result_text,
        test_specifics={
            'alpha': alpha,
            'sample_size_1': len(data),
            'sample_size_2': len(data2),
        },
    )


def spearman_corr(data: np.ndarray, data2: np.ndarray, alpha: float = 0.05) -> StatTestResult:
    """Performs Spearman’s rank correlation test between two variables.

    Null hypothesis:
        No monotonic association between the two variables.

    Alternative hypothesis:
        There is a monotonic association between the two variables.

    Args:
        data: First sample of observations.
        data2: Second sample of observations.
        alpha: Significance level for the test.

    Returns:
        A StatTestResult containing statistic, p-value, decision text, and details.
    """
    corr_coef, p_value = spearmanr(data, data2)
    if p_value < alpha:  # type: ignore # not true
        result_text = (
            f'Reject Null hypothesis (p = {p_value:.4f} < alpha = {alpha}); '
            f'corr_coeff = {corr_coef:.4f} indicates significant monotonic association.'
        )
    else:
        result_text = (
            f'Fail to reject Null  (p = {p_value:.4f} ≥ alpha = {alpha}); '
            f'corr_coeff = {corr_coef:.4f} indicates no significant monotonic association.'
        )

    return StatTestResult(
        test_name='Spearman’s correlation',
        statistics=corr_coef,  # type: ignore # not true
        p_value=p_value,  # type: ignore # not true
        null_hypothesis='No monotonic association between the two variables.',
        alternative='There is a monotonic association between the two variables.',
        statistical_test_results=result_text,
        test_specifics={
            'alpha': alpha,
            'sample_size_1': len(data),
            'sample_size_2': len(data2),
        },
    )
