"""ANOVA module for statistical tests."""

import numpy as np
import pandas as pd
import scipy.stats
from statsmodels.stats.anova import AnovaRM

from statmate.statistical_core.base import StatTestResult


def anova_one_way_test(*groups: np.ndarray, alpha: float = 0.05) -> StatTestResult:
    statistic, p_value = scipy.stats.f_oneway(*groups)
    if p_value < alpha:
        result_text = (
            f'We must reject the null hypothesis (p = {p_value:.4f} < α = {alpha}); at least one group mean differs.'
        )
    else:
        result_text = (
            f'We cannot reject the null hypothesis (p = {p_value:.4f} ≥ α = {alpha}); all group means appear equal.'
        )

    return StatTestResult(
        test_name='One-way ANOVA',
        statistics=statistic,
        p_value=p_value,
        null_hypothesis='All groups have equal means.',
        alternative='At least one group mean is different.',
        statistical_test_results=result_text,
        test_specifics={
            'alpha': alpha,
            'sample_sizes': [len(g) for g in groups],
            'number_of_groups': len(groups),
        },
    )


def anova_rm_test(
    input_data: pd.DataFrame, dependent_variable: str, subject: str, within: list[str], alpha: float = 0.05
) -> StatTestResult:
    # … validation as before …

    # ensure factors are categorical
    input_data[subject] = input_data[subject].astype('category')
    for col in within:
        input_data[col] = input_data[col].astype('category')

    model = AnovaRM(
        data=input_data,
        depvar=dependent_variable,
        subject=subject,
        within=within,
    ).fit()

    anova_table = model.anova_table
    first = anova_table.iloc[0]
    f_value, p_value = first['F Value'], first['Pr > F']

    if p_value < alpha:
        result_text = (
            f'We must reject the null hypothesis (p = {p_value:.4f} < α = {alpha}); '
            "at least one condition's mean is different."
        )
    else:
        result_text = (
            f'We cannot reject the null hypothesis (p = {p_value:.4f} ≥ α = {alpha}); all condition means appear equal.'
        )

    return StatTestResult(
        test_name='Repeated Measures ANOVA',
        statistics=f_value,
        p_value=p_value,
        null_hypothesis='The means across conditions (within‑subject factors) are equal.',
        alternative="At least one condition's mean is different.",
        statistical_test_results=result_text,
        test_specifics={
            'alpha': alpha,
            'dependent_variable': dependent_variable,
            'subject': subject,
            'within_factors': within,
            'anova_table': anova_table.to_dict(),
        },
    )


if __name__ == '__main__':
    # Example usage
    print('Running ANOVA tests...')
    print('One-way ANOVA Test:')
    data = np.random.rand(10, 3)  # Example data
    result = anova_one_way_test(data[:, 0], data[:, 1], data[:, 2])
    print(result)

    print('Repeated Measures ANOVA Test:')
    n_subj, n_cond = 10, 3
    subjects = np.repeat(np.arange(n_subj), n_cond)
    conditions = np.tile([f'C{i + 1}' for i in range(n_cond)], n_subj)
    values = np.random.randn(n_subj * n_cond)  # or your real data

    rm_df = pd.DataFrame({'subject': subjects, 'condition': conditions, 'value': values})

    # make factors categorical
    rm_df['subject'] = rm_df['subject'].astype('category')
    rm_df['condition'] = rm_df['condition'].astype('category')
    print(rm_df)
    # now this will work:
    rm_result = anova_rm_test(
        input_data=rm_df,
        dependent_variable='value',
        subject='subject',
        within=['condition'],
    )
    print(rm_result)
