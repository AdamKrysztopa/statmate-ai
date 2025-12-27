"""Categorical comparison module for statistical tests."""

import numpy as np
import pandas as pd
import scipy.stats

from statmate.core.config import default_config
from statmate.core.exceptions import DataValidationError
from statmate.core.validation import validate_contingency_table
from statmate.statistical_core.base import StatTestResult


def chi2_test(
    contingency_table: np.ndarray | pd.DataFrame,
    alpha: float | None = None,
) -> StatTestResult:
    """Performs the Chi-Square test of independence on a contingency table.

    Args:
        contingency_table: Contingency table as 2D array or DataFrame.
        alpha: Significance level. If None, uses default from config.

    Returns:
        StatTestResult containing test outcomes.

    Raises:
        DataValidationError: If contingency table is invalid.

    Null hypothesis:
        The two categorical variables are independent.

    Alternative hypothesis:
        The two categorical variables are associated.
    """
    if alpha is None:
        alpha = default_config.statistical.default_alpha

    # Validate input
    validate_contingency_table(contingency_table, min_cell_count=5)

    results = scipy.stats.chi2_contingency(contingency_table)
    chi2 = float(results.statistic)
    p_value = float(results.pvalue)
    dof = int(results.dof)

    if p_value < alpha:
        result_text = (
            f'We must reject the null hypothesis (p = {p_value:.4f} < alpha = {alpha}); the variables are associated.'
        )
    else:
        result_text = (
            f'We cannot reject the null hypothesis (p = {p_value:.4f} >= alpha = {alpha}); '
            f'the variables appear independent.'
        )

    return StatTestResult(
        test_name='Chi-Square Test of Independence',
        statistics=chi2,
        p_value=p_value,
        null_hypothesis='The two categorical variables are independent.',
        alternative='The two categorical variables are associated.',
        statistical_test_results=result_text,
        test_specifics={'alpha': alpha, 'degrees_of_freedom': dof},
    )


def fisher_exact_test(
    table: np.ndarray | list[list[int]],
    alpha: float | None = None,
) -> StatTestResult:
    """Performs Fisher's Exact Test on a 2x2 contingency table.

    Args:
        table: 2x2 contingency table.
        alpha: Significance level. If None, uses default from config.

    Returns:
        StatTestResult containing test outcomes.

    Raises:
        DataValidationError: If table is not 2x2.

    Null hypothesis:
        There is no association between the two categorical variables.

    Alternative hypothesis:
        There is an association between the two categorical variables.
    """
    if alpha is None:
        alpha = default_config.statistical.default_alpha

    # Validate 2x2 table
    if isinstance(table, np.ndarray):
        if table.shape != (2, 2):
            raise DataValidationError(f"Fisher's exact test requires 2x2 table, got shape {table.shape}")
    elif isinstance(table, list):
        if len(table) != 2 or any(len(row) != 2 for row in table):
            raise DataValidationError("Fisher's exact test requires 2x2 table")

    results = scipy.stats.fisher_exact(table)
    statistic = float(results.statistic)
    p_value = float(results.pvalue)

    if p_value < alpha:
        result_text = (
            f'We must reject the null hypothesis (p = {p_value:.4f} < alpha = {alpha}); '
            f'there is evidence of association.'
        )
    else:
        result_text = (
            f'We cannot reject the null hypothesis (p = {p_value:.4f} >= alpha = {alpha}); '
            f'no evidence of association is found.'
        )

    return StatTestResult(
        test_name="Fisher's Exact Test",
        statistics=statistic,
        p_value=p_value,
        null_hypothesis='There is no association between the two categorical variables (in a 2x2 table).',
        alternative='There is an association between the two categorical variables.',
        statistical_test_results=result_text,
        test_specifics={'alpha': alpha, 'table': table},
    )


if __name__ == '__main__':
    from pprint import pprint

    # Categorical groups
    age_groups = ['<18', '18-24', '25-34', '35-44', '45-54', '55-64', '65-74', '75-84', '85-94', '95+']
    beverages = ['Coffee', 'Tea', 'Juice']

    np.random.seed(42)

    # DEPENDENT DATA: age groups prefer different beverages
    # Define "likelihood profiles" per age group
    preference_profiles = {
        '<18': [0.2, 0.4, 0.4],
        '18-24': [0.4, 0.4, 0.2],
        '25-34': [0.6, 0.3, 0.1],
        '35-44': [0.7, 0.2, 0.1],
        '45-54': [0.7, 0.2, 0.1],
        '55-64': [0.6, 0.3, 0.1],
        '65-74': [0.5, 0.4, 0.1],
        '75-84': [0.4, 0.5, 0.1],
        '85-94': [0.3, 0.6, 0.1],
        '95+': [0.2, 0.7, 0.1],
    }

    # Generate dependent data
    dependent_data = np.array([np.random.multinomial(100, preference_profiles[age_group]) for age_group in age_groups])

    # INDEPENDENT DATA: all age groups have the same distribution
    independent_data = np.tile([33, 33, 34], (len(age_groups), 1))

    # Create DataFrames
    dep_df = pd.DataFrame(dependent_data, index=age_groups, columns=beverages)
    dep_df.index.name = 'Age Group'

    ind_df = pd.DataFrame(independent_data, index=age_groups, columns=beverages)
    ind_df.index.name = 'Age Group'

    # Run Chi-Square Tests
    chi2_result_dep = chi2_test(dep_df)
    print('\nChi-Square Test Result (Dependent Data):')
    pprint(chi2_result_dep.model_dump())

    chi2_result_ind = chi2_test(ind_df)
    print('\nChi-Square Test Result (Independent Data):')
    pprint(chi2_result_ind.model_dump())

    # Fisher's Exact Test
    fisher_result = fisher_exact_test([[10, 20], [30, 40]])
    print('\nFisher Exact Test Result:')
    pprint(fisher_result)
