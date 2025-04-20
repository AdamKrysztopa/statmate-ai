import numpy as np
import pandas as pd
import scipy.stats

from statmate.statistical_core.base import StatTestResult


def chi2_test(contingency_table: np.ndarray | pd.DataFrame, alpha: float = 0.05) -> StatTestResult:
    """Performs the Chi-Square test of independence on a contingency table.

    Null hypothesis:
        The two categorical variables are independent.

    Alternative hypothesis:
        The two categorical variables are associated.
    """
    results = scipy.stats.chi2_contingency(contingency_table)
    chi2 = results.statistic  # type: ignore # not true
    p_value = results.pvalue  # type: ignore # not true
    dof = results.dof  # type: ignore # not true
    # expected = results.expected  # type: ignore # not true

    if p_value < alpha:
        result_text = (
            f'We must reject the null hypothesis (p = {p_value:.4f} < alpha = {alpha}); the variables are associated.'
        )
    else:
        result_text = f'We cannot reject the null hypothesis (p = {p_value:.4f} >= alpha = {alpha}); the variables appear independent.'

    return StatTestResult(
        test_name='Chi-Square Test of Independence',
        statistics=chi2,
        p_value=p_value,
        null_hypothesis='The two categorical variables are independent.',
        alternative='The two categorical variables are associated.',
        statistical_test_results=result_text,
        test_specifics={'alpha': alpha, 'degrees_of_freedom': dof},
    )


# 11. Fisher Exact Test (typically for 2x2 tables)
def fisher_exact_test(table: np.ndarray | list[list[int]], alpha: float = 0.05) -> StatTestResult:
    """Performs Fisher's Exact Test on a 2x2 contingency table.

    Null hypothesis:
        There is no association between the two categorical variables.

    Alternative hypothesis:
        There is an association between the two categorical variables.
    """
    results = scipy.stats.fisher_exact(table)
    statistic = results.statistic  # type: ignore # not true
    p_value = results.pvalue  # type: ignore # not true

    if p_value < alpha:
        result_text = f'We must reject the null hypothesis (p = {p_value:.4f} < alpha = {alpha}); there is evidence of association.'
    else:
        result_text = f'We cannot reject the null hypothesis (p = {p_value:.4f} >= alpha = {alpha}); no evidence of association is found.'

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

    import numpy as np
    import pandas as pd

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
    base_prob = [1 / len(beverages)] * len(beverages)
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
