"""Categorical comparison module for statistical tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.stats
from statsmodels.stats.contingency_tables import Table, mcnemar

from statmate.core.config import default_config
from statmate.core.exceptions import DataValidationError
from statmate.statistical_core.base import StatTestResult
from statmate.statistical_core.categorical import (
    categorical_effect_size,
    has_small_expected_counts,
    is_2x2_table,
    prepare_contingency_table,
    phi_coefficient,
)


def chi2_test(
    contingency_table: np.ndarray | pd.DataFrame | list[list[int]],
    alpha: float | None = None,
) -> StatTestResult:
    """Perform the Chi-Square test of independence on a contingency table."""
    if alpha is None:
        alpha = default_config.statistical.default_alpha

    table = prepare_contingency_table(contingency_table)

    if has_small_expected_counts(table, threshold=5.0):
        if is_2x2_table(table):
            fisher_result = fisher_exact_test(table, alpha=alpha)
            specifics = fisher_result.test_specifics or {}
            specifics['note'] = "Used Fisher's Exact Test because expected frequencies were below 5."
            fisher_result.test_specifics = specifics
            return fisher_result
        raise DataValidationError(
            "Chi-square test requires expected frequencies >= 5 for all cells; consider combining categories or "
            "using an exact test."
        )

    chi2, p_value, dof, expected = scipy.stats.chi2_contingency(table, correction=False)
    effect_size, effect_label = categorical_effect_size(table, chi2)
    effect_text = (
        f" Effect size ({effect_label.replace('_', ' ').title()}): {effect_size:.3f}."
        if effect_size is not None and effect_label
        else ''
    )

    if p_value < alpha:
        result_text = (
            f'We must reject the null hypothesis (p = {p_value:.4f} < alpha = {alpha}); '
            f'the variables are associated.{effect_text}'
        )
    else:
        result_text = (
            f'We cannot reject the null hypothesis (p = {p_value:.4f} >= alpha = {alpha}); '
            f'the variables appear independent.{effect_text}'
        )

    test_specifics: dict[str, object] = {
        'alpha': alpha,
        'degrees_of_freedom': dof,
        'expected_min': float(np.min(expected)),
        'table_shape': table.shape,
    }
    if effect_size is not None and effect_label:
        test_specifics['effect_size'] = {'metric': effect_label, 'value': effect_size}

    return StatTestResult(
        test_name='Chi-Square Test of Independence',
        statistics=float(chi2),
        p_value=float(p_value),
        null_hypothesis='The two categorical variables are independent.',
        alternative='The two categorical variables are associated.',
        statistical_test_results=result_text,
        test_specifics=test_specifics,
        effect_size_type=effect_label,
    )


def fisher_exact_test(
    table: np.ndarray | list[list[int]] | pd.DataFrame,
    alpha: float | None = None,
) -> StatTestResult:
    """Perform Fisher's Exact Test on a 2x2 contingency table, with Phi effect size."""
    if alpha is None:
        alpha = default_config.statistical.default_alpha

    prepared = prepare_contingency_table(table)
    if prepared.shape != (2, 2):
        raise DataValidationError(f"Fisher's exact test requires 2x2 table, got shape {prepared.shape}")

    odds_ratio, p_value = scipy.stats.fisher_exact(prepared)
    effect_size = phi_coefficient(prepared)
    effect_text = (
        f" Effect size (Phi): {effect_size:.3f}." if effect_size is not None else ''
    )

    if p_value < alpha:
        result_text = (
            f'We must reject the null hypothesis (p = {p_value:.4f} < alpha = {alpha}); '
            f'there is evidence of association.{effect_text}'
        )
    else:
        result_text = (
            f'We cannot reject the null hypothesis (p = {p_value:.4f} >= alpha = {alpha}); '
            f'no evidence of association is found.{effect_text}'
        )

    test_specifics: dict[str, object] = {
        'alpha': alpha,
        'odds_ratio': float(odds_ratio),
        'table': prepared.tolist(),
    }
    if effect_size is not None:
        test_specifics['effect_size'] = {'metric': 'phi', 'value': effect_size}

    return StatTestResult(
        test_name="Fisher's Exact Test",
        statistics=float(odds_ratio),
        p_value=float(p_value),
        null_hypothesis='There is no association between the two categorical variables (in a 2x2 table).',
        alternative='There is an association between the two categorical variables.',
        statistical_test_results=result_text,
        test_specifics=test_specifics,
        effect_size_type='phi' if effect_size is not None else None,
    )


def mcnemar_test(
    table: np.ndarray | list[list[int]] | pd.DataFrame,
    alpha: float | None = None,
    exact: bool | None = None,
    correction: bool = True,
    column_pair: tuple[str, str] | None = None,
) -> StatTestResult:
    """Perform McNemar's Test for paired categorical data (2x2 table)."""
    if alpha is None:
        alpha = default_config.statistical.default_alpha

    prepared: np.ndarray
    if isinstance(table, pd.DataFrame) and table.shape != (2, 2):
        cols = list(column_pair) if column_pair else list(table.columns[:2])
        if len(cols) < 2:
            raise DataValidationError("McNemar's test requires two categorical columns to form pairs.")
        if any(col not in table.columns for col in cols[:2]):
            raise DataValidationError(f'Missing required columns for paired comparison: {cols[:2]}')
        contingency = pd.crosstab(table[cols[0]], table[cols[1]])
        if contingency.shape != (2, 2):
            raise DataValidationError(
                f"McNemar's test expects binary categories; found table of shape {contingency.shape}."
            )
        prepared = contingency.to_numpy()
    else:
        prepared = prepare_contingency_table(table)

    prepared = prepare_contingency_table(prepared)
    if prepared.shape != (2, 2):
        raise DataValidationError(f"McNemar's test requires a 2x2 table, got shape {prepared.shape}")

    b, c = float(prepared[0, 1]), float(prepared[1, 0])
    if exact is None:
        exact = b + c < 25

    result = mcnemar(prepared, exact=exact, correction=correction)
    statistic = float(result.statistic)
    p_value = float(result.pvalue)
    effect_size = phi_coefficient(prepared)
    effect_text = f' Effect size (Phi): {effect_size:.3f}.' if effect_size is not None else ''

    if p_value < alpha:
        result_text = (
            f'We must reject the null hypothesis (p = {p_value:.4f} < alpha = {alpha}); '
            f'the paired proportions differ.{effect_text}'
        )
    else:
        result_text = (
            f'We cannot reject the null hypothesis (p = {p_value:.4f} >= alpha = {alpha}); '
            f'the paired proportions appear equal.{effect_text}'
        )

    test_specifics: dict[str, object] = {
        'alpha': alpha,
        'exact': exact,
        'correction': False if exact else correction,
        'discordant_pairs': {'b': b, 'c': c},
        'total_pairs': float(np.sum(prepared)),
    }
    if effect_size is not None:
        test_specifics['effect_size'] = {'metric': 'phi', 'value': effect_size}

    return StatTestResult(
        test_name="McNemar's Test",
        statistics=statistic,
        p_value=p_value,
        null_hypothesis='The paired proportions are equal.',
        alternative='The paired proportions are not equal.',
        statistical_test_results=result_text,
        test_specifics=test_specifics,
        effect_size_type='phi' if effect_size is not None else None,
    )


def cochran_armitage_trend_test(
    table: np.ndarray | list[list[int]] | pd.DataFrame,
    alpha: float | None = None,
    row_scores: list[float] | None = None,
    col_scores: list[float] | None = None,
    outcome_col: str | None = None,
    group_col: str | None = None,
    alternative: str = 'two-sided',
) -> StatTestResult:
    """Perform the Cochran-Armitage trend test for ordered categorical variables."""
    if alpha is None:
        alpha = default_config.statistical.default_alpha

    alt = alternative.lower()
    if alt not in {'two-sided', 'less', 'greater'}:
        raise DataValidationError(f'Invalid alternative "{alternative}". Use two-sided, less, or greater.')

    if isinstance(table, pd.DataFrame) and table.shape != (2, 2):
        outcome_field = outcome_col or table.columns[0]
        group_field = group_col or table.columns[1]
        if outcome_field not in table.columns or group_field not in table.columns:
            raise DataValidationError('Outcome or group column not found for trend test.')
        contingency = pd.crosstab(table[outcome_field], table[group_field])
        prepared = contingency.to_numpy()
        resolved_row_scores = row_scores or list(range(prepared.shape[0]))
        resolved_col_scores = col_scores or list(range(prepared.shape[1]))
    else:
        prepared = prepare_contingency_table(table)
        resolved_row_scores = row_scores or list(range(prepared.shape[0]))
        resolved_col_scores = col_scores or list(range(prepared.shape[1]))

    if prepared.shape[0] != 2:
        raise DataValidationError('Cochran-Armitage trend test expects a 2xK table (binary outcome by ordered group).')

    row_scores_arr = np.asarray(resolved_row_scores, dtype=float)
    col_scores_arr = np.asarray(resolved_col_scores, dtype=float)
    res = Table(prepared).test_ordinal_association(row_scores=row_scores_arr, col_scores=col_scores_arr)
    z_score = float(res.zscore)
    if alt == 'less':
        p_value = float(scipy.stats.norm.cdf(z_score))
    elif alt == 'greater':
        p_value = float(1 - scipy.stats.norm.cdf(z_score))
    else:
        p_value = float(2 * scipy.stats.norm.cdf(-abs(z_score)))

    direction = 'there is a monotonic trend.' if p_value < alpha else 'no monotonic trend detected.'
    if alt == 'greater':
        alt_text = 'Proportion increases with the ordered predictor.'
    elif alt == 'less':
        alt_text = 'Proportion decreases with the ordered predictor.'
    else:
        alt_text = 'There is a monotonic trend with the ordered predictor.'

    result_text = (
        f'Z = {z_score:.3f}, p = {p_value:.4f}. '
        f'{direction}'
    )

    test_specifics: dict[str, object] = {
        'alpha': alpha,
        'row_scores': row_scores_arr.tolist(),
        'col_scores': col_scores_arr.tolist(),
        'statistic_raw': float(res.statistic),
        'null_mean': float(res.null_mean),
        'null_sd': float(res.null_sd),
    }

    return StatTestResult(
        test_name='Cochran-Armitage Trend Test',
        statistics=z_score,
        p_value=p_value,
        null_hypothesis='There is no linear trend in proportions across ordered categories.',
        alternative=alt_text,
        statistical_test_results=result_text,
        test_specifics=test_specifics,
    )
