"""Regression statistical tests.

This module provides OLS and logistic regression functions returning StatTestResult,
following the same pattern as comparison.py and anova.py.
"""

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler
from statsmodels.regression.linear_model import OLS
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools import add_constant

from statmate.statistical_core.base import StatTestResult


def linear_regression(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    alpha: float = 0.05,
) -> StatTestResult:
    """Simple or multiple OLS linear regression via statsmodels.

    Args:
        X: Predictor variables as a DataFrame.
        y: Target/response variable as a Series.
        alpha: Significance level for hypothesis decision text.

    Returns:
        StatTestResult with F-statistic, p-value, R², and coefficient table.
    """
    X_const = add_constant(X)
    model = OLS(y, X_const).fit()
    r_squared = float(model.rsquared)
    f_pvalue = float(model.f_pvalue)
    coef_table = {
        str(col): {
            'coef': float(model.params[col]),
            'p_value': float(model.pvalues[col]),
            'std_err': float(model.bse[col]),
        }
        for col in model.params.index
    }

    if f_pvalue < alpha:
        result_text = (
            f'We must reject the null hypothesis (p = {f_pvalue:.4f} < alpha = {alpha}). R² = {r_squared:.4f}.'
        )
    else:
        result_text = (
            f'We cannot reject the null hypothesis (p = {f_pvalue:.4f} >= alpha = {alpha}). R² = {r_squared:.4f}.'
        )

    return StatTestResult(
        test_name='linear_regression',
        statistics=float(model.fvalue),
        p_value=f_pvalue,
        null_hypothesis='All regression coefficients are zero.',
        alternative='At least one regression coefficient is non-zero.',
        statistical_test_results=result_text,
        test_specifics={
            'coefficient_table': coef_table,
            'r_squared': r_squared,
            'n_obs': int(model.nobs),
            'df_model': int(model.df_model),
        },
        effect_size_type='r_squared',
    )


def multiple_regression_with_vif(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    alpha: float = 0.05,
) -> StatTestResult:
    """OLS regression with VIF diagnostics for each predictor.

    Args:
        X: Predictor variables as a DataFrame (must have ≥ 2 columns).
        y: Target/response variable as a Series.
        alpha: Significance level for hypothesis decision text.

    Returns:
        StatTestResult with F-statistic, p-value, R², coefficient table, and VIF table.
    """
    X_const = add_constant(X)
    model = OLS(y, X_const).fit()
    r_squared = float(model.rsquared)
    f_pvalue = float(model.f_pvalue)
    coef_table = {
        str(col): {
            'coef': float(model.params[col]),
            'p_value': float(model.pvalues[col]),
            'std_err': float(model.bse[col]),
        }
        for col in model.params.index
    }

    # VIF: skip the constant column
    X_arr = X_const.values
    vif_table: dict[str, float] = {}
    for i, col in enumerate(X_const.columns):
        if str(col) == 'const':
            continue
        vif_table[str(col)] = float(variance_inflation_factor(X_arr, i))

    if f_pvalue < alpha:
        result_text = (
            f'We must reject the null hypothesis (p = {f_pvalue:.4f} < alpha = {alpha}). R² = {r_squared:.4f}.'
        )
    else:
        result_text = (
            f'We cannot reject the null hypothesis (p = {f_pvalue:.4f} >= alpha = {alpha}). R² = {r_squared:.4f}.'
        )

    return StatTestResult(
        test_name='multiple_regression',
        statistics=float(model.fvalue),
        p_value=f_pvalue,
        null_hypothesis='All regression coefficients are zero.',
        alternative='At least one regression coefficient is non-zero.',
        statistical_test_results=result_text,
        test_specifics={
            'coefficient_table': coef_table,
            'vif_table': vif_table,
            'r_squared': r_squared,
            'n_obs': int(model.nobs),
            'df_model': int(model.df_model),
        },
        effect_size_type='r_squared',
    )


def logistic_regression(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    alpha: float = 0.05,
) -> StatTestResult:
    """Binary logistic regression via sklearn; returns log-odds coefficients.

    Uses a likelihood-ratio chi² test for the omnibus p-value and
    Nagelkerke pseudo-R² as the effect-size metric.

    Args:
        X: Predictor variables as a DataFrame.
        y: Binary target variable (0/1) as a Series.
        alpha: Significance level for hypothesis decision text.

    Returns:
        StatTestResult with LR chi² statistic, p-value, Nagelkerke R², and
        log-odds coefficient table.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    clf = LogisticRegression(max_iter=1000, solver='lbfgs')
    clf.fit(X_scaled, y)

    n = len(y)
    null_proba = float(y.mean())
    null_log_loss = log_loss(y, [null_proba] * n)
    model_log_loss = log_loss(y, clf.predict_proba(X_scaled))

    # Cox-Snell R²
    cox_snell = 1.0 - np.exp(-2.0 * n * (null_log_loss - model_log_loss))
    # Nagelkerke R² (scaled by maximum possible Cox-Snell)
    max_cox_snell = 1.0 - np.exp((-2.0 / n) * (n * null_log_loss))
    nagelkerke = float(np.clip(cox_snell / max_cox_snell if max_cox_snell > 0 else 0.0, 0.0, 1.0))

    # Likelihood-ratio test statistic
    lr_stat = 2.0 * n * (null_log_loss - model_log_loss)
    p_value = float(scipy_stats.chi2.sf(lr_stat, df=X.shape[1]))

    coef_table = {str(col): float(coef) for col, coef in zip(X.columns, clf.coef_[0], strict=True)}

    if p_value < alpha:
        result_text = (
            f'We must reject the null hypothesis (p = {p_value:.4f} < alpha = {alpha}). '
            f'Nagelkerke R² = {nagelkerke:.4f}.'
        )
    else:
        result_text = (
            f'We cannot reject the null hypothesis (p = {p_value:.4f} >= alpha = {alpha}). '
            f'Nagelkerke R² = {nagelkerke:.4f}.'
        )

    return StatTestResult(
        test_name='logistic_regression',
        statistics=float(lr_stat),
        p_value=p_value,
        null_hypothesis='All regression coefficients are zero.',
        alternative='At least one regression coefficient is non-zero.',
        statistical_test_results=result_text,
        test_specifics={
            'log_odds_table': coef_table,
            'nagelkerke_r2': nagelkerke,
            'n_obs': n,
        },
        effect_size_type='nagelkerke_r2',
    )
