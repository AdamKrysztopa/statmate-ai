"""Unit tests for the regression statistical functions."""

import numpy as np
import pandas as pd

from statmate.statistical_core.regression import (
    linear_regression,
    logistic_regression,
    multiple_regression_with_vif,
)


def test_linear_regression_simple() -> None:
    rng = np.random.default_rng(42)
    x = rng.uniform(0, 10, 100)
    y = pd.Series(2 * x + rng.normal(0, 0.5, 100))
    X = pd.DataFrame({'x': x})
    result = linear_regression(X, y)
    assert result.p_value < 0.05
    assert result.effect_size_type == 'r_squared'
    assert result.statistics > 0


def test_multiple_regression_vif() -> None:
    rng = np.random.default_rng(0)
    x1 = rng.uniform(0, 5, 80)
    x2 = rng.uniform(0, 5, 80)
    y = pd.Series(1.5 * x1 - 0.8 * x2 + rng.normal(0, 0.3, 80))
    X = pd.DataFrame({'x1': x1, 'x2': x2})
    result = multiple_regression_with_vif(X, y)
    assert result.p_value < 0.05
    assert result.test_specifics is not None
    assert 'vif_table' in result.test_specifics
    assert len(result.test_specifics['vif_table']) == 2  # x1 and x2


def test_logistic_regression_correct_sign() -> None:
    rng = np.random.default_rng(7)
    x = rng.uniform(-3, 3, 200)
    y = pd.Series((x > 0).astype(int))
    X = pd.DataFrame({'x': x})
    result = logistic_regression(X, y)
    assert result.test_specifics is not None
    assert result.test_specifics['log_odds_table']['x'] > 0  # positive x → class 1
    assert result.effect_size_type == 'nagelkerke_r2'
