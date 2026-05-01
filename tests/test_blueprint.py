"""Unit tests for DataBlueprint construction and influence diagnostics."""
import numpy as np
import pandas as pd

from statmate.workflow.blueprint import InfluenceDiagnostics, build_data_blueprint


def test_influence_diagnostics_populated_with_outlier():
    rng = np.random.default_rng(42)
    n = 60
    x = rng.uniform(0, 10, n)
    y = 3 * x + rng.normal(0, 0.5, n)
    # Inject one obvious outlier
    y[0] = y.mean() + 20 * y.std()
    df = pd.DataFrame({'x': x, 'y': y})
    blueprint = build_data_blueprint(df, target_column='y')
    assert blueprint.influence_diagnostics is not None
    assert blueprint.influence_diagnostics.outlier_count_iqr >= 1


def test_influence_diagnostics_none_without_target():
    df = pd.DataFrame({'a': [1.0, 2.0, 3.0], 'b': [4.0, 5.0, 6.0]})
    blueprint = build_data_blueprint(df)
    # No target_column → influence_diagnostics should be None
    assert blueprint.influence_diagnostics is None


def test_influence_diagnostics_type():
    rng = np.random.default_rng(1)
    x = rng.uniform(0, 5, 50)
    y = 2 * x + rng.normal(0, 0.3, 50)
    df = pd.DataFrame({'predictor': x, 'outcome': y})
    blueprint = build_data_blueprint(df, target_column='outcome')
    diag = blueprint.influence_diagnostics
    assert diag is not None
    assert isinstance(diag, InfluenceDiagnostics)
    assert diag.cooks_distance_max is not None
    assert diag.cooks_distance_max >= 0.0
    assert diag.high_leverage_count is not None
    assert diag.outlier_count_iqr is not None
