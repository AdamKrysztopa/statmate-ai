import numpy as np
import pandas as pd
import pytest

from statmate.core.config import NodeName
from statmate.statistical_core.anova import friedman_test, kruskal_wallis_test
from statmate.statistical_core.categorical_comparison import (
    chi2_test,
    cochran_armitage_trend_test,
    mcnemar_test,
)
from statmate.workflow.blueprint import DataBlueprint, DistributionMetric
from statmate.workflow.edges import decision_engine
from statmate.workflow.state import create_initial_state


def test_kruskal_wallis_includes_dunn_posthoc():
    rng = np.random.default_rng(42)
    group1 = rng.normal(0.0, 1.0, 30)
    group2 = rng.normal(0.5, 1.0, 30)
    group3 = rng.normal(2.5, 1.0, 30)

    result = kruskal_wallis_test(group1, group2, group3, alpha=0.05, group_labels=['A', 'B', 'C'])
    assert result.p_value < 0.05
    posthoc = (result.test_specifics or {}).get('posthoc') or []
    assert any(item.get('reject') for item in posthoc), 'Dunn post-hoc should flag a difference'


def test_friedman_detects_condition_effects():
    rng = np.random.default_rng(0)
    subjects = np.repeat(np.arange(10), 3)
    conditions = np.tile(['A', 'B', 'C'], 10)
    base = np.tile([1.0, 2.0, 3.0], 10)
    noise = rng.normal(0, 0.05, size=30)
    scores = base + noise
    df = pd.DataFrame({'subject': subjects, 'condition': conditions, 'score': scores})

    result = friedman_test(df, dependent_variable='score', subject='subject', within=['condition'], alpha=0.05)
    assert result.p_value < 0.05
    assert 'posthoc_guidance' in (result.test_specifics or {})


def test_decision_engine_routes_multigroup_independent_to_anova_path():
    blueprint = DataBlueprint(group_samples={'A': 5, 'B': 5, 'C': 5}, is_paired=False)
    state = create_initial_state(df=pd.DataFrame({'group': ['A', 'B', 'C'], 'value': [1, 2, 3]}))
    state.attach_blueprint(blueprint)
    route = decision_engine.evaluate_routing(state)
    assert route in {NodeName.ANOVA_ASSUMPTIONS, NodeName.ANOVA_ONE_WAY, NodeName.KRUSKAL_WALLIS}


def test_decision_engine_routes_repeated_measures_to_friedman_when_non_normal():
    metrics = {'value': DistributionMetric(normality_p_value=0.01)}
    blueprint = DataBlueprint(
        group_samples={'A': 5, 'B': 5, 'C': 5},
        is_paired=True,
        distribution_metrics=metrics,
    )
    state = create_initial_state(df=pd.DataFrame({'condition': ['A', 'B', 'C'], 'value': [1, 2, 3]}))
    state.attach_blueprint(blueprint)
    state.paired = True
    route = decision_engine.evaluate_routing(state, prefer_terminal=True)
    assert route == NodeName.FRIEDMAN


def test_chi2_falls_back_to_fisher_when_expected_counts_low():
    table = np.array([[1, 1], [1, 3]])
    result = chi2_test(table, alpha=0.05)
    assert result.test_name == "Fisher's Exact Test"
    assert result.effect_size_type == 'phi'
    effect = (result.test_specifics or {}).get('effect_size', {})
    assert effect.get('metric') == 'phi'
    assert effect.get('value') == pytest.approx(0.25, rel=1e-3)
    assert 'note' in (result.test_specifics or {})


def test_mcnemar_handles_paired_frame_and_reports_phi():
    pairs = np.array([[1, 1]] * 30 + [[1, 0]] * 10 + [[0, 1]] * 5 + [[0, 0]] * 25)
    df = pd.DataFrame(pairs, columns=['pre', 'post'])
    result = mcnemar_test(df, alpha=0.05)
    assert result.test_name == "McNemar's Test"
    assert result.effect_size_type == 'phi'
    assert result.p_value == pytest.approx(0.30176, rel=1e-4)


def test_cochran_armitage_trend_detects_monotonic_change():
    table = np.array([[2, 5, 12], [18, 15, 8]])
    result = cochran_armitage_trend_test(table, alpha=0.05)
    assert result.p_value < 0.01
    assert abs(result.statistics) > 2
