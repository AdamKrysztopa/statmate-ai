from collections.abc import Iterable
from typing import cast

import numpy as np
import pandas as pd
import pytest

from statmate.agents.agent_builder import AgentResult
from statmate.agents.initial_insights_agent import InitialInsightsAgentResults, format_data_by_recommendation
from statmate.agents.initial_insights_agent import NodeName as AgentNodeName
from statmate.core.config import NodeName
from statmate.core.validation import StatisticalDesign, validate_statistical_design
from statmate.statistical_core.base import StatTestResult
from statmate.workflow.edges import decide_two_independent, parametric_assumptions
from statmate.workflow.nodes import call_test_agent, design_verification_node
from statmate.workflow.state import create_initial_state


class DummyAgent:
    """Lightweight stand-in for a Pydantic AI agent."""

    def __init__(self, name, func):
        self.name = name
        self._statmate_test_function = func  # type: ignore[attr-defined]


def _stat_result(p_value: float) -> StatTestResult:
    return StatTestResult(
        test_name='dummy',
        statistics=0.0,
        p_value=p_value,
        null_hypothesis='H0',
        alternative=None,
        statistical_test_results='',
    )


def test_call_test_agent_uses_explicit_probability_key(monkeypatch):
    """call_test_agent should store p-values under the provided key and run the raw test."""

    def fake_stat_func(data, data_secondary=None, alpha=0.05):
        return _stat_result(0.01)

    def fake_run_sync_agent(agent, user_prompt, deps):
        # Return an AgentResult with a different p-value to ensure fallback is applied
        return AgentResult(statistical_test_result=_stat_result(0.99), result='llm', comments='llm')

    monkeypatch.setattr('statmate.workflow.nodes.run_sync_agent', fake_run_sync_agent)

    agent = DummyAgent('Statistical Test Agent: Dummy', fake_stat_func)
    state = create_initial_state(df=pd.Series(np.arange(5)))

    updated = call_test_agent(agent, state, alpha=0.05, probability_key='dummy_key')  # type: ignore[arg-type]

    assert updated.get_probability('dummy_key') == 0.01
    assert updated.results, 'Agent result should be recorded'


def test_decide_two_independent_respects_named_probabilities():
    state = create_initial_state(df=pd.DataFrame({'a': [1, 2]}))
    state.add_probability('shapiro_group1', 0.2)
    state.add_probability('shapiro_group2', 0.2)
    state.add_probability('levene', 0.2)

    assert decide_two_independent(state, alpha=0.05) == NodeName.INDEP_T

    state.probabilities['shapiro_group1'] = 0.01
    assert decide_two_independent(state, alpha=0.05) == NodeName.NONPARAMETRIC


def test_parametric_assumptions_for_paired_branch():
    state = create_initial_state(df=pd.DataFrame({'a': [1, 2]}))
    state.add_probability('normality_of_difference', 0.2)
    assert parametric_assumptions(state, alpha=0.05) == NodeName.PAIRED_T

    state.probabilities['normality_of_difference'] = 0.01
    assert parametric_assumptions(state, alpha=0.05) == NodeName.WILCOXON


def test_design_verification_records_mismatch():
    state = create_initial_state(df=pd.DataFrame({'a': [1, 2]}))
    state.statistical_design = StatisticalDesign(
        design_type='paired',
        is_paired=True,
        grouping_variable='group',
        subject_id_column='subject_id',
        rationale='overlap detected',
    )
    state.agent_design_hypothesis = 'independent'

    updated = design_verification_node(state)

    assert updated.design_verification is not None
    assert updated.design_verification['mismatch'] is True
    assert updated.design_verification['structural_design']['design_type'] == 'paired'
    assert updated.design_verification['agent_design'] == 'independent'


def test_format_data_by_recommendation_pads_missing_columns():
    df = pd.DataFrame({'value': [1, 2, 3], 'group': ['a', 'a', 'b']})
    rec = InitialInsightsAgentResults(
        analysis_columns=['value'],
        group_column=None,
        output_format='pd.Series',
        data_analysis_result='',
        route_to_test=[
            AgentNodeName.ASSESS_STUDY_DESIGN,
            AgentNodeName.TWO_INDEPENDENT_GROUPS,
            AgentNodeName.INDEP_T,
        ],
        comments='',
        data_type='CONTINUOUS',
        data_design='independent',
        data_transformation='None',
        tool_arguments={},
        data_size=len(df),
        number_of_columns=df.shape[1],
    )

    s1, s2 = format_data_by_recommendation(df, rec)
    assert len(list(cast(Iterable[float], s1))) == len(df)
    assert len(list(cast(Iterable[float], s2))) == len(df)


def test_format_data_by_recommendation_raises_when_only_one_column():
    df = pd.DataFrame({'value': [1, 2, 3]})
    rec = InitialInsightsAgentResults(
        analysis_columns=['value'],
        group_column=None,
        output_format='pd.Series',
        data_analysis_result='',
        route_to_test=[AgentNodeName.INDEP_T],
        comments='',
        data_type='CONTINUOUS',
        data_design='independent',
        data_transformation='None',
        tool_arguments={},
        data_size=len(df),
        number_of_columns=df.shape[1],
    )

    with pytest.raises(ValueError):
        format_data_by_recommendation(df, rec)


def test_format_data_by_recommendation_splits_by_group_for_independent_design():
    df = pd.DataFrame({'value': [1, 2, 3, 4], 'group': ['A', 'A', 'B', 'B']})
    rec = InitialInsightsAgentResults(
        analysis_columns=['value'],
        group_column='group',
        output_format='pd.Series',
        data_analysis_result='',
        route_to_test=[
            AgentNodeName.ASSESS_STUDY_DESIGN,
            AgentNodeName.TWO_INDEPENDENT_GROUPS,
            AgentNodeName.INDEP_T,
        ],
        comments='',
        data_type='CONTINUOUS',
        data_design='independent',
        data_transformation='None',
        tool_arguments={},
        data_size=len(df),
        number_of_columns=df.shape[1],
    )
    design = StatisticalDesign(
        design_type='independent',
        is_paired=False,
        grouping_variable='group',
        subject_id_column=None,
        rationale='No overlap',
        suggested_groups=['A', 'B'],
    )

    s1, s2 = format_data_by_recommendation(df, rec, design)
    assert list(cast(Iterable[float], s1)) == [1, 2]
    assert list(cast(Iterable[float], s2)) == [3, 4]


def test_validate_statistical_design_prioritises_wide_dep_list_with_group():
    df = pd.DataFrame(
        {
            'pre_treatment_value': [1, 2, 3],
            'post_treatment_value': [2, 3, 4],
            'subject_id': [101, 102, 103],
            'group': ['A', 'B', 'C'],
        }
    )

    design = validate_statistical_design(
        df,
        dependent_var=['pre_treatment_value', 'post_treatment_value'],
        group_var='group',
        subject_id='subject_id',
    )

    assert design.design_type == 'paired'
    assert design.is_paired is True
    rationale = (design.rationale or '').lower()
    assert 'wide-format' in rationale
    assert 'pair' in rationale


def test_format_data_by_recommendation_prefers_validator_paired_design():
    df = pd.DataFrame(
        {
            'pre': [1, 2, 3],
            'post': [2, 3, 4],
            'subject': ['a', 'b', 'c'],
        }
    )
    rec = InitialInsightsAgentResults(
        analysis_columns=['post'],
        group_column='subject',
        output_format='pd.Series',
        data_analysis_result='',
        route_to_test=[AgentNodeName.TWO_INDEPENDENT_GROUPS, AgentNodeName.INDEP_T],
        comments='',
        data_type='CONTINUOUS',
        data_design='independent',
        data_transformation='None',
        tool_arguments={},
        data_size=len(df),
        number_of_columns=df.shape[1],
    )
    design = StatisticalDesign(
        design_type='paired',
        is_paired=True,
        grouping_variable=None,
        subject_id_column=None,
        rationale='Multiple measurement columns per row (wide-format) detected.',
        dependent_variable='pre, post',
    )

    s1, s2 = format_data_by_recommendation(df, rec, design)
    assert list(cast(Iterable[float], s1)) == [1, 2, 3]
    assert list(cast(Iterable[float], s2)) == [2, 3, 4]


def test_validate_statistical_design_does_not_force_paired_when_deps_include_ids_and_group():
    df = pd.DataFrame(
        {
            'patient_id': range(6),
            'treatment method': ['A', 'A', 'A', 'B', 'B', 'B'],
            'result': [1.1, 2.2, 3.3, 2.1, 2.9, 3.5],
        }
    )
    design = validate_statistical_design(
        df,
        dependent_var=list(df.columns),  # fallback path when agent omits analysis_columns
        group_var='treatment method',
        subject_id='patient_id',
    )
    assert design.design_type == 'independent'
    assert design.is_paired is False


def test_validate_statistical_design_handles_long_format_with_id_and_value_only():
    df = pd.DataFrame(
        {
            'patient_id': range(1, 6),
            'treatment method': ['A', 'A', 'A', 'B', 'B'],
            'result': [100.0, 90.0, 110.0, 105.0, 95.0],
        }
    )
    design = validate_statistical_design(
        df,
        dependent_var=list(df.columns),
        group_var=None,
        subject_id='patient_id',
    )
    assert design.design_type == 'independent'
    assert design.grouping_variable in (None, 'treatment method')


def test_validate_statistical_design_respects_hint_group_var_when_columns_are_pivoted():
    wide_df = pd.DataFrame(
        {
            'result_A': [1.0, 2.0, 3.0],
            'result_B': [1.5, 2.5, 3.5],
        }
    )
    design = validate_statistical_design(
        wide_df,
        dependent_var=list(wide_df.columns),
        group_var='treatment method',
        subject_id='patient_id',
    )

    assert design.design_type == 'independent'
    assert design.grouping_variable == 'treatment method'
