import numpy as np
import pandas as pd
import pytest

from statmate.agents.agent_builder import AgentResult
from statmate.core.config import NodeName
from statmate.core.exceptions import NodeExecutionError
from statmate.core.validation import StatisticalDesign
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

    updated = call_test_agent(agent, state, alpha=0.05, probability_key='dummy_key')

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


def test_design_verification_blocks_mismatch():
    state = create_initial_state(df=pd.DataFrame({'a': [1, 2]}))
    state.statistical_design = StatisticalDesign(
        design_type='paired',
        is_paired=True,
        grouping_variable='group',
        subject_id_column='subject_id',
        rationale='overlap detected',
    )
    state.agent_design_hypothesis = 'independent'

    with pytest.raises(NodeExecutionError):
        design_verification_node(state)
