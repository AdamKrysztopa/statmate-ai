"""Unit tests for the cox_regression_node hard-disable."""
import pandas as pd

from statmate.workflow.nodes import cox_regression_node
from statmate.workflow.state import WorkflowState


def _make_state() -> WorkflowState:
    df = pd.DataFrame({'time': [1.0, 2.0, 3.0], 'event': [1, 0, 1]})
    return WorkflowState(df=df)


def test_cox_regression_node_sets_not_implemented() -> None:
    state = _make_state()
    result = cox_regression_node(state)
    assert result is state  # mutates in-place
    assert len(result.execution_trace) == 1
    step = result.execution_trace[0]
    assert step['data']['status'] == 'NOT_IMPLEMENTED'
    assert step['data']['error_message']  # non-empty string


def test_cox_regression_node_error_message_is_not_empty() -> None:
    state = _make_state()
    result = cox_regression_node(state)
    msg = result.execution_trace[0]['data']['error_message']
    assert isinstance(msg, str) and len(msg) > 10
