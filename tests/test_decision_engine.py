import pandas as pd
import pytest

from statmate.core.config import DataType, NodeName
from statmate.core.exceptions import StatisticalAssumptionError
from statmate.core.validation import requires_assumptions
from statmate.workflow.blueprint import build_data_blueprint
from statmate.workflow.edges import decision_engine
from statmate.workflow.methodology_auditor import MethodologyAuditor
from statmate.workflow.nodes import choice_node, resolve_choice
from statmate.workflow.state import WorkflowState, create_initial_state


def test_build_data_blueprint_uses_roles_and_balance():
    df = pd.DataFrame({'value': [1, 2, 3, 4], 'group': ['a', 'a', 'b', 'b'], 'covariate': [0, 1, 0, 1]})
    blueprint = build_data_blueprint(
        df,
        dependent_vars=['value'],
        group_var='group',
        raw_payload={'variable_roles': [{'name': 'value', 'role': 'dependent'}]},
    )

    assert blueprint.sample_balance is not None
    assert blueprint.sample_balance.balance_ratio == pytest.approx(1.0)
    roles = {r.name: r.role for r in blueprint.variable_roles}
    assert roles['group'] == 'group'
    assert 'value' in blueprint.distribution_metrics


def test_decision_engine_survival_and_mcnemar_routing():
    survival_state = create_initial_state(df=pd.DataFrame({'time': [1, 2, 3], 'event': [1, 0, 1]}))
    survival_state.data_type = DataType.SURVIVAL
    survival_state.attach_blueprint(build_data_blueprint(survival_state.df))
    assert decision_engine.evaluate_routing(survival_state) == NodeName.COX_REGRESSION

    paired_state = create_initial_state(df=pd.DataFrame({'a': [0, 1], 'b': [1, 0]}))
    paired_state.data_type = DataType.CATEGORICAL
    paired_state.paired = True
    paired_state.attach_blueprint(build_data_blueprint(paired_state.df))
    assert decision_engine.evaluate_routing(paired_state) == NodeName.MCNEMAR


def test_methodology_auditor_requests_welch_on_variance_failure():
    df = pd.DataFrame({'value': [1, 2, 3, 4], 'group': [0, 0, 1, 1]})
    state = create_initial_state(df=df)
    state.data_type = DataType.CONTINUOUS
    state.paired = False
    state.number_of_samples = len(df)
    state.attach_blueprint(build_data_blueprint(df, dependent_vars=['value'], group_var='group'))
    state.assumption_log.append(
        {'status': 'fail', 'variance_ratio': 10.0, 'failures': ['Variance ratio 10.0 exceeds threshold']}
    )
    state.add_step(step=NodeName.INDEP_T, detail='ran independent t-test', data={})

    auditor = MethodologyAuditor()
    result = auditor.audit(state)
    assert result.correction_step
    assert result.correction_step['suggested_node'] == NodeName.WELCH
    assert state.correction_steps


def test_requires_assumptions_blocks_when_blueprint_non_normal():
    blueprint = build_data_blueprint(
        pd.DataFrame({'x': [1.0, 2.0, 3.0]}),
        raw_payload={'distribution_metrics': {'x': {'skewness': 0.0, 'kurtosis': 0.0, 'normality_p_value': 0.01}}},
    )
    state = create_initial_state(df=pd.DataFrame({'x': [1, 2, 3]}))
    state.attach_blueprint(blueprint)

    @requires_assumptions(normality=True)
    def guarded(state: WorkflowState):
        return True

    with pytest.raises(StatisticalAssumptionError):
        guarded(state)


def test_choice_node_respects_user_selection():
    state = create_initial_state(df=pd.DataFrame({'value': [1, 2, 3], 'group': [0, 0, 1]}))
    state.pending_routing_decision = {'primary': NodeName.INDEP_T, 'alternatives': [NodeName.WELCH]}
    state.user_selected_option = NodeName.WELCH

    updated = choice_node(state)
    assert updated.pending_routing_decision['selected'] == NodeName.WELCH
    assert resolve_choice(updated) == NodeName.WELCH
