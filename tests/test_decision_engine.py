import pandas as pd
import pytest

from statmate.core.config import DataType, NodeName
from statmate.core.validation import requires_assumptions, validate_statistical_design
from statmate.workflow.blueprint import build_data_blueprint
from statmate.workflow.edges import decision_engine
from statmate.workflow.methodology_auditor import MethodologyAuditor, StructureAuditor
from statmate.workflow.nodes import choice_node, resolve_choice
from statmate.workflow.state import WorkflowState, create_initial_state


def test_build_data_blueprint_uses_roles_and_balance():
    df = pd.DataFrame({"value": [1, 2, 3, 4], "group": ["a", "a", "b", "b"], "covariate": [0, 1, 0, 1]})
    blueprint = build_data_blueprint(
        df,
        dependent_vars=["value"],
        group_var="group",
        raw_payload={"variable_roles": [{"name": "value", "role": "dependent"}]},
    )

    assert blueprint.sample_balance is not None
    assert blueprint.sample_balance.balance_ratio == pytest.approx(1.0)
    assert blueprint.group_samples == {"a": 2, "b": 2}
    roles = {r.name: r.role for r in blueprint.variable_roles}
    assert roles["group"] == "group"
    assert "value" in blueprint.distribution_metrics


def test_decision_engine_survival_and_mcnemar_routing():
    survival_state = create_initial_state(df=pd.DataFrame({"time": [1, 2, 3], "event": [1, 0, 1]}))
    survival_state.data_type = DataType.SURVIVAL
    survival_state.attach_blueprint(build_data_blueprint(survival_state.df))
    assert decision_engine.evaluate_routing(survival_state) == NodeName.COX_REGRESSION

    paired_state = create_initial_state(df=pd.DataFrame({"a": [0, 1], "b": [1, 0]}))
    paired_state.data_type = DataType.CATEGORICAL
    paired_state.paired = True
    paired_state.attach_blueprint(build_data_blueprint(paired_state.df))
    assert decision_engine.evaluate_routing(paired_state) == NodeName.MCNEMAR


def test_sufficiency_validator_routes_to_descriptive_summary():
    df = pd.DataFrame({"value": [1], "group": ["solo"]})
    state = create_initial_state(df=df)
    state.data_type = DataType.CONTINUOUS
    state.attach_blueprint(build_data_blueprint(df, dependent_vars=["value"], group_var="group"))
    assert decision_engine.evaluate_routing(state) == NodeName.DESCRIPTIVE_SUMMARY


def test_methodology_auditor_requests_welch_on_variance_failure():
    df = pd.DataFrame({"value": [1, 2, 3, 4], "group": [0, 0, 1, 1]})
    state = create_initial_state(df=df)
    state.data_type = DataType.CONTINUOUS
    state.paired = False
    state.number_of_samples = len(df)
    state.attach_blueprint(build_data_blueprint(df, dependent_vars=["value"], group_var="group"))
    state.assumption_log.append(
        {"status": "fail", "variance_ratio": 10.0, "failures": ["Variance ratio 10.0 exceeds threshold"]}
    )
    state.add_step(step=NodeName.INDEP_T, detail="ran independent t-test", data={})

    auditor = MethodologyAuditor()
    result = auditor.audit(state)
    assert result.correction_step
    assert result.correction_step["suggested_node"] == NodeName.WELCH
    assert state.correction_steps


def test_structure_auditor_enforces_paired_tests():
    df = pd.DataFrame({"value": [1, 2, 3, 4], "group": ["a", "a", "b", "b"], "id": [1, 2, 1, 2]})
    state = create_initial_state(df=df)
    blueprint = build_data_blueprint(df, dependent_vars=["value"], group_var="group", is_paired=True, index_column="id")
    state.attach_blueprint(blueprint)
    state.execution_trace.append({"step": NodeName.INDEP_T})

    auditor = StructureAuditor()
    result = auditor.audit(state)
    assert result is not None
    assert result.recommended == NodeName.PAIRED_T
    assert state.pending_routing_decision["selected"] == NodeName.PAIRED_T


def test_requires_assumptions_passes_when_normality_skipped():
    # normality=True now means "skip the normality check" (guardrail removed per P0 roadmap).
    # The decorated function should return normally regardless of blueprint distribution metrics.
    blueprint = build_data_blueprint(
        pd.DataFrame({"x": [1.0, 2.0, 3.0]}),
        raw_payload={"distribution_metrics": {"x": {"skewness": 0.0, "kurtosis": 0.0, "normality_p_value": 0.01}}},
    )
    state = create_initial_state(df=pd.DataFrame({"x": [1, 2, 3]}))
    state.attach_blueprint(blueprint)

    @requires_assumptions(normality=True)
    def guarded(state: WorkflowState):
        return True

    assert guarded(state) is True


def test_choice_node_respects_user_selection():
    state = create_initial_state(df=pd.DataFrame({"value": [1, 2, 3], "group": [0, 0, 1]}))
    state.pending_routing_decision = {"primary": NodeName.INDEP_T, "alternatives": [NodeName.WELCH]}
    state.user_selected_option = NodeName.WELCH

    updated = choice_node(state)
    assert updated.pending_routing_decision["selected"] == NodeName.WELCH
    assert resolve_choice(updated) == NodeName.WELCH


def test_validate_statistical_design_detects_wide_format_pairs():
    df = pd.DataFrame({"pre_score": [1, 2, 3], "post_score": [2, 3, 4]})
    design = validate_statistical_design(df, dependent_var=["pre_score", "post_score"])
    assert design.is_paired
    assert "wide_format_pairs" in design.overlap_summary


def test_decision_engine_routes_to_design_reconciliation_on_mismatch():
    df = pd.DataFrame({"pre": [1, 2, 3], "post": [2, 3, 4]})
    state = create_initial_state(df=df)
    state.data_type = DataType.CONTINUOUS
    state.attach_blueprint(build_data_blueprint(df, dependent_vars=["pre", "post"], is_paired=True))
    state.design_verification = {
        "mismatch": True,
        "structural_design": {"design_type": "paired"},
        "agent_design": "independent",
    }
    assert decision_engine.evaluate_routing(state) == NodeName.DESIGN_RECONCILIATION
