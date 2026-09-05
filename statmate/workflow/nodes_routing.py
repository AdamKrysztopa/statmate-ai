"""Routing, fallback, and special-case workflow nodes.

Covers: intent_discovery_node, choice_node, resolve_choice,
user_intervention_node, descriptive_summary_node, cox_regression_node,
regression_node.
"""

import json

import pandas as pd
from langchain_core.messages import AIMessage

from statmate.core import NodeExecutionError, get_logger
from statmate.core.config import NodeName
from statmate.workflow.state import WorkflowState

logger = get_logger(__name__)


def intent_discovery_node(state: WorkflowState) -> WorkflowState:
    """Lightweight intent check to trigger ambiguity modal when confidence is low."""
    try:
        confidence = 0.5
        if state.target_columns:
            confidence = 0.9
        elif state.statistical_design:
            confidence = 0.8

        summary = state.intent_summary or "Explore relationships in the provided data."
        trigger_modal = confidence < 0.8
        state.intent_confidence = confidence
        state.intent_summary = summary
        state.add_step(
            step=NodeName.INTENT,
            detail=summary,
            data={
                "confidence": confidence,
                "trigger_ambiguity_modal": trigger_modal,
            },
        )
        return state
    except Exception as e:  # pragma: no cover - defensive
        logger.error(f"Error in intent_discovery_node: {e}")
        raise NodeExecutionError(node_name="intent_discovery_node", original_error=e) from e


def choice_node(state: WorkflowState) -> WorkflowState:
    """Expose routing options to allow user or UI to decide."""
    try:
        decision = state.pending_routing_decision or {}
        primary = decision.get("primary")
        alternatives = decision.get("alternatives") or []

        if state.user_selected_option:
            valid_options = [opt for opt in [primary, *alternatives] if opt]
            if state.user_selected_option not in valid_options:
                state.add_step(
                    step=NodeName.CHOICE,
                    detail="Invalid override ignored; falling back to default option.",
                    data={
                        "invalid_override": state.user_selected_option,
                        "valid_options": valid_options,
                    },
                )
                state.user_selected_option = None

        selected = state.user_selected_option or decision.get("selected") or primary

        entry = {
            "primary": primary,
            "alternatives": alternatives,
            "selected": selected,
            "reason": decision.get("reason"),
        }
        state.choice_log.append(entry)
        state.pending_routing_decision = {**decision, "selected": selected}
        state.add_step(step=NodeName.CHOICE, detail=f"Chosen {selected or primary}", data=entry)
        return state
    except Exception as e:  # pragma: no cover - defensive
        logger.error("Error in choice_node: %s", e)
        raise NodeExecutionError(node_name="choice_node", original_error=e) from e


def resolve_choice(state: WorkflowState) -> str:
    """Resolve the chosen next node after presenting options."""
    decision = state.pending_routing_decision or {}
    return decision.get("selected") or decision.get("primary") or NodeName.ASSESS_STUDY_DESIGN


def user_intervention_node(state: WorkflowState) -> WorkflowState:
    """Stop the graph and surface a user-facing intervention request."""
    state.add_step(
        step=NodeName.USER_INTERVENTION,
        detail="Routing blocked by guardrails; manual choice required.",
        data={
            "pending_decision": state.pending_routing_decision,
            "blueprint": state.data_blueprint.model_dump() if state.data_blueprint else None,
        },
    )
    return state


def descriptive_summary_node(state: WorkflowState) -> WorkflowState:
    """Fallback node that returns descriptive statistics when inferential tests are blocked."""
    frame = state.df if isinstance(state.df, pd.DataFrame) else state.df.to_frame()
    summary = frame.describe(include="all").to_dict()
    group_samples = state.data_blueprint.group_samples if state.data_blueprint else None
    payload = {"summary": summary, "group_samples": group_samples}
    state.add_result(AIMessage(content=json.dumps(payload, default=str)))
    state.add_step(
        step=NodeName.DESCRIPTIVE_SUMMARY,
        detail="Insufficient sample size for inferential testing; returning descriptive summary.",
        data=payload,
    )
    return state


def cox_regression_node(state: WorkflowState) -> WorkflowState:
    """Cox proportional hazards regression — not yet implemented."""
    error_msg = (
        "Survival analysis (Cox regression) is not yet supported in this version. "
        "Please use a dedicated survival analysis tool or contact the team to prioritise this feature."
    )
    state.add_step(
        step=NodeName.COX_REGRESSION,
        detail=error_msg,
        data={"status": "NOT_IMPLEMENTED", "error_message": error_msg},
    )
    return state


def regression_node(state: WorkflowState) -> WorkflowState:
    """Run regression analysis based on the blueprint's regression_intent."""
    from statmate.statistical_core.regression import (
        linear_regression,
        logistic_regression,
        multiple_regression_with_vif,
    )

    blueprint = state.data_blueprint
    intent = blueprint.regression_intent if blueprint else "none"
    target_col = blueprint.target_column if blueprint else None

    if intent == "none" or not target_col:
        state.add_step(
            step="regression_node",
            detail="No regression intent detected; skipping regression analysis.",
            data={"status": "skipped"},
        )
        return state

    df = state.df if isinstance(state.df, pd.DataFrame) else state.df.to_frame()
    y = df[target_col]
    X = df.drop(columns=[target_col]).select_dtypes(include="number")

    try:
        if intent == "logistic":
            result = logistic_regression(X, y)
        elif X.shape[1] > 1:
            result = multiple_regression_with_vif(X, y)
        else:
            result = linear_regression(X, y)
        result_dump = result.model_dump()
        state.add_result(AIMessage(content=json.dumps(result_dump, default=str)))
        state.add_step(
            step="regression_node",
            detail=f"Regression analysis complete ({result.test_name}).",
            data=result_dump,
            p_value=result.p_value if isinstance(result.p_value, float) else None,
        )
    except ValueError as exc:
        logger.error("Error in regression_node: %s", exc)
        state.add_step(
            step="regression_node",
            detail=f"Regression analysis failed: {exc}",
            data={"error": str(exc)},
        )
    return state
