"""Shared workflow node helpers.

Internal module — not part of the public API.  Import ``call_test_agent``
from ``statmate.workflow.nodes`` instead.
"""

import inspect
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from langchain_core.messages import AIMessage
from pydantic_ai import Agent

from statmate.agents.agent_builder import StatTestDeps, run_sync_agent
from statmate.core import NodeExecutionError, get_logger
from statmate.core.config import default_config
from statmate.core.model_provider import execute_with_backoff
from statmate.core.validation import validate_assumptions
from statmate.workflow.state import WorkflowState

logger = get_logger(__name__)


def _force_pairing_transform(state: WorkflowState, df: pd.DataFrame, pair_cols: list[tuple[str, str]]) -> WorkflowState:
    """Align data into paired series when wide-format pairing is detected."""
    if not pair_cols:
        return state
    first_pair = pair_cols[0]
    col_a, col_b = first_pair
    if col_a not in df.columns or col_b not in df.columns:
        return state

    aligned = df[[col_a, col_b]].dropna()
    state.df = aligned[col_a]
    state.secondary_df = aligned[col_b]

    if state.data_blueprint:
        samples = {str(col_a): int(len(aligned)), str(col_b): int(len(aligned))}
        updated = state.data_blueprint.model_copy(update={"is_paired": True, "group_samples": samples})
        state.attach_blueprint(updated)

    return state


def call_test_agent(
    test_agent: Agent,
    state: WorkflowState,
    alpha: float | None = None,
    probability_key: str | None = None,
    assumption_test_type: str | None = None,
    assumption_secondary: pd.Series | pd.DataFrame | np.ndarray | None = None,
    assess_assumptions: bool = True,
    test_params: dict[str, Any] | None = None,
) -> WorkflowState:
    """Call a statistical agent and append its result.

    Args:
        test_agent: The agent to call.
        state: Current workflow state.
        alpha: Significance level. If None, uses default from config.
        probability_key: Explicit key under which to store the p-value.
        assumption_test_type: Override for labeling assumption diagnostics.
        assumption_secondary: Explicit secondary data for assumption checks.
        assess_assumptions: When False, skip assumption diagnostics.

    Returns:
        Updated workflow state.

    Raises:
        NodeExecutionError: If the agent execution fails.
    """
    if alpha is None:
        alpha = default_config.statistical.default_alpha

    try:
        fallback_func = getattr(test_agent, "_statmate_test_function", None)
        test_params_payload: dict[str, Any] = {"alpha": alpha}
        if state.statistical_design and callable(fallback_func):
            func_sig = inspect.signature(fallback_func)
            if "design_type" in func_sig.parameters:
                test_params_payload["design_type"] = state.statistical_design.design_type
        if test_params:
            test_params_payload.update(test_params)

        deps = StatTestDeps(
            data=state.df,
            data_secondary=state.secondary_df,
            test_params=test_params_payload,
        )

        assumption_entry: dict[str, object] | None = None
        if assess_assumptions:
            secondary = assumption_secondary if assumption_secondary is not None else state.secondary_df
            diag = validate_assumptions(
                state.df,
                test_type=assumption_test_type or getattr(test_agent, "_statmate_test_name", test_agent.name),
                secondary_data=secondary,
            )
            assumption_entry = {**diag, "node": test_agent.name, "timestamp": datetime.utcnow().isoformat()}
            state.add_assumption_entry(assumption_entry)

        # Build a descriptive prompt – Anthropic rejects empty text content blocks.
        agent_prompt = f"Run the {test_agent.name} on the provided data. Alpha={alpha}. Return structured results."
        result = execute_with_backoff(
            lambda: run_sync_agent(test_agent, user_prompt=agent_prompt, deps=deps),
            on_retry=lambda attempt, delay, exc: state.add_step(
                step="Rate limit backoff",
                detail=f"Retrying {test_agent.name} in {delay:.1f}s (attempt {attempt})",
                data={"error": str(exc)},
            ),
        )

        # Always compute the underlying statistical test to guarantee tool execution,
        # even if the LLM skipped the run_test tool.
        if callable(fallback_func):
            if isinstance(deps.data, pd.Series):
                primary = deps.data.to_numpy()
            else:
                primary = deps.data

            if isinstance(deps.data_secondary, pd.Series):
                secondary = deps.data_secondary.to_numpy()
            else:
                secondary = deps.data_secondary

            computed_params = deps.test_params or {}
            computed = (
                fallback_func(primary, secondary, **computed_params)
                if secondary is not None
                else fallback_func(primary, **computed_params)
            )
            result.statistical_test_result = computed

        state.add_result(AIMessage(content=str(result)))

        p_val = result.statistical_test_result.p_value
        p_float = float(p_val) if isinstance(p_val, float) else float(np.mean(p_val))
        prob_key = probability_key or test_agent.name
        state.add_probability(prob_key, p_float)

        # Record structured step for UI/clients
        test_label = (
            getattr(result.statistical_test_result, "test_name", None)
            or getattr(test_agent, "_statmate_test_name", None)
            or test_agent.name
        )
        stats_value = result.statistical_test_result.statistics
        if isinstance(stats_value, np.ndarray):
            stats_value = stats_value.tolist()
        specifics = result.statistical_test_result.test_specifics or {}
        effect_entry = None
        if isinstance(specifics, dict):
            effect_entry = specifics.get("effect_size")
        effect_size_value = None
        if isinstance(effect_entry, dict):
            effect_size_value = effect_entry.get("value")
        elif isinstance(effect_entry, (int, float, np.floating)):
            effect_size_value = effect_entry
        if isinstance(effect_size_value, np.floating):
            effect_size_value = float(effect_size_value)
        state.add_step(
            step=test_label,
            detail=result.result,
            data={
                "test_name": test_label,
                "statistics": float(stats_value) if isinstance(stats_value, (float, int, np.floating)) else stats_value,
                "null_hypothesis": result.statistical_test_result.null_hypothesis,
                "alternative": result.statistical_test_result.alternative,
                "effect_size_type": result.statistical_test_result.effect_size_type,
                "effect_size": effect_size_value,
                "confidence_interval": result.statistical_test_result.confidence_interval,
                "comments": result.comments,
                "assumptions": assumption_entry,
                "test_specifics": specifics,
            },
            p_value=p_float,
        )

        return state
    except Exception as e:
        logger.error(f"Error in call_test_agent {test_agent.name}: {e}")
        raise NodeExecutionError(node_name=f"call_test_agent({test_agent.name})", original_error=e) from e
