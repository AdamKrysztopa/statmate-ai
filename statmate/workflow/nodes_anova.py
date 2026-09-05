"""ANOVA and repeated-measures workflow nodes.

Covers: anova_assumptions_node, anova_one_way_node, kruskal_wallis_node,
anova_rm_node, friedman_node.
"""

import numpy as np
import pandas as pd
import scipy.stats

from statmate.core import NodeExecutionError, get_logger
from statmate.core.config import NodeName, default_config
from statmate.workflow._node_helpers import call_test_agent
from statmate.workflow.model_factory import create_model, create_model_settings
from statmate.workflow.state import WorkflowState

logger = get_logger(__name__)


def _infer_group_and_value_columns(state: WorkflowState) -> tuple[pd.DataFrame, str, str]:
    """Resolve grouping and value columns from design/blueprint hints."""
    df = state.df if isinstance(state.df, pd.DataFrame) else pd.DataFrame(state.df)
    blueprint = state.data_blueprint
    design = state.statistical_design

    group_col = None
    if design and design.grouping_variable and design.grouping_variable in df.columns:
        group_col = design.grouping_variable
    elif blueprint:
        for role in blueprint.variable_roles:
            if getattr(role, "role", "") == "group" and role.name in df.columns:
                group_col = role.name
                break
    if group_col is None:
        fallback_groups = [col for col in df.columns if not pd.api.types.is_numeric_dtype(df[col])]
        group_col = fallback_groups[0] if fallback_groups else None

    if not group_col or group_col not in df.columns:
        raise ValueError("Grouping column could not be inferred.")

    dep_candidates: list[str] = []
    if design and design.dependent_variable:
        dep_candidates.extend([col.strip(" []'\"") for col in str(design.dependent_variable).split(",") if col])
    if blueprint:
        dep_candidates.extend(
            [role.name for role in blueprint.variable_roles if getattr(role, "role", "") == "dependent"]
        )
    value_col = next((c for c in dep_candidates if c in df.columns and pd.api.types.is_numeric_dtype(df[c])), None)
    if value_col is None:
        numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col]) and col != group_col]
        value_col = numeric_cols[0] if numeric_cols else None

    if not value_col:
        raise ValueError("No numeric measurement column available for group comparison.")

    return df, group_col, value_col


def _extract_group_arrays(df: pd.DataFrame, group_col: str, value_col: str) -> tuple[list[np.ndarray], list[str]]:
    """Return per-group numeric arrays and labels."""
    groups: list[np.ndarray] = []
    labels: list[str] = []
    for label, series in df.groupby(group_col)[value_col]:
        arr = pd.to_numeric(series, errors="coerce").dropna().to_numpy()
        if arr.size == 0:
            continue
        groups.append(arr)
        labels.append(str(label))
    if len(groups) < 2:
        raise ValueError("At least two non-empty groups are required.")
    return groups, labels


def _resolve_repeated_measures_columns(state: WorkflowState) -> tuple[pd.DataFrame, str, str, list[str]]:
    """Infer dependent, subject, and within-factor columns for repeated-measures tests."""
    df = state.df if isinstance(state.df, pd.DataFrame) else pd.DataFrame(state.df)
    frame = df.copy()
    design = state.statistical_design
    blueprint = state.data_blueprint

    subject_col = None
    if design and design.subject_id_column:
        subject_col = design.subject_id_column
    elif blueprint and blueprint.index_column:
        subject_col = blueprint.index_column
    elif frame.index.name:
        subject_col = frame.index.name
        frame = frame.reset_index()

    if subject_col and subject_col not in frame.columns and df.index.name == subject_col:
        frame = df.reset_index()

    if subject_col is None:
        frame = frame.reset_index()
        subject_col = frame.columns[0]

    if subject_col not in frame.columns:
        raise ValueError("Subject identifier column is required for repeated measures.")

    within_col = None
    if design and design.grouping_variable and design.grouping_variable in frame.columns:
        within_col = design.grouping_variable
    elif blueprint:
        for role in blueprint.variable_roles:
            if getattr(role, "role", "") == "group" and role.name in frame.columns:
                within_col = role.name
                break
    if within_col is None:
        categorical_cols = [
            col for col in frame.columns if not pd.api.types.is_numeric_dtype(frame[col]) and col != subject_col
        ]
        within_col = categorical_cols[0] if categorical_cols else None

    if within_col is None or within_col not in frame.columns:
        raise ValueError("Within-subject factor column is required for repeated measures.")

    dep_candidates: list[str] = []
    if design and design.dependent_variable:
        dep_candidates.extend([col.strip(" []'\"") for col in str(design.dependent_variable).split(",") if col])
    if blueprint:
        dep_candidates.extend(
            [role.name for role in blueprint.variable_roles if getattr(role, "role", "") == "dependent"]
        )
    value_col = next(
        (
            c
            for c in dep_candidates
            if c in frame.columns and pd.api.types.is_numeric_dtype(frame[c]) and c not in (within_col, subject_col)
        ),
        None,
    )
    if value_col is None:
        numeric_cols = [
            col
            for col in frame.columns
            if pd.api.types.is_numeric_dtype(frame[col]) and col not in (within_col, subject_col)
        ]
        value_col = numeric_cols[0] if numeric_cols else None

    if value_col is None:
        raise ValueError("No numeric dependent variable column found for repeated measures.")

    return frame, value_col, subject_col, [within_col]


def anova_assumptions_node(state: WorkflowState, alpha: float | None = None) -> WorkflowState:
    """Check normality per group and variance homogeneity before ANOVA."""
    if alpha is None:
        alpha = default_config.statistical.variance_threshold

    try:
        df, group_col, value_col = _infer_group_and_value_columns(state)
        groups, labels = _extract_group_arrays(df, group_col, value_col)

        shapiro_pvalues: dict[str, float | None] = {}
        min_normal: float | None = None
        for label, arr in zip(labels, groups):
            if len(arr) < 3:
                shapiro_pvalues[label] = None
                continue
            _, p_val = scipy.stats.shapiro(arr)
            p_float = float(p_val)
            shapiro_pvalues[label] = p_float
            min_normal = p_float if min_normal is None else min(min_normal, p_float)

        levene_stat, levene_p = scipy.stats.levene(*groups, center="median")
        variance_pass = float(levene_p) >= alpha
        normal_pass = min_normal is None or min_normal >= alpha
        status = "pass" if variance_pass and normal_pass else "fail"

        assumption_entry = {
            "node": NodeName.ANOVA_ASSUMPTIONS,
            "group_column": group_col,
            "value_column": value_col,
            "shapiro_p_values": shapiro_pvalues,
            "levene_statistic": float(levene_stat),
            "levene_p_value": float(levene_p),
            "alpha": alpha,
            "status": status,
        }
        state.add_assumption_entry(assumption_entry)
        if min_normal is not None:
            state.add_probability("anova_min_shapiro", float(min_normal))
        state.add_probability("anova_levene", float(levene_p))
        state.add_step(
            step=NodeName.ANOVA_ASSUMPTIONS,
            detail="ANOVA assumption check (Shapiro per group + Levene)",
            data=assumption_entry,
            p_value=float(levene_p),
        )
        state.pending_routing_decision = (state.pending_routing_decision or {}) | {
            "group_column": group_col,
            "value_column": value_col,
        }
        return state
    except Exception as e:
        logger.error(f"Error in anova_assumptions_node: {e}")
        raise NodeExecutionError(node_name="anova_assumptions_node", original_error=e) from e


def anova_one_way_node(state: WorkflowState) -> WorkflowState:
    """Run one-way ANOVA on 3+ independent groups."""
    try:
        from statmate.agents import anova_one_way_agent

        df, group_col, value_col = _infer_group_and_value_columns(state)
        model = create_model(model_name=state.model_name, provider=state.provider)
        settings = create_model_settings(model_name=state.model_name)
        agent = anova_one_way_agent(model=model, model_settings=settings)

        return call_test_agent(
            agent,
            state,
            probability_key="anova_one_way",
            assess_assumptions=False,
            test_params={"group_column": group_col, "value_column": value_col},
        )
    except Exception as e:
        logger.error(f"Error in anova_one_way_node: {e}")
        raise NodeExecutionError(node_name="anova_one_way_node", original_error=e) from e


def kruskal_wallis_node(state: WorkflowState) -> WorkflowState:
    """Run Kruskal-Wallis with Dunn post-hoc support."""
    try:
        from statmate.agents import kruskal_wallis_agent

        df, group_col, value_col = _infer_group_and_value_columns(state)
        model = create_model(model_name=state.model_name, provider=state.provider)
        settings = create_model_settings(model_name=state.model_name)
        agent = kruskal_wallis_agent(model=model, model_settings=settings)

        return call_test_agent(
            agent,
            state,
            probability_key="kruskal_wallis",
            assess_assumptions=False,
            test_params={"group_column": group_col, "value_column": value_col, "perform_dunn": True},
        )
    except Exception as e:
        logger.error(f"Error in kruskal_wallis_node: {e}")
        raise NodeExecutionError(node_name="kruskal_wallis_node", original_error=e) from e


def anova_rm_node(state: WorkflowState) -> WorkflowState:
    """Run repeated-measures ANOVA."""
    try:
        from statmate.agents import anova_rm_agent

        frame, value_col, subject_col, within_cols = _resolve_repeated_measures_columns(state)
        model = create_model(model_name=state.model_name, provider=state.provider)
        settings = create_model_settings(model_name=state.model_name)
        agent = anova_rm_agent(model=model, model_settings=settings)

        state.df = frame
        return call_test_agent(
            agent,
            state,
            probability_key="anova_rm",
            assess_assumptions=False,
            test_params={
                "dependent_variable": value_col,
                "subject": subject_col,
                "within": within_cols,
            },
        )
    except Exception as e:
        logger.error(f"Error in anova_rm_node: {e}")
        raise NodeExecutionError(node_name="anova_rm_node", original_error=e) from e


def friedman_node(state: WorkflowState) -> WorkflowState:
    """Run Friedman test for repeated measures."""
    try:
        from statmate.agents import friedman_agent

        frame, value_col, subject_col, within_cols = _resolve_repeated_measures_columns(state)
        model = create_model(model_name=state.model_name, provider=state.provider)
        settings = create_model_settings(model_name=state.model_name)
        agent = friedman_agent(model=model, model_settings=settings)

        state.df = frame
        return call_test_agent(
            agent,
            state,
            probability_key="friedman_test",
            assess_assumptions=False,
            test_params={
                "dependent_variable": value_col,
                "subject": subject_col,
                "within": within_cols,
            },
        )
    except Exception as e:
        logger.error(f"Error in friedman_node: {e}")
        raise NodeExecutionError(node_name="friedman_node", original_error=e) from e
