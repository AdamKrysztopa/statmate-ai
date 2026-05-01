"""Workflow node functions.

This module contains all the node functions for the statistical test workflow,
extracted from the monolithic statmate_flow.py for better organization.
"""

import inspect
import json
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
import scipy.stats
from langchain_core.messages import AIMessage
from pydantic_ai import Agent

from statmate.agents import (
    shapiro_wilk_agent,
)
from statmate.agents.agent_builder import StatTestDeps, run_sync_agent
from statmate.agents.auxiliary_agents import AssessDesignDeps, get_assess_design_study_agent
from statmate.agents.initial_insights_agent import (
    INITIAL_INSIGHTS_PROMPT,
    TOOL_FUNCS,
    InitialInsightsAgentDeps,
    InitialInsightsAgentResults,
    build_initial_insights_agent,
    build_partition_report,
    format_data_by_recommendation,
    validate_tool_args,
)
from statmate.agents.reviewer_agent import ReviewerDeps, get_reviewer_agent
from statmate.agents.summarizer_agent import SummariserDeps, get_summariser_agent
from statmate.core import NodeExecutionError, get_logger
from statmate.core.config import NodeName, default_config
from statmate.core.exceptions import RoutingError
from statmate.core.model_provider import execute_with_backoff
from statmate.core.validation import (
    StatisticalDesign,
    detect_wide_format_pairing,
    get_structural_summary,
    infer_statistical_design,
    validate_assumptions,
    validate_statistical_design,
)
from statmate.workflow.blueprint import build_data_blueprint
from statmate.workflow.edges import decision_engine
from statmate.workflow.initialization.column_role_agent import propose_column_roles
from statmate.workflow.initialization.route_proposal import propose_route
from statmate.workflow.initialization.structural_check import check_structural_validity
from statmate.workflow.methodology_auditor import MethodologyAuditor, StructureAuditor
from statmate.workflow.model_factory import create_model, create_model_settings
from statmate.workflow.state import WorkflowState

logger = get_logger(__name__)


# Append strict tool_arguments requirement to prompt
ENHANCED_INITIAL_INSIGHTS_PROMPT = (
    INITIAL_INSIGHTS_PROMPT
    + "\nData Validation:\n  - Always include a non-null 'tool_arguments' dict (empty if no transform)."
)


def _run_structural_precheck(state: WorkflowState) -> tuple[StatisticalDesign, dict[str, Any]]:
    """Run deterministic schema checks before LLM reasoning."""
    design, structural_summary = infer_statistical_design(state.df)
    state.statistical_design = design
    state.paired = design.is_paired
    state.comparison_matrix = design.comparison_matrix
    state.add_step(
        step='Structural Validation (pre-agent)',
        detail=design.rationale or 'Structural design check completed.',
        data={
            'statistical_design': design.as_dict(),
            'structural_summary': structural_summary,
        },
    )
    return design, structural_summary


def _build_initialization_agent(state: WorkflowState) -> Agent:
    """Create the initialization agent using configured model settings."""
    model = create_model(model_name=state.model_name, provider=state.provider)
    settings = create_model_settings(model_name=state.model_name)
    return build_initial_insights_agent(
        model=model,
        system_prompt=ENHANCED_INITIAL_INSIGHTS_PROMPT,
        model_settings=settings,
    )


def _build_structural_prompt(design: StatisticalDesign, structural_summary: dict[str, Any]) -> str:
    """Compact structural payload for the agent prompt."""
    payload = {
        'design_type': design.design_type,
        'is_paired': design.is_paired,
        'grouping_variable': design.grouping_variable,
        'subject_id_column': design.subject_id_column,
        'suggested_groups': design.suggested_groups,
        'summary': structural_summary,
    }
    return json.dumps(payload, default=str)


def _run_initial_insights_agent(
    state: WorkflowState,
    design: StatisticalDesign,
    structural_summary: dict[str, Any],
) -> InitialInsightsAgentResults:
    """Run the LLM-assisted column/role proposal phase."""
    agent = _build_initialization_agent(state)
    structural_prompt = _build_structural_prompt(design, structural_summary)
    results = execute_with_backoff(
        lambda: agent.run_sync(
            user_prompt=(
                'Analyze the data and suggest the appropriate test. '
                'Use the structural summary to stay consistent with detected design. '
                f'STRUCTURAL SUMMARY: {structural_prompt}'
            ),
            deps=InitialInsightsAgentDeps(
                user_input='Perform a statistical test.',
                input_data=state.df,
                columns_decision=None,
            ),
        ),
        on_retry=lambda attempt, delay, exc: state.add_step(
            step='Rate limit backoff',
            detail=f'Retrying initialization in {delay:.1f}s (attempt {attempt})',
            data={'error': str(exc)},
        ),
    )
    return results.data


def _sanitize_grouping_column(
    state: WorkflowState,
    df: pd.DataFrame,
    results: InitialInsightsAgentResults,
) -> tuple[InitialInsightsAgentResults, str | None]:
    """Apply grouping guardrails and record warnings when needed."""
    sanitized_group, grouping_warning = _guard_grouping_column(df, results.group_column)
    dropped_group = None
    if sanitized_group != results.group_column:
        dropped_group = results.group_column
        results.group_column = sanitized_group
        if not results.index_column and dropped_group:
            results.index_column = dropped_group
        if grouping_warning:
            state.add_step(
                step='Grouping Guardrail',
                detail=grouping_warning,
                data={'dropped_group_column': dropped_group},
            )
    return results, dropped_group


def _apply_agent_transformations(
    state: WorkflowState,
    results: InitialInsightsAgentResults,
) -> pd.DataFrame:
    """Apply agent-requested data transformations and validate tool arguments."""
    tool_args = results.tool_arguments
    if results.data_transformation != 'None':
        if tool_args is None:
            raise ValueError('tool_arguments must be provided when data_transformation is set.')
        validated = validate_tool_args(results.data_transformation, tool_args)
        state.df = TOOL_FUNCS[results.data_transformation](state.df, **validated)

    return state.df if isinstance(state.df, pd.DataFrame) else pd.DataFrame(state.df)


def _reconcile_design_and_blueprint(
    state: WorkflowState,
    df: pd.DataFrame,
    results: InitialInsightsAgentResults,
    baseline_design: StatisticalDesign,
) -> tuple[StatisticalDesign, dict[str, Any] | None, str | None, str | None]:
    """Run deterministic design validation, pairing reconciliation, and blueprint assembly."""
    validated_design = validate_statistical_design(
        df,
        dependent_var=results.analysis_columns or list(df.columns),
        group_var=results.group_column or baseline_design.grouping_variable,
        subject_id=baseline_design.subject_id_column,
    )
    state.statistical_design = validated_design
    state.paired = validated_design.is_paired
    state.comparison_matrix = validated_design.comparison_matrix or state.comparison_matrix

    index_column = (
        validated_design.subject_id_column or baseline_design.subject_id_column or results.index_column or df.index.name
    )
    index_column = index_column if isinstance(index_column, str) else None
    target_column = (results.analysis_columns or [validated_design.dependent_variable or None])[0]
    partition_report = build_partition_report(
        df, results.group_column or validated_design.grouping_variable, index_column
    )
    if partition_report and partition_report.get('overlap', {}).get('overlap_count'):
        state.paired = True
        validated_design.is_paired = True
        validated_design.design_type = 'paired'
        results.data_design = 'paired'
    if not validated_design.subject_id_column and index_column:
        validated_design.subject_id_column = index_column

    results.partition_report = partition_report
    results.index_column = index_column
    results.target_column = target_column

    blueprint = build_data_blueprint(
        df,
        dependent_vars=results.analysis_columns or list(df.columns),
        group_var=results.group_column or validated_design.grouping_variable,
        covariates=[],
        is_paired=validated_design.is_paired,
        index_column=index_column,
        target_column=target_column,
        partition_report=partition_report,
        raw_payload=results.model_dump(),
    )
    state.attach_blueprint(blueprint)

    return validated_design, partition_report, index_column, target_column


def _preview_route_proposal(state: WorkflowState, agent_route: list[Any]) -> dict[str, Any]:
    """Compute deterministic route suggestion without mutating state."""
    preview = state.model_copy(deep=True)
    preview.pending_routing_decision = None
    try:
        engine_route = decision_engine.evaluate_routing(preview)
    except Exception as exc:  # pragma: no cover - defensive
        engine_route = None
        return {'engine_route': None, 'agent_route': agent_route, 'error': str(exc)}
    return {'engine_route': engine_route, 'agent_route': agent_route}


def _guard_grouping_column(df: pd.DataFrame, group_col: str | None) -> tuple[str | None, str | None]:
    """Block ID-like grouping suggestions while allowing recovery."""
    if not group_col or group_col not in df.columns:
        return group_col, None

    total = len(df)
    nunique = int(df[group_col].nunique(dropna=False))
    if nunique >= max(int(total * 0.6), 25):
        warning = (
            f'Grouping column "{group_col}" is high-cardinality ({nunique} unique of {total}); '
            'treating it as an identifier and dropping it from grouping.'
        )
        return None, warning

    counts = df[group_col].value_counts(dropna=False)
    if total >= 10 and all(count < 2 for count in counts):
        raise RoutingError(
            f'Grouping column "{group_col}" yields singleton groups across {total} rows; cannot route analysis.'
        )

    return group_col, None


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
        updated = state.data_blueprint.model_copy(update={'is_paired': True, 'group_samples': samples})
        state.attach_blueprint(updated)

    return state


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
            if getattr(role, 'role', '') == 'group' and role.name in df.columns:
                group_col = role.name
                break
    if group_col is None:
        fallback_groups = [col for col in df.columns if not pd.api.types.is_numeric_dtype(df[col])]
        group_col = fallback_groups[0] if fallback_groups else None

    if not group_col or group_col not in df.columns:
        raise ValueError('Grouping column could not be inferred.')

    dep_candidates: list[str] = []
    if design and design.dependent_variable:
        dep_candidates.extend([col.strip(" []'\"") for col in str(design.dependent_variable).split(',') if col])
    if blueprint:
        dep_candidates.extend(
            [role.name for role in blueprint.variable_roles if getattr(role, 'role', '') == 'dependent']
        )
    value_col = next((c for c in dep_candidates if c in df.columns and pd.api.types.is_numeric_dtype(df[c])), None)
    if value_col is None:
        numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col]) and col != group_col]
        value_col = numeric_cols[0] if numeric_cols else None

    if not value_col:
        raise ValueError('No numeric measurement column available for group comparison.')

    return df, group_col, value_col


def _extract_group_arrays(df: pd.DataFrame, group_col: str, value_col: str) -> tuple[list[np.ndarray], list[str]]:
    """Return per-group numeric arrays and labels."""
    groups: list[np.ndarray] = []
    labels: list[str] = []
    for label, series in df.groupby(group_col)[value_col]:
        arr = pd.to_numeric(series, errors='coerce').dropna().to_numpy()
        if arr.size == 0:
            continue
        groups.append(arr)
        labels.append(str(label))
    if len(groups) < 2:
        raise ValueError('At least two non-empty groups are required.')
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
        raise ValueError('Subject identifier column is required for repeated measures.')

    within_col = None
    if design and design.grouping_variable and design.grouping_variable in frame.columns:
        within_col = design.grouping_variable
    elif blueprint:
        for role in blueprint.variable_roles:
            if getattr(role, 'role', '') == 'group' and role.name in frame.columns:
                within_col = role.name
                break
    if within_col is None:
        categorical_cols = [
            col for col in frame.columns if not pd.api.types.is_numeric_dtype(frame[col]) and col != subject_col
        ]
        within_col = categorical_cols[0] if categorical_cols else None

    if within_col is None or within_col not in frame.columns:
        raise ValueError('Within-subject factor column is required for repeated measures.')

    dep_candidates: list[str] = []
    if design and design.dependent_variable:
        dep_candidates.extend([col.strip(" []'\"") for col in str(design.dependent_variable).split(',') if col])
    if blueprint:
        dep_candidates.extend(
            [role.name for role in blueprint.variable_roles if getattr(role, 'role', '') == 'dependent']
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
        raise ValueError('No numeric dependent variable column found for repeated measures.')

    return frame, value_col, subject_col, [within_col]


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
        fallback_func = getattr(test_agent, '_statmate_test_function', None)
        test_params_payload: dict[str, Any] = {'alpha': alpha}
        if state.statistical_design and callable(fallback_func):
            func_sig = inspect.signature(fallback_func)
            if 'design_type' in func_sig.parameters:
                test_params_payload['design_type'] = state.statistical_design.design_type
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
                test_type=assumption_test_type or getattr(test_agent, '_statmate_test_name', test_agent.name),
                secondary_data=secondary,
            )
            assumption_entry = {**diag, 'node': test_agent.name, 'timestamp': datetime.utcnow().isoformat()}
            state.add_assumption_entry(assumption_entry)

        # Build a descriptive prompt – Anthropic rejects empty text content blocks.
        agent_prompt = f'Run the {test_agent.name} on the provided data. Alpha={alpha}. Return structured results.'
        result = execute_with_backoff(
            lambda: run_sync_agent(test_agent, user_prompt=agent_prompt, deps=deps),
            on_retry=lambda attempt, delay, exc: state.add_step(
                step='Rate limit backoff',
                detail=f'Retrying {test_agent.name} in {delay:.1f}s (attempt {attempt})',
                data={'error': str(exc)},
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
            getattr(result.statistical_test_result, 'test_name', None)
            or getattr(test_agent, '_statmate_test_name', None)
            or test_agent.name
        )
        stats_value = result.statistical_test_result.statistics
        if isinstance(stats_value, np.ndarray):
            stats_value = stats_value.tolist()
        specifics = result.statistical_test_result.test_specifics or {}
        effect_entry = None
        if isinstance(specifics, dict):
            effect_entry = specifics.get('effect_size')
        effect_size_value = None
        if isinstance(effect_entry, dict):
            effect_size_value = effect_entry.get('value')
        elif isinstance(effect_entry, (int, float, np.floating)):
            effect_size_value = effect_entry
        if isinstance(effect_size_value, np.floating):
            effect_size_value = float(effect_size_value)
        state.add_step(
            step=test_label,
            detail=result.result,
            data={
                'test_name': test_label,
                'statistics': float(stats_value) if isinstance(stats_value, (float, int, np.floating)) else stats_value,
                'null_hypothesis': result.statistical_test_result.null_hypothesis,
                'alternative': result.statistical_test_result.alternative,
                'effect_size_type': result.statistical_test_result.effect_size_type,
                'effect_size': effect_size_value,
                'confidence_interval': result.statistical_test_result.confidence_interval,
                'comments': result.comments,
                'assumptions': assumption_entry,
                'test_specifics': specifics,
            },
            p_value=p_float,
        )

        return state
    except Exception as e:
        logger.error(f'Error in call_test_agent {test_agent.name}: {e}')
        raise NodeExecutionError(node_name=f'call_test_agent({test_agent.name})', original_error=e) from e


def call_initialization_agent(state: WorkflowState) -> WorkflowState:
    """Run the initialization agent to analyze data and suggest tests.

    Args:
        state: Current workflow state.

    Returns:
        Updated workflow state.

    Raises:
        NodeExecutionError: If the initialization fails.
    """
    try:
        structural = check_structural_validity(state.df)
        if not structural.is_valid:
            state.add_step(
                step='Structural Validation (pre-agent)',
                detail='Structural validation failed.',
                data={'errors': structural.errors},
            )
            raise NodeExecutionError(
                node_name='call_initialization_agent', original_error=ValueError('Structural validation failed')
            )

        state.statistical_design = structural.design
        state.paired = structural.design.is_paired if structural.design else state.paired
        state.comparison_matrix = structural.design.comparison_matrix if structural.design else state.comparison_matrix
        state.add_step(
            step='Structural Validation (pre-agent)',
            detail=structural.design.rationale if structural.design else 'Structural design check completed.',
            data={
                'statistical_design': structural.design.as_dict() if structural.design else None,
                'structural_summary': structural.structural_summary,
            },
        )

        model = create_model(model_name=state.model_name, provider=state.provider)
        settings = create_model_settings(model_name=state.model_name)

        roles = propose_column_roles(
            df=state.df,
            structural=structural,
            model=model,
            model_settings=settings,
        )
        if roles.raw_response:
            state.add_step(
                step='Initialization Agent (raw)',
                detail='Captured raw column role agent output.',
                data={'response': roles.raw_response},
            )

        # Apply transformation if any
        if roles.data_transformation != 'None':
            try:
                state.df = TOOL_FUNCS[roles.data_transformation](state.df, **roles.tool_arguments)
            except Exception as exc:
                logger.warning('Failed to apply transformation %s: %s', roles.data_transformation, exc)
                roles.data_transformation = 'None'
                roles.tool_arguments = {}

        inp_df = state.df if isinstance(state.df, pd.DataFrame) else pd.DataFrame(state.df)

        validated_design = validate_statistical_design(
            inp_df,
            dependent_var=roles.analysis_columns or list(inp_df.columns),
            group_var=roles.group_column or (structural.design.grouping_variable if structural.design else None),
            subject_id=structural.design.subject_id_column if structural.design else None,
        )

        state.statistical_design = validated_design
        state.paired = validated_design.is_paired
        state.comparison_matrix = validated_design.comparison_matrix or state.comparison_matrix

        index_column = validated_design.subject_id_column or inp_df.index.name
        index_column = index_column if isinstance(index_column, str) else None
        target_column = (roles.analysis_columns or [validated_design.dependent_variable or None])[0]
        partition_report = build_partition_report(
            inp_df,
            roles.group_column or validated_design.grouping_variable,
            index_column,
        )

        blueprint = build_data_blueprint(
            inp_df,
            dependent_vars=roles.analysis_columns or list(inp_df.columns),
            group_var=roles.group_column or validated_design.grouping_variable,
            covariates=[],
            is_paired=validated_design.is_paired,
            index_column=index_column,
            target_column=target_column if isinstance(target_column, str) else None,
            partition_report=partition_report,
            raw_payload={
                'analysis_columns': roles.analysis_columns,
                'group_column': roles.group_column,
                'data_transformation': roles.data_transformation,
                'tool_arguments': roles.tool_arguments,
                'data_type': roles.data_type,
                'data_design': roles.data_design,
            },
        )
        state.attach_blueprint(blueprint)

        structural_text = (
            get_structural_summary(inp_df, roles.group_column) if roles.group_column else 'No grouping column provided.'
        )
        state.add_step(
            step='Structural Validation',
            detail=validated_design.rationale or 'Structural design confirmed.',
            data={
                'statistical_design': validated_design.as_dict(),
                'structural_summary': structural.structural_summary,
                'structural_text': structural_text,
                'data_blueprint': blueprint.model_dump(),
            },
        )

        route = propose_route(structural=structural, roles=roles, blueprint=blueprint)
        state.pending_routing_decision = {
            'primary': route.primary,
            'alternatives': route.alternatives,
            'reason': route.metadata.get('reason'),
            'profile': route.metadata,
        }

        from statmate.agents.initial_insights_agent import InitialInsightsAgentResults
        from statmate.agents.initial_insights_agent import NodeName as AgentNodeName

        rec = InitialInsightsAgentResults(
            analysis_columns=roles.analysis_columns,
            group_column=roles.group_column,
            output_format='pd.DataFrame',
            data_analysis_result='Initialization pipeline completed.',
            route_to_test=[AgentNodeName(str(value)) for value in route.ordered_nodes],
            comments='Generated by phased initialization pipeline.',
            data_type=roles.data_type,
            data_design=roles.data_design,
            data_transformation=roles.data_transformation,
            tool_arguments=roles.tool_arguments,
            data_size=int(inp_df.shape[0]),
            number_of_columns=int(inp_df.shape[1]),
            variable_roles=[],
            distribution_metrics={},
            sample_balance=None,
            index_column=index_column if isinstance(index_column, str) else None,
            target_column=target_column if isinstance(target_column, str) else None,
            partition_report=partition_report,
        )

        formatted = format_data_by_recommendation(inp_df, rec, state.statistical_design)
        if isinstance(formatted, tuple):
            state.df, state.secondary_df = formatted
        else:
            state.df = formatted

        state.data_type = roles.data_type if roles.data_type in ('CONTINUOUS', 'CATEGORICAL') else None
        state.target_columns = roles.analysis_columns if roles.analysis_columns else list(inp_df.columns)
        state.agent_design_hypothesis = roles.data_design
        state.add_result(AIMessage(content=str(rec)))
        state.add_step(
            step='Initialization',
            detail='Initialization pipeline completed.',
            data={
                'route_to_test': route.ordered_nodes,
                'data_type': roles.data_type,
                'analysis_columns': roles.analysis_columns,
                'group_column': roles.group_column,
                'data_transformation': roles.data_transformation,
                'tool_arguments': roles.tool_arguments,
                'data_design': roles.data_design,
            },
        )

        logger.info('Initialization pipeline completed.')
        return state
    except RoutingError as e:
        logger.error('Routing error in initialization: %s', e)
        raise
    except Exception as e:
        logger.error(f'Error in call_initialization_agent: {e}')
        raise NodeExecutionError(node_name='call_initialization_agent', original_error=e) from e


def design_verification_node(state: WorkflowState) -> WorkflowState:
    """Verify agent design hypothesis against the deterministic structural design.

    Contract: when a mismatch is detected, record it on ``state.design_verification`` and
    allow routing to proceed to ``DESIGN_RECONCILIATION`` rather than raising. This preserves
    the deterministic structural design for downstream routing while giving reconciliation a
    chance to resolve agent/structure disagreements.
    """
    try:
        design = state.statistical_design
        agent_design = state.agent_design_hypothesis
        mismatch = bool(design and agent_design and agent_design != design.design_type)

        detail = 'Structural and agent designs are aligned.'
        if mismatch and design:
            detail = (
                f'Design mismatch: validator={design.design_type} vs agent={agent_design}. Routing to reconciliation.'
            )
        state.design_verification = {
            'mismatch': mismatch,
            'structural_design': design.as_dict() if design else None,
            'agent_design': agent_design,
        }
        if design:
            # Keep downstream routing consistent with deterministic structural design.
            state.paired = design.is_paired
        state.add_step(
            step='Design Verification',
            detail=detail,
            data=state.design_verification,
        )
        return state
    except Exception as e:
        logger.error(f'Error in design_verification_node: {e}')
        raise NodeExecutionError(node_name='design_verification', original_error=e) from e


def design_reconciliation_node(state: WorkflowState) -> WorkflowState:
    """Resolve structural mismatches, prioritizing wide-format pairing cues."""
    try:
        df = state.df if isinstance(state.df, pd.DataFrame) else pd.DataFrame(state.df)
        blueprint = state.data_blueprint
        verification = state.design_verification or {}
        wide_detection = detect_wide_format_pairing(list(df.columns))
        partition = blueprint.partition_report if blueprint else None
        overlap = (partition or {}).get('overlap', {})
        overlap_detected = bool(overlap.get('overlap_count'))

        # Prefer agent-specified grouping role when available
        group_col = None
        if blueprint and blueprint.variable_roles:
            for role in blueprint.variable_roles:
                if getattr(role, 'role', '') == 'group':
                    group_col = role.name
                    break
        if state.statistical_design and state.statistical_design.grouping_variable:
            group_col = group_col or state.statistical_design.grouping_variable

        resolved_design = state.statistical_design
        detail_parts: list[str] = []
        if wide_detection.get('detected'):
            detail_parts.append('Wide-format pairing detected from column names.')
            resolved_design = resolved_design or StatisticalDesign(
                design_type='paired',
                is_paired=True,
                grouping_variable=group_col,
                subject_id_column=getattr(state.statistical_design, 'subject_id_column', None),
                rationale='',
                keyword_cues=state.statistical_design.keyword_cues if state.statistical_design else {},
            )
            resolved_design.design_type = 'paired'
            resolved_design.is_paired = True
            resolved_design.rationale = (
                wide_detection.get('reason') or 'Detected paired measurement columns; coercing to paired design.'
            )
            resolved_design.overlap_summary = resolved_design.overlap_summary or {}
            resolved_design.overlap_summary['wide_format_pairs'] = wide_detection.get('pairs', [])
            state = _force_pairing_transform(state, df, wide_detection.get('pairs', []))
        elif overlap_detected and resolved_design and not resolved_design.is_paired:
            detail_parts.append('Subject overlap across groups indicates paired/mixed design.')
            resolved_design.design_type = 'paired'
            resolved_design.is_paired = True
            resolved_design.overlap_summary = resolved_design.overlap_summary or {}
            resolved_design.overlap_summary['partition_overlap'] = overlap

        if resolved_design:
            state.statistical_design = resolved_design
            state.paired = resolved_design.is_paired
            if blueprint:
                updates: dict[str, Any] = {'is_paired': resolved_design.is_paired}
                if wide_detection.get('pairs') and state.secondary_df is not None:
                    samples = {
                        str(wide_detection['pairs'][0][0]): int(len(state.df)),
                        str(wide_detection['pairs'][0][1]): int(len(state.secondary_df)),
                    }
                    updates['group_samples'] = samples
                state.attach_blueprint(blueprint.model_copy(update=updates))

        state.design_verification = (verification or {}) | {'mismatch': False, 'resolved': True}
        state.add_step(
            step=NodeName.DESIGN_RECONCILIATION,
            detail='; '.join(detail_parts) or 'Design reconciliation completed.',
            data={
                'wide_format_detection': wide_detection,
                'partition_overlap': overlap,
                'resolved_design': resolved_design.as_dict() if resolved_design else None,
            },
        )
        return state
    except Exception as e:
        logger.error(f'Error in design_reconciliation_node: {e}')
        raise NodeExecutionError(node_name='design_reconciliation_node', original_error=e) from e


def assess_study_design_node(state: WorkflowState) -> WorkflowState:
    """Assess whether the study design is paired or independent.

    Args:
        state: Current workflow state.

    Returns:
        Updated workflow state.

    Raises:
        NodeExecutionError: If the assessment fails.
    """
    try:
        if state.statistical_design:
            state.paired = state.statistical_design.is_paired
            state.add_step(
                step='Assess Study Design',
                detail='Using structural validator output',
                data=state.statistical_design.as_dict(),
            )
            return state

        # Use model from state if specified
        model = create_model(model_name=state.model_name, provider=state.provider)
        settings = create_model_settings(model_name=state.model_name)
        agent = get_assess_design_study_agent(model=model, model_settings=settings)

        logger.info('assess_study_design_node\nmodel is fed with those data:')
        logger.info(state.results)

        res = execute_with_backoff(
            lambda: agent.run_sync(deps=AssessDesignDeps(msg=state.results)),
            on_retry=lambda attempt, delay, exc: state.add_step(
                step='Rate limit backoff',
                detail=f'Retrying study design check in {delay:.1f}s (attempt {attempt})',
                data={'error': str(exc)},
            ),
        )
        state.paired = res.data.paired

        msg = '~~~Paired comparison~~~' if res.data.paired else '~~~Two independent groups~~~'
        logger.info(msg)
        state.add_step(
            step='Assess Study Design',
            detail='Paired comparison' if res.data.paired else 'Two independent groups',
            data={'paired': res.data.paired},
        )

        return state
    except Exception as e:
        logger.error(f'Error in assess_study_design_node: {e}')
        raise NodeExecutionError(node_name='assess_study_design_node', original_error=e) from e


def two_independent_node(
    state: WorkflowState,
    shapiro_agent_func=shapiro_wilk_agent,
    levene_agent_func=None,
) -> WorkflowState:
    """Run normality and variance tests for two independent groups.

    This node runs Shapiro-Wilk test on each group and Levene's test
    for equality of variances.

    Args:
        state: Current workflow state.
        shapiro_agent_func: Function to create Shapiro-Wilk agent.
        levene_agent_func: Function to create Levene agent. If None, imports from agents.

    Returns:
        Updated workflow state.

    Raises:
        NodeExecutionError: If the node execution fails.
    """
    if levene_agent_func is None:
        from statmate.agents import levene_agent as levene_agent_func

    try:
        # Save original data
        secondary_df = state.secondary_df
        orig_df = state.df

        # Test group 1
        state.secondary_df = None
        model = create_model(model_name=state.model_name, provider=state.provider)
        settings = create_model_settings(model_name=state.model_name)
        agent1 = shapiro_agent_func(model=model, model_settings=settings)
        state = call_test_agent(agent1, state, probability_key='shapiro_group1')
        p1 = state.get_probability('shapiro_group1', 0)

        # Test group 2
        if secondary_df is not None:
            state.df = secondary_df
            agent2 = shapiro_agent_func(model=model, model_settings=settings)
            state = call_test_agent(agent2, state, probability_key='shapiro_group2')
        else:
            logger.error('secondary_df is None, cannot test group 2')
            raise ValueError('secondary_df is required for two independent groups test')

        p2 = state.get_probability('shapiro_group2', 0)

        # Restore original data
        state.df = orig_df
        state.secondary_df = secondary_df

        # Store distinct keys
        state.add_probability('shapiro_group1', p1)
        state.add_probability('shapiro_group2', p2)

        # Levene's test for equal variances
        levene_agent_inst = levene_agent_func(model=model, model_settings=settings)
        state = call_test_agent(levene_agent_inst, state, probability_key='levene')

        return state
    except Exception as e:
        logger.error(f'Error in two_independent_node: {e}')
        raise NodeExecutionError(node_name='two_independent_node', original_error=e) from e


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

        levene_stat, levene_p = scipy.stats.levene(*groups, center='median')
        variance_pass = float(levene_p) >= alpha
        normal_pass = min_normal is None or min_normal >= alpha
        status = 'pass' if variance_pass and normal_pass else 'fail'

        assumption_entry = {
            'node': NodeName.ANOVA_ASSUMPTIONS,
            'group_column': group_col,
            'value_column': value_col,
            'shapiro_p_values': shapiro_pvalues,
            'levene_statistic': float(levene_stat),
            'levene_p_value': float(levene_p),
            'alpha': alpha,
            'status': status,
        }
        state.add_assumption_entry(assumption_entry)
        if min_normal is not None:
            state.add_probability('anova_min_shapiro', float(min_normal))
        state.add_probability('anova_levene', float(levene_p))
        state.add_step(
            step=NodeName.ANOVA_ASSUMPTIONS,
            detail='ANOVA assumption check (Shapiro per group + Levene)',
            data=assumption_entry,
            p_value=float(levene_p),
        )
        state.pending_routing_decision = (state.pending_routing_decision or {}) | {
            'group_column': group_col,
            'value_column': value_col,
        }
        return state
    except Exception as e:
        logger.error(f'Error in anova_assumptions_node: {e}')
        raise NodeExecutionError(node_name='anova_assumptions_node', original_error=e) from e


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
            probability_key='anova_one_way',
            assess_assumptions=False,
            test_params={'group_column': group_col, 'value_column': value_col},
        )
    except Exception as e:
        logger.error(f'Error in anova_one_way_node: {e}')
        raise NodeExecutionError(node_name='anova_one_way_node', original_error=e) from e


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
            probability_key='kruskal_wallis',
            assess_assumptions=False,
            test_params={'group_column': group_col, 'value_column': value_col, 'perform_dunn': True},
        )
    except Exception as e:
        logger.error(f'Error in kruskal_wallis_node: {e}')
        raise NodeExecutionError(node_name='kruskal_wallis_node', original_error=e) from e


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
            probability_key='anova_rm',
            assess_assumptions=False,
            test_params={
                'dependent_variable': value_col,
                'subject': subject_col,
                'within': within_cols,
            },
        )
    except Exception as e:
        logger.error(f'Error in anova_rm_node: {e}')
        raise NodeExecutionError(node_name='anova_rm_node', original_error=e) from e


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
            probability_key='friedman_test',
            assess_assumptions=False,
            test_params={
                'dependent_variable': value_col,
                'subject': subject_col,
                'within': within_cols,
            },
        )
    except Exception as e:
        logger.error(f'Error in friedman_node: {e}')
        raise NodeExecutionError(node_name='friedman_node', original_error=e) from e


def nonparametric_node(
    state: WorkflowState,
    welch_agent_func=None,
    mann_whitney_agent_func=None,
) -> WorkflowState:
    """Run nonparametric tests (Welch's t-test and Mann-Whitney U).

    Args:
        state: Current workflow state.
        welch_agent_func: Function to create Welch's t-test agent.
        mann_whitney_agent_func: Function to create Mann-Whitney U agent.

    Returns:
        Updated workflow state.

    Raises:
        NodeExecutionError: If the node execution fails.
    """
    if welch_agent_func is None:
        from statmate.agents import welch_t_agent as welch_agent_func
    if mann_whitney_agent_func is None:
        from statmate.agents import mannwhitneyu_agent as mann_whitney_agent_func

    try:
        model = create_model(model_name=state.model_name, provider=state.provider)
        settings = create_model_settings(model_name=state.model_name)

        welch_agent = welch_agent_func(model=model, model_settings=settings)
        state = call_test_agent(welch_agent, state, probability_key='welch_t_test')

        mann_agent = mann_whitney_agent_func(model=model, model_settings=settings)
        state = call_test_agent(mann_agent, state, probability_key='mann_whitney_u')

        return state
    except Exception as e:
        logger.error(f'Error in nonparametric_node: {e}')
        raise NodeExecutionError(node_name='nonparametric_node', original_error=e) from e


def summariser_node(state: WorkflowState) -> WorkflowState:
    """Generate a summary of all performed tests.

    Args:
        state: Current workflow state.

    Returns:
        Updated workflow state with summary.

    Raises:
        NodeExecutionError: If summary generation fails.
    """
    try:
        model = create_model(model_name=state.model_name, provider=state.provider)
        settings = create_model_settings(model_name=state.model_name)

        deps = SummariserDeps(
            results=state.results,
            performed_tests=list(state.probabilities.keys()),
        )

        agent = get_summariser_agent(model, settings)
        res = execute_with_backoff(
            lambda: agent.run_sync(deps=deps),
            on_retry=lambda attempt, delay, exc: state.add_step(
                step='Rate limit backoff',
                detail=f'Retrying summary in {delay:.1f}s (attempt {attempt})',
                data={'error': str(exc)},
            ),
        )

        state.add_result(AIMessage(content=str(res.data)))
        logger.info(f'Summariser output: {res.data}')
        state.add_step(
            step='Summary',
            detail=res.data.summary if hasattr(res, 'data') and hasattr(res.data, 'summary') else str(res.data),
            data={'performed_tests': deps.performed_tests},
        )

        return state
    except Exception as e:
        logger.error(f'Error in summariser_node: {e}')
        raise NodeExecutionError(node_name='summariser_node', original_error=e) from e


def reviewer_node(state: WorkflowState) -> WorkflowState:
    """Review the generated summary against raw outputs to catch hallucinations."""
    try:
        model = create_model(model_name=state.model_name, provider=state.provider)
        settings = create_model_settings(model_name=state.model_name)

        summary_text = ''
        if state.results:
            summary_text = str(state.results[-1].content)

        agent = get_reviewer_agent(model, settings)
        deps = ReviewerDeps(summary=summary_text, results=state.results, probabilities=state.probabilities)
        res = execute_with_backoff(
            lambda: agent.run_sync(deps=deps),
            on_retry=lambda attempt, delay, exc: state.add_step(
                step='Rate limit backoff',
                detail=f'Retrying reviewer in {delay:.1f}s (attempt {attempt})',
                data={'error': str(exc)},
            ),
        )

        state.reviewer_report = res.data.model_dump()
        adjusted_summary = res.data.adjusted_summary or summary_text

        # Track reviewer decision for downstream clients
        state.add_result(AIMessage(content=str(res.data)))
        state.add_step(
            step='Reviewer',
            detail='Approved summary' if res.data.approved else 'Adjusted summary to match evidence',
            data={
                'approved': res.data.approved,
                'risk_score': res.data.risk_score,
                'flags': res.data.hallucination_flags,
                'adjusted_summary': adjusted_summary,
            },
        )

        # Preserve vetted summary for API consumers
        state.test_hierarchy = state.test_hierarchy or {}
        state.test_hierarchy['reviewer'] = state.reviewer_report
        return state
    except Exception as e:
        logger.error(f'Error in reviewer_node: {e}')
        raise NodeExecutionError(node_name='reviewer_node', original_error=e) from e


def intent_discovery_node(state: WorkflowState) -> WorkflowState:
    """Lightweight intent check to trigger ambiguity modal when confidence is low."""
    try:
        confidence = 0.5
        if state.target_columns:
            confidence = 0.9
        elif state.statistical_design:
            confidence = 0.8

        summary = state.intent_summary or 'Explore relationships in the provided data.'
        trigger_modal = confidence < 0.8
        state.intent_confidence = confidence
        state.intent_summary = summary
        state.add_step(
            step=NodeName.INTENT,
            detail=summary,
            data={
                'confidence': confidence,
                'trigger_ambiguity_modal': trigger_modal,
            },
        )
        return state
    except Exception as e:  # pragma: no cover - defensive
        logger.error(f'Error in intent_discovery_node: {e}')
        raise NodeExecutionError(node_name='intent_discovery_node', original_error=e) from e


def choice_node(state: WorkflowState) -> WorkflowState:
    """Expose routing options to allow user or UI to decide."""
    try:
        decision = state.pending_routing_decision or {}
        primary = decision.get('primary')
        alternatives = decision.get('alternatives') or []

        if state.user_selected_option:
            valid_options = [opt for opt in [primary, *alternatives] if opt]
            if state.user_selected_option not in valid_options:
                state.add_step(
                    step=NodeName.CHOICE,
                    detail='Invalid override ignored; falling back to default option.',
                    data={
                        'invalid_override': state.user_selected_option,
                        'valid_options': valid_options,
                    },
                )
                state.user_selected_option = None

        selected = state.user_selected_option or decision.get('selected') or primary

        entry = {
            'primary': primary,
            'alternatives': alternatives,
            'selected': selected,
            'reason': decision.get('reason'),
        }
        state.choice_log.append(entry)
        state.pending_routing_decision = {**decision, 'selected': selected}
        state.add_step(step=NodeName.CHOICE, detail=f'Chosen {selected or primary}', data=entry)
        return state
    except Exception as e:  # pragma: no cover - defensive
        logger.error('Error in choice_node: %s', e)
        raise NodeExecutionError(node_name='choice_node', original_error=e) from e


def resolve_choice(state: WorkflowState) -> str:
    """Resolve the chosen next node after presenting options."""
    decision = state.pending_routing_decision or {}
    return decision.get('selected') or decision.get('primary') or NodeName.ASSESS_STUDY_DESIGN


def mcnemar_node(state: WorkflowState) -> WorkflowState:
    """Execute McNemar's test on paired categorical data."""
    try:
        from statmate.agents import mcnemar_agent

        model = create_model(model_name=state.model_name, provider=state.provider)
        settings = create_model_settings(model_name=state.model_name)
        agent = mcnemar_agent(model=model, model_settings=settings)
        return call_test_agent(agent, state, probability_key='mcnemar', assess_assumptions=False)
    except Exception as e:  # pragma: no cover - defensive
        logger.error(f'Error in mcnemar_node: {e}')
        raise NodeExecutionError(node_name='mcnemar_node', original_error=e) from e


def cox_regression_node(state: WorkflowState) -> WorkflowState:
    """Cox proportional hazards regression — not yet implemented."""
    error_msg = (
        'Survival analysis (Cox regression) is not yet supported in this version. '
        'Please use a dedicated survival analysis tool or contact the team to prioritise this feature.'
    )
    state.add_step(
        step=NodeName.COX_REGRESSION,
        detail=error_msg,
        data={'status': 'NOT_IMPLEMENTED', 'error_message': error_msg},
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
    intent = blueprint.regression_intent if blueprint else 'none'
    target_col = blueprint.target_column if blueprint else None

    if intent == 'none' or not target_col:
        state.add_step(
            step='regression_node',
            detail='No regression intent detected; skipping regression analysis.',
            data={'status': 'skipped'},
        )
        return state

    df = state.df if isinstance(state.df, pd.DataFrame) else state.df.to_frame()
    y = df[target_col]
    X = df.drop(columns=[target_col]).select_dtypes(include='number')

    try:
        if intent == 'logistic':
            result = logistic_regression(X, y)
        elif X.shape[1] > 1:
            result = multiple_regression_with_vif(X, y)
        else:
            result = linear_regression(X, y)
        result_dump = result.model_dump()
        state.add_result(AIMessage(content=json.dumps(result_dump, default=str)))
        state.add_step(
            step='regression_node',
            detail=f'Regression analysis complete ({result.test_name}).',
            data=result_dump,
            p_value=result.p_value if isinstance(result.p_value, float) else None,
        )
    except ValueError as exc:
        logger.error('Error in regression_node: %s', exc)
        state.add_step(
            step='regression_node',
            detail=f'Regression analysis failed: {exc}',
            data={'error': str(exc)},
        )
    return state


def descriptive_summary_node(state: WorkflowState) -> WorkflowState:
    """Fallback node that returns descriptive statistics when inferential tests are blocked."""
    frame = state.df if isinstance(state.df, pd.DataFrame) else state.df.to_frame()
    summary = frame.describe(include='all').to_dict()
    group_samples = state.data_blueprint.group_samples if state.data_blueprint else None
    payload = {'summary': summary, 'group_samples': group_samples}
    state.add_result(AIMessage(content=json.dumps(payload, default=str)))
    state.add_step(
        step=NodeName.DESCRIPTIVE_SUMMARY,
        detail='Insufficient sample size for inferential testing; returning descriptive summary.',
        data=payload,
    )
    return state


def user_intervention_node(state: WorkflowState) -> WorkflowState:
    """Stop the graph and surface a user-facing intervention request."""
    state.add_step(
        step=NodeName.USER_INTERVENTION,
        detail='Routing blocked by guardrails; manual choice required.',
        data={
            'pending_decision': state.pending_routing_decision,
            'blueprint': state.data_blueprint.model_dump() if state.data_blueprint else None,
        },
    )
    return state


def methodology_auditor_node(state: WorkflowState) -> WorkflowState:
    """Audit executed tests and propose corrections when assumptions fail."""
    structure_auditor = StructureAuditor()
    structural = structure_auditor.audit(state)
    auditor = MethodologyAuditor()
    result = auditor.audit(state)
    payload = {
        'executed': result.executed,
        'recommended': result.recommended,
        'correction_step': result.correction_step,
        'conflicts': result.conflicts,
    }
    if structural:
        payload['structural_recommended'] = structural.recommended
        payload['structural_correction'] = structural.correction_step
    state.test_hierarchy = state.test_hierarchy or {}
    state.test_hierarchy['auditor'] = payload
    state.add_step(step=NodeName.METHODOLOGY_AUDITOR, detail='Auditor review complete', data=payload)
    return state
