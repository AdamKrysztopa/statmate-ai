"""Workflow node functions.

This module contains all the node functions for the statistical test workflow,
extracted from the monolithic statmate_flow.py for better organization.
"""

import inspect
import json
from datetime import datetime

import numpy as np
import pandas as pd
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
    build_initial_insights_agent,
    build_partition_report,
    format_data_by_recommendation,
    validate_tool_args,
)
from statmate.agents.reviewer_agent import ReviewerDeps, get_reviewer_agent
from statmate.agents.summarizer_agent import SummariserDeps, get_summariser_agent
from statmate.core import NodeExecutionError, get_logger
from statmate.core.config import NodeName, default_config
from statmate.core.model_provider import execute_with_backoff
from statmate.core.validation import (
    get_structural_summary,
    infer_statistical_design,
    validate_assumptions,
    validate_statistical_design,
)
from statmate.workflow.blueprint import build_data_blueprint
from statmate.workflow.model_factory import create_model, create_model_settings
from statmate.workflow.methodology_auditor import MethodologyAuditor, StructureAuditor
from statmate.workflow.state import WorkflowState

logger = get_logger(__name__)


# Append strict tool_arguments requirement to prompt
ENHANCED_INITIAL_INSIGHTS_PROMPT = (
    INITIAL_INSIGHTS_PROMPT
    + "\nData Validation:\n  - Always include a non-null 'tool_arguments' dict (empty if no transform)."
)


def call_test_agent(
    test_agent: Agent,
    state: WorkflowState,
    alpha: float | None = None,
    probability_key: str | None = None,
    assumption_test_type: str | None = None,
    assumption_secondary: pd.Series | pd.DataFrame | np.ndarray | None = None,
    assess_assumptions: bool = True,
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
        test_params: dict[str, Any] = {'alpha': alpha}
        if state.statistical_design and callable(fallback_func):
            func_sig = inspect.signature(fallback_func)
            if 'design_type' in func_sig.parameters:
                test_params['design_type'] = state.statistical_design.design_type

        deps = StatTestDeps(
            data=state.df,
            data_secondary=state.secondary_df,
            test_params=test_params,
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

        result = execute_with_backoff(
            lambda: run_sync_agent(test_agent, user_prompt='', deps=deps),
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
        test_label = getattr(test_agent, '_statmate_test_name', None) or test_agent.name
        stats_value = result.statistical_test_result.statistics
        if isinstance(stats_value, np.ndarray):
            stats_value = stats_value.tolist()
        state.add_step(
            step=test_label,
            detail=result.result,
            data={
                'test_name': test_label,
                'statistics': float(stats_value)
                if isinstance(stats_value, (float, int, np.floating))
                else stats_value,
                'null_hypothesis': result.statistical_test_result.null_hypothesis,
                'alternative': result.statistical_test_result.alternative,
                'effect_size_type': result.statistical_test_result.effect_size_type,
                'confidence_interval': result.statistical_test_result.confidence_interval,
                'comments': result.comments,
                'assumptions': assumption_entry,
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
        # Deterministic structural validation before any LLM reasoning
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

        # Use model from state if specified
        model = create_model(model_name=state.model_name, provider=state.provider)
        settings = create_model_settings(model_name=state.model_name)
        agent = build_initial_insights_agent(
            model=model,
            system_prompt=ENHANCED_INITIAL_INSIGHTS_PROMPT,
            model_settings=settings,
        )

        structural_prompt = json.dumps(
            {
                'statistical_design': design.as_dict(),
                'structural_summary': structural_summary,
            },
            default=str,
        )

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

        # Update sample count
        state.number_of_samples = int(results.data.data_size)

        # Ensure tool_arguments exists
        tool_args = results.data.tool_arguments or {}

        # Apply transformation if any
        if results.data.data_transformation != 'None':
            validated = validate_tool_args(results.data.data_transformation, tool_args)
            state.df = TOOL_FUNCS[results.data.data_transformation](state.df, **validated)

        inp_df = state.df if isinstance(state.df, pd.DataFrame) else pd.DataFrame(state.df)

        # Re-run structural validation using agent-selected columns to lock design.
        validated_design = validate_statistical_design(
            inp_df,
            dependent_var=results.data.analysis_columns or list(inp_df.columns),
            group_var=results.data.group_column or design.grouping_variable,
            subject_id=design.subject_id_column,
        )
        state.statistical_design = validated_design
        state.paired = validated_design.is_paired
        state.comparison_matrix = validated_design.comparison_matrix or state.comparison_matrix

        index_column = validated_design.subject_id_column or design.subject_id_column or inp_df.index.name
        target_column = (results.data.analysis_columns or [validated_design.dependent_variable or None])[0]
        partition_report = build_partition_report(
            inp_df, results.data.group_column or validated_design.grouping_variable, index_column
        )
        if partition_report and partition_report.get('overlap', {}).get('overlap_count'):
            state.paired = True
            validated_design.is_paired = True
            validated_design.design_type = 'paired'
            results.data.data_design = 'paired'

        results.data.partition_report = partition_report
        results.data.index_column = index_column
        results.data.target_column = target_column

        blueprint = build_data_blueprint(
            inp_df,
            dependent_vars=results.data.analysis_columns or list(inp_df.columns),
            group_var=results.data.group_column or validated_design.grouping_variable,
            covariates=[],
            is_paired=validated_design.is_paired,
            index_column=index_column,
            target_column=target_column,
            partition_report=partition_report,
            raw_payload=results.data.model_dump() if hasattr(results, 'data') else {},
        )
        state.attach_blueprint(blueprint)

        structural_text = (
            get_structural_summary(inp_df, results.data.group_column)
            if results.data.group_column
            else 'No grouping column provided.'
        )
        state.add_step(
            step='Structural Validation',
            detail=validated_design.rationale or 'Structural design confirmed.',
            data={
                'statistical_design': validated_design.as_dict(),
                'structural_summary': structural_summary,
                'structural_text': structural_text,
                'data_blueprint': blueprint.model_dump(),
            },
        )

        # Format for downstream tests
        formatted = format_data_by_recommendation(inp_df, results.data, state.statistical_design)
        if isinstance(formatted, tuple):
            state.df, state.secondary_df = formatted
        else:
            state.df = formatted

        # Set metadata
        state.data_type = results.data.data_type
        cols = results.data.analysis_columns
        state.target_columns = cols if set(cols).issubset(set(inp_df.columns)) else list(inp_df.columns)
        state.agent_design_hypothesis = results.data.data_design
        state.add_result(AIMessage(content=str(results.data)))
        state.add_step(
            step='Initialization',
            detail=results.data.data_analysis_result,
            data={
                'route_to_test': [getattr(r, 'value', str(r)) for r in results.data.route_to_test],
                'data_type': results.data.data_type,
                'analysis_columns': results.data.analysis_columns,
                'group_column': results.data.group_column,
                'data_transformation': results.data.data_transformation,
                'tool_arguments': results.data.tool_arguments,
                'data_design': results.data.data_design,
            },
        )

        logger.info(f'Data type set: {state.data_type}')
        return state
    except Exception as e:
        logger.error(f'Error in call_initialization_agent: {e}')
        raise NodeExecutionError(node_name='call_initialization_agent', original_error=e) from e


def design_verification_node(state: WorkflowState) -> WorkflowState:
    """Verify that agent design hypothesis aligns with structural validator output."""
    try:
        design = state.statistical_design
        agent_design = state.agent_design_hypothesis
        mismatch = bool(design and agent_design and agent_design != design.design_type)

        detail = 'Structural and agent designs are aligned.'
        if mismatch:
            detail = (
                f'Design mismatch: validator={design.design_type} vs agent={agent_design}. '
                'Halting until clarified.'
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
        logger.error(f'Error in choice_node: %s', e)
        raise NodeExecutionError(node_name='choice_node', original_error=e) from e


def resolve_choice(state: WorkflowState) -> str:
    """Resolve the chosen next node after presenting options."""
    decision = state.pending_routing_decision or {}
    return decision.get('selected') or decision.get('primary') or NodeName.ASSESS_STUDY_DESIGN


def mcnemar_node(state: WorkflowState) -> WorkflowState:
    """Placeholder node for McNemar's test on paired categorical data."""
    state.add_step(
        step=NodeName.MCNEMAR,
        detail='McNemar test placeholder (paired categorical).',
        data={'status': 'queued'},
    )
    return state


def cox_regression_node(state: WorkflowState) -> WorkflowState:
    """Placeholder node for Cox regression on survival data."""
    state.add_step(
        step=NodeName.COX_REGRESSION,
        detail='Cox regression placeholder node (survival analysis).',
        data={'status': 'queued'},
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
