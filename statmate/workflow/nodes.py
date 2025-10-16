"""Workflow node functions.

This module contains all the node functions for the statistical test workflow,
extracted from the monolithic statmate_flow.py for better organization.
"""

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
    format_data_by_recommendation,
    validate_tool_args,
)
from statmate.agents.summarizer_agent import SummariserDeps, get_summariser_agent
from statmate.config import default_config
from statmate.exceptions import NodeExecutionError
from statmate.logging_config import get_logger
from statmate.workflow.model_factory import create_model, create_model_settings
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
) -> WorkflowState:
    """Call a statistical agent and append its result.

    Args:
        test_agent: The agent to call.
        state: Current workflow state.
        alpha: Significance level. If None, uses default from config.

    Returns:
        Updated workflow state.

    Raises:
        NodeExecutionError: If the agent execution fails.
    """
    if alpha is None:
        alpha = default_config.statistical.default_alpha

    try:
        deps = StatTestDeps(
            data=state.df,
            data_secondary=state.secondary_df,
            test_params={'alpha': alpha},
        )
        result = run_sync_agent(test_agent, user_prompt='', deps=deps)

        state.add_result(AIMessage(content=str(result)))

        p_val = result.statistical_test_result.p_value
        p_float = float(p_val) if isinstance(p_val, float) else float(np.mean(p_val))
        state.add_probability(test_agent.name, p_float)

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
        model = create_model()
        settings = create_model_settings()
        agent = build_initial_insights_agent(
            model=model,
            system_prompt=ENHANCED_INITIAL_INSIGHTS_PROMPT,
            model_settings=settings,
        )

        results = agent.run_sync(
            user_prompt='Analyze the data and suggest the appropriate test.',
            deps=InitialInsightsAgentDeps(
                user_input='Perform a statistical test.',
                input_data=state.df,
                columns_decision=None,
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

        # Format for downstream tests
        formatted = format_data_by_recommendation(inp_df, results.data)
        if isinstance(formatted, tuple):
            state.df, state.secondary_df = formatted
        else:
            state.df = formatted

        # Set metadata
        state.data_type = results.data.data_type
        cols = results.data.analysis_columns
        state.target_columns = cols if set(cols).issubset(set(inp_df.columns)) else list(inp_df.columns)
        state.add_result(AIMessage(content=str(results.data)))

        logger.info(f'Data type set: {state.data_type}')
        return state
    except Exception as e:
        logger.error(f'Error in call_initialization_agent: {e}')
        raise NodeExecutionError(node_name='call_initialization_agent', original_error=e) from e


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
        model = create_model()
        settings = create_model_settings()
        agent = get_assess_design_study_agent(model=model, model_settings=settings)

        logger.info('assess_study_design_node\nmodel is fed with those data:')
        logger.info(state.results)

        res = agent.run_sync(deps=AssessDesignDeps(msg=state.results))
        state.paired = res.data.paired

        msg = '~~~Paired comparison~~~' if res.data.paired else '~~~Two independent groups~~~'
        logger.info(msg)

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
        model = create_model()
        settings = create_model_settings()
        agent1 = shapiro_agent_func(model=model, model_settings=settings)
        state = call_test_agent(agent1, state)
        p1 = state.probabilities.pop('shapiro_wilk_agent', 0)

        # Test group 2
        if secondary_df is not None:
            state.df = secondary_df
            agent2 = shapiro_agent_func(model=model, model_settings=settings)
            state = call_test_agent(agent2, state)
        else:
            logger.error('secondary_df is None, cannot test group 2')
            raise ValueError('secondary_df is required for two independent groups test')

        p2 = state.probabilities.pop('shapiro_wilk_agent', 0)

        # Restore original data
        state.df = orig_df
        state.secondary_df = secondary_df

        # Store distinct keys
        state.add_probability('shapiro_group1', p1)
        state.add_probability('shapiro_group2', p2)

        # Levene's test for equal variances
        levene_agent_inst = levene_agent_func(model=model, model_settings=settings)
        state = call_test_agent(levene_agent_inst, state)

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
        model = create_model()
        settings = create_model_settings()

        welch_agent = welch_agent_func(model=model, model_settings=settings)
        state = call_test_agent(welch_agent, state)

        mann_agent = mann_whitney_agent_func(model=model, model_settings=settings)
        state = call_test_agent(mann_agent, state)

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
        model = create_model()
        settings = create_model_settings()

        deps = SummariserDeps(
            results=state.results,
            performed_tests=list(state.probabilities.keys()),
        )

        agent = get_summariser_agent(model, settings)
        res = agent.run_sync(deps=deps)

        state.add_result(AIMessage(content=str(res.data)))
        logger.info(f'Summariser output: {res.data}')

        return state
    except Exception as e:
        logger.error(f'Error in summariser_node: {e}')
        raise NodeExecutionError(node_name='summariser_node', original_error=e) from e
