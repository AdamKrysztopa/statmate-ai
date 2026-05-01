"""Comparison workflow nodes.

Covers: two_independent_node, nonparametric_node, mcnemar_node.
"""

from statmate.agents import shapiro_wilk_agent
from statmate.core import NodeExecutionError, get_logger
from statmate.workflow._node_helpers import call_test_agent
from statmate.workflow.model_factory import create_model, create_model_settings
from statmate.workflow.state import WorkflowState

logger = get_logger(__name__)


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
        state = call_test_agent(agent1, state, probability_key="shapiro_group1")
        p1 = state.get_probability("shapiro_group1", 0)

        # Test group 2
        if secondary_df is not None:
            state.df = secondary_df
            agent2 = shapiro_agent_func(model=model, model_settings=settings)
            state = call_test_agent(agent2, state, probability_key="shapiro_group2")
        else:
            logger.error("secondary_df is None, cannot test group 2")
            raise ValueError("secondary_df is required for two independent groups test")

        p2 = state.get_probability("shapiro_group2", 0)

        # Restore original data
        state.df = orig_df
        state.secondary_df = secondary_df

        # Store distinct keys
        state.add_probability("shapiro_group1", p1)
        state.add_probability("shapiro_group2", p2)

        # Levene's test for equal variances
        levene_agent_inst = levene_agent_func(model=model, model_settings=settings)
        state = call_test_agent(levene_agent_inst, state, probability_key="levene")

        return state
    except Exception as e:
        logger.error(f"Error in two_independent_node: {e}")
        raise NodeExecutionError(node_name="two_independent_node", original_error=e) from e


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
        state = call_test_agent(welch_agent, state, probability_key="welch_t_test")

        mann_agent = mann_whitney_agent_func(model=model, model_settings=settings)
        state = call_test_agent(mann_agent, state, probability_key="mann_whitney_u")

        return state
    except Exception as e:
        logger.error(f"Error in nonparametric_node: {e}")
        raise NodeExecutionError(node_name="nonparametric_node", original_error=e) from e


def mcnemar_node(state: WorkflowState) -> WorkflowState:
    """Execute McNemar's test on paired categorical data."""
    try:
        from statmate.agents import mcnemar_agent

        model = create_model(model_name=state.model_name, provider=state.provider)
        settings = create_model_settings(model_name=state.model_name)
        agent = mcnemar_agent(model=model, model_settings=settings)
        return call_test_agent(agent, state, probability_key="mcnemar", assess_assumptions=False)
    except Exception as e:  # pragma: no cover - defensive
        logger.error(f"Error in mcnemar_node: {e}")
        raise NodeExecutionError(node_name="mcnemar_node", original_error=e) from e
