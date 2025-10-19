"""Workflow edge decision functions.

This module contains all the edge/decision functions for the statistical
test workflow, determining which node to execute next based on state.
"""

from langgraph.graph import END

from statmate.core import get_logger
from statmate.core.config import NodeName, default_config
from statmate.workflow.state import WorkflowState

logger = get_logger(__name__)


def decide_outcome(
    state: WorkflowState,
    categorical_threshold: int | None = None,
) -> str:
    """Determine whether to use continuous or categorical analysis path.

    Args:
        state: Current workflow state.
        categorical_threshold: Sample size threshold for chi-square vs Fisher.
            If None, uses config default.

    Returns:
        Name of the next node to execute.
    """
    if categorical_threshold is None:
        categorical_threshold = default_config.statistical.categorical_sample_size_threshold

    try:
        if state.data_type == 'CONTINUOUS':
            return NodeName.ASSESS_STUDY_DESIGN
        return NodeName.CHI2 if state.number_of_samples > categorical_threshold else NodeName.FISHER
    except Exception as e:
        logger.error(f'Error in decide_outcome: {e}')
        return END


def assess_study_design(state: WorkflowState) -> str:
    """Branch on paired vs independent groups.

    Args:
        state: Current workflow state.

    Returns:
        Name of the next node to execute.
    """
    try:
        return NodeName.NORMALITY_OF_DIFFERENCE if state.paired else NodeName.TWO_INDEPENDENT_GROUPS
    except Exception as e:
        logger.error(f'Error in assess_study_design: {e}')
        return END


def parametric_assumptions(
    state: WorkflowState,
    alpha: float | None = None,
) -> str:
    """Decide between parametric and non-parametric tests for paired data.

    Args:
        state: Current workflow state.
        alpha: Significance level. If None, uses config default.

    Returns:
        Name of the next node to execute.
    """
    if alpha is None:
        alpha = default_config.statistical.normality_threshold

    try:
        p_normality = state.get_probability('normality_of_difference', 0)
        return NodeName.PAIRED_T if p_normality > alpha else NodeName.WILCOXON
    except Exception as e:
        logger.error(f'Error in parametric_assumptions: {e}')
        return END


def decide_two_independent(
    state: WorkflowState,
    alpha: float | None = None,
) -> str:
    """Choose between parametric and non-parametric tests for independent groups.

    This decision is based on normality of both groups and equality of variances.

    Args:
        state: Current workflow state.
        alpha: Significance level. If None, uses config default.

    Returns:
        Name of the next node to execute.
    """
    if alpha is None:
        alpha = default_config.statistical.variance_threshold

    try:
        p_shapiro1 = state.get_probability('shapiro_group1', 0)
        p_shapiro2 = state.get_probability('shapiro_group2', 0)
        p_levene = state.get_probability('levene_agent', 0)

        # All assumptions met: both groups normal and equal variances
        if p_shapiro1 > alpha and p_shapiro2 > alpha and p_levene > alpha:
            return NodeName.INDEP_T

        # Assumptions violated: use non-parametric alternatives
        return NodeName.NONPARAMETRIC
    except Exception as e:
        logger.error(f'Error in decide_two_independent: {e}')
        return END
