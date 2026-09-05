"""Anova Agents for statistical analysis using LLM models.

For now only Anova RM test is implemented.
"""

from collections.abc import Callable, Iterable

import numpy as np
import pandas as pd
from pydantic_ai import Agent
from pydantic_ai.models.openai import Model, ModelSettings, OpenAIChatModel

from statmate.agents import (
    AgentResult,
    StatTestDeps,
    build_stat_test_agent,
    run_sync_agent,
)
from statmate.statistical_core import (
    StatTestResult,
    anova_one_way_test,
    anova_rm_test,
    friedman_test,
    kruskal_wallis_test,
    prepare_groups_from_frame,
)


def anova_rm_agent(
    model: Model | str | None,
    test_name: str = 'Anova RM Test',
    test_function: Callable[..., StatTestResult] = anova_rm_test,
    system_prompt: str | None = None,
    model_settings: ModelSettings | None = None,
    retries: int = 3,
    **prompt_kwargs: str,
) -> Agent[StatTestDeps, AgentResult]:
    """Builds an Anova RM agent."""
    return build_stat_test_agent(
        model=model,
        test_name=test_name,
        test_function=test_function,
        system_prompt=system_prompt,
        model_settings=model_settings,
        retries=retries,
        potential_suggestions='Please suggest the best way to perform the test. '
        'If results are not clear, propose different tests.',
        **prompt_kwargs,
    )


def anova_one_way_from_frame(
    data: pd.DataFrame,
    group_column: str,
    value_column: str,
    alpha: float | None = None,
) -> StatTestResult:
    """Wrapper to run one-way ANOVA from a long-format DataFrame."""
    groups, _ = prepare_groups_from_frame(data, group_column, value_column)
    return anova_one_way_test(*groups, alpha=alpha)


def kruskal_wallis_from_frame(
    data: pd.DataFrame,
    group_column: str,
    value_column: str,
    alpha: float | None = None,
    perform_dunn: bool = True,
    p_adjust: str = 'holm',
) -> StatTestResult:
    """Wrapper to run Kruskal-Wallis (and Dunn post-hoc) from a DataFrame."""
    groups, labels = prepare_groups_from_frame(data, group_column, value_column)
    return kruskal_wallis_test(*groups, alpha=alpha, group_labels=labels, perform_dunn=perform_dunn, p_adjust=p_adjust)


def anova_one_way_agent(
    model: Model | str | None,
    test_name: str = 'One-way ANOVA',
    test_function: Callable[..., StatTestResult] = anova_one_way_from_frame,
    system_prompt: str | None = None,
    model_settings: ModelSettings | None = None,
    retries: int = 3,
) -> Agent[StatTestDeps, AgentResult]:
    """Builds a One-way ANOVA agent."""
    return build_stat_test_agent(
        model=model,
        test_name=test_name,
        test_function=test_function,
        system_prompt=system_prompt,
        model_settings=model_settings,
        retries=retries,
        potential_suggestions='Consider non-parametric alternatives if normality or homoscedasticity are violated.',
    )


def kruskal_wallis_agent(
    model: Model | str | None,
    test_name: str = 'Kruskal-Wallis H-test',
    test_function: Callable[..., StatTestResult] = kruskal_wallis_from_frame,
    system_prompt: str | None = None,
    model_settings: ModelSettings | None = None,
    retries: int = 3,
) -> Agent[StatTestDeps, AgentResult]:
    """Builds a Kruskal-Wallis agent."""
    return build_stat_test_agent(
        model=model,
        test_name=test_name,
        test_function=test_function,
        system_prompt=system_prompt,
        model_settings=model_settings,
        retries=retries,
        potential_suggestions="If significant, include pairwise Dunn's post-hoc comparisons with multiplicity control.",
    )


def friedman_agent(
    model: Model | str | None,
    test_name: str = 'Friedman Test',
    test_function: Callable[..., StatTestResult] = friedman_test,
    system_prompt: str | None = None,
    model_settings: ModelSettings | None = None,
    retries: int = 3,
) -> Agent[StatTestDeps, AgentResult]:
    """Builds a Friedman test agent for repeated measures."""
    return build_stat_test_agent(
        model=model,
        test_name=test_name,
        test_function=test_function,
        system_prompt=system_prompt,
        model_settings=model_settings,
        retries=retries,
        potential_suggestions='Use after repeated-measures ANOVA assumptions fail; suggest post-hoc Wilcoxon/Nemenyi.',
    )


def run_anova_rm_agent(
    agent: Agent[StatTestDeps, AgentResult],
    user_prompt: str,
    deps: StatTestDeps,
) -> AgentResult:
    """Run the Anova RM agent."""
    # Convert pandas Series to numpy arrays
    if isinstance(deps.data, Iterable):
        deps.data = np.array(deps.data)
    if deps.data_secondary is not None and isinstance(deps.data_secondary, Iterable):
        deps.data_secondary = np.array(deps.data_secondary)
    return run_sync_agent(agent, user_prompt, deps)


if __name__ == '__main__':
    # Example usage
    model = OpenAIChatModel('gpt-4o')
    agent = anova_rm_agent(model=model)
    user_prompt = 'Perform a repeated measures ANOVA test.'
    deps = StatTestDeps(
        data=np.random.rand(10, 3),  # Example data
        data_secondary=None,
        test_params={'dependent_variable': 'value', 'subject': 'subject', 'within': ['condition']},
    )
    result = run_anova_rm_agent(agent, user_prompt, deps)
    print(result)
