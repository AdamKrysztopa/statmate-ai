"""Anova Agents for statistical analysis using LLM models.

For now only Anova RM test is implemented.
"""

from collections.abc import Callable, Iterable

import numpy as np
from pydantic_ai import Agent
from pydantic_ai.models.openai import Model, ModelSettings, OpenAIModel

from statmate.agents import (
    AgentResult,
    StatTestDeps,
    build_stat_test_agent,
    run_sync_agent,
)
from statmate.statistical_core import StatTestResult, anova_rm_test


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
        potential_suggertions='Please suggest the best way to perform the test, '
        'if results are not clear, propose different tests.',
        **prompt_kwargs,
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
    model = OpenAIModel('gpt-4o')
    agent = anova_rm_agent(model=model)
    user_prompt = 'Perform a repeated measures ANOVA test.'
    deps = StatTestDeps(
        data=np.random.rand(10, 3),  # Example data
        data_secondary=None,
        test_params={'dependent_variable': 'value', 'subject': 'subject', 'within': ['condition']},
    )
    result = run_anova_rm_agent(agent, user_prompt, deps)
    print(result)
