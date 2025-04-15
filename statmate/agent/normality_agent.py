from collections.abc import Callable

import numpy as np
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel

from statmate.agent import AgentResult, StatTestDeps, build_stat_test_agent
from statmate.statistical_core import StatTestResult, normality_of_difference, shapiro_wilk_test


def shapiro_wilk_agent(
    model: OpenAIModel,
    test_name: str = 'Shapiro-Wilk Test',
    test_function: Callable[..., StatTestResult] = shapiro_wilk_test,
) -> Agent[StatTestDeps, AgentResult]:
    """Builds a Shapiro-Wilk test agent."""
    return build_stat_test_agent(
        model=model,
        test_name=test_name,
        test_function=test_function,
        potential_suggertions='Please suggest the best way to perform the test, '
        'if results are not clear, propose different tests.',
    )


def normality_of_difference_agent(
    model: OpenAIModel,
    test_name: str = 'Normality of Difference Test',
    test_function: Callable[..., StatTestResult] = normality_of_difference,
) -> Agent[StatTestDeps, AgentResult]:
    """Builds a Normality of Difference test agent."""
    return build_stat_test_agent(
        model=model,
        test_name=test_name,
        test_function=test_function,
        difference_with_standard_normality='Tests if difference between the two samples is drawn from a normal distribution.',
        extra_comments='The statistical test first calculates the difference between the two samples, '
        'and then performs the Shapiro-Wilk test on the resulting differences.',
    )


if __name__ == '__main__':
    # Example usage
    model = OpenAIModel('gpt-4o')

    agent = shapiro_wilk_agent(model=model)
    data_1 = np.random.normal(0, 1, 100)
    data_2 = np.random.uniform(-3, 3, 100)
    data_3 = np.random.normal(1, 1, 100)
    test_params = {'alpha': 0.05}
    result = agent.run_sync(
        user_prompt='Please let me know the result of the test.',
        deps=StatTestDeps(data=data_2, test_params=test_params),
    )
    print(result.data)

    agent_2 = normality_of_difference_agent(model=model)
    result_2 = agent_2.run_sync(
        user_prompt='Please let me know the result of the test.',
        deps=StatTestDeps(data=data_1, data_secondary=data_3, test_params=test_params),
    )
    print(result_2.data)
