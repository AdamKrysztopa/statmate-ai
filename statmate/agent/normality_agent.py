from collections.abc import Callable

import numpy as np
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel

from statmate.agent import StatTestDeps, build_stat_test_agent
from statmate.statistical_core import StatTestResult, shapiro_wilk_test


def shapiro_wilk_agent(
    model: OpenAIModel,
    test_name: str = 'Shapiro-Wilk Test',
    test_function: Callable[..., StatTestResult] = shapiro_wilk_test,
) -> Agent[StatTestDeps, StatTestResult]:
    """Builds a Shapiro-Wilk test agent."""
    system_prompt = (
        'You are a statistical test agent. '
        'Your task is to perform a statistical test on the provided data. '
        'You will receive a dataset and parameters for the test. '
        'Return the results of the test in a structured format.'
    )
    return build_stat_test_agent(
        model=model,
        test_name=test_name,
        test_function=test_function,
        system_prompt=system_prompt,
    )


if __name__ == '__main__':
    # Example usage
    model = OpenAIModel('gpt-4o')

    agent = shapiro_wilk_agent(model=model)
    data_1 = np.random.normal(0, 1, 100)
    data_2 = np.random.uniform(0, 1, 100)
    test_params = {'alpha': 0.05}
    result = agent.run_sync(
        'Please let me know the result of the test.',  # type: ignore
        StatTestDeps(data=data_1, test_params=test_params),
    )
    print(result)
