from collections.abc import Callable

import numpy as np
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel

from statmate.agent import AgentResult, StatTestDeps, build_stat_test_agent
from statmate.statistical_core import StatTestResult, ttest_rel_test, wilcoxon_test


def wilcoxon_agent(
    model: OpenAIModel,
    test_name: str = 'Wilcoxon Test',
    test_function: Callable[..., StatTestResult] = wilcoxon_test,
) -> Agent[StatTestDeps, AgentResult]:
    """Builds a Wilcoxon test agent."""
    return build_stat_test_agent(
        model=model,
        test_name=test_name,
        test_function=test_function,
        potential_suggertions='Please suggest the best way to perform the test, '
        'if results are not clear, propose different tests.',
    )


def ttest_rel_agent(
    model: OpenAIModel,
    test_name: str = 'T-Test',
    test_function: Callable[..., StatTestResult] = ttest_rel_test,
) -> Agent[StatTestDeps, AgentResult]:
    """Builds a T-Test agent."""
    return build_stat_test_agent(
        model=model,
        test_name=test_name,
        test_function=test_function,
        potential_suggertions='Please suggest the best way to perform the test, '
        'if results are not clear, propose different tests.',
    )


if __name__ == '__main__':
    # Example usage
    model = OpenAIModel('gpt-4o')

    agent = wilcoxon_agent(model=model)
    data_1 = np.random.normal(0, 1, 100)
    data_2 = np.random.uniform(-1, 1, 100)
    test_params = {'alpha': 0.05}
    result = agent.run_sync(
        user_prompt='Please let me know the result of the test.',  # type: ignore
        deps=StatTestDeps(data=data_1, data_secondary=data_2, test_params=test_params),
    )
    print(result.data)

    agent_2 = ttest_rel_agent(model=model)
    result_2 = agent_2.run_sync(
        user_prompt='Please let me know the result of the test.',  # type: ignore
        deps=StatTestDeps(data=data_1, data_secondary=data_2, test_params=test_params),
    )
    print(result_2.data)
