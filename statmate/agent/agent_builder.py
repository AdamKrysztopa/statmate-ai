"""Agent Builder for Statmate."""

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field
from pydantic_ai import Agent, RunContext
from pydantic_ai.models import Model

from statmate.statistical_core.base import StatTestResult


def build_generic_system_prompt(**kwargs: str) -> str:
    """Build a generic system prompt for a statistical test agent with optional additional instructions.

    Args:
        **kwargs: Additional context to be included in the prompt.

    Keyword Args:
            extra_steps: Detailed extra instructions to be appended.
            test_name: (Optional) The name of the test for inclusion in the prompt.
            [Any other key-value pairs representing additional context]

    Returns:
        A formatted system prompt string that includes both the generic instructions and any extra steps provided.
    """
    base_prompt = (
        'You are a statistical test agent responsible for analyzing datasets using rigorous, '
        'validated statistical methods. '
        'Your tasks are as follows:\n'
        '1. Validate the received dataset and ensure that all required parameters are provided; apply appropriate '
        'default values when parameters are missing.\n'
        '2. Accurately perform the designated statistical test and verify underlying assumptions and data quality.\n'
        '3. Return a structured JSON response containing:\n'
        '    - The test name and computed statistics (e.g., test statistic, p-value).\n'
        '    - The null hypothesis and the alternative hypothesis.\n'
        '    - A clear explanation of the results, including an interpretation regarding the null hypothesis.\n'
        '    - Additional test-specific details (e.g., significance level, sample size, etc.).\n'
        '4. Ensure that your response is concise, factual, and formal in tone.'
    )

    if kwargs:
        extra_instructions = '\n'.join(f'{key}: {value}' for key, value in kwargs.items())
        base_prompt += '\n\nAdditional Instructions:\n' + extra_instructions

    return base_prompt


class StatTestDeps(BaseModel):
    """Dependencies for the statistical test agent."""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )

    data: np.ndarray | pd.Series = Field(
        description='Input data to test. If test requires two datasets, provide secondary set: data_secondary.',
    )
    data_secondary: np.ndarray | pd.Series | None = Field(
        description='Secondary dataset for two-sample tests.',
        default=None,
    )
    test_params: dict[str, Any] | None = Field(
        description='Parameters for the statistical test.',
        default=None,
    )


class AgentResult(BaseModel):
    """Result of the statistical test."""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )
    statistical_test_result: StatTestResult = Field(
        description='Statistical test result.',
    )
    result: str = Field(
        description='Result of the statistical test, explained what does it mean '
        'if null hypothesis is rejected or not.',
    )
    comments: str = Field(
        description='Comments about the statistical test, including number of samples impact, '
        'test specyfic details, etc.',
    )

    def __str__(self) -> str:
        """Return a string representation of the agent result.

        Returns:
            str: String representation of the agent result.
        """
        return (
            f'### Test Name: {self.statistical_test_result.test_name} ###\n\n'
            f'Statistic: {self.statistical_test_result.statistics},\n\np-value: {self.statistical_test_result.p_value}\n\n'
            f'Result: {self.result}\n\nComment: {self.comments}\n\n'
        )


def build_stat_test_agent(
    model: Model | str | None,
    test_name: str,
    test_function: Callable[..., StatTestResult],
    system_prompt: str | None = None,
    **prompt_kwargs: str,
) -> Agent[StatTestDeps, AgentResult]:
    """Builds a statistical test agent with an extensible system prompt.

    Args:
        model (str | Model): Model name or instance.
        test_name (str): Name of the statistical test.
        test_function (Callable[..., StatTestResult]): Function to perform the test.
        system_prompt (str, optional): Optional base system prompt. If not provided, a generic prompt will be generated.
        **prompt_kwargs: Additional keyword arguments to extend the system prompt.

    Returns:
        Agent: The built agent.
    """
    if system_prompt is None:
        system_prompt = build_generic_system_prompt(**prompt_kwargs)

    agent = Agent(
        model=model,
        deps_type=StatTestDeps,
        result_type=AgentResult,
        name=f'Statistical Test Agent: {test_name}',
        system_prompt=system_prompt,
        result_tool_description=f'Performing {test_name} statistical test.',
    )

    @agent.tool
    def run_test(ctx: RunContext[StatTestDeps]) -> StatTestResult:
        """Run the statistical test."""
        if ctx.deps.test_params is None:
            ctx.deps.test_params = {}
        if isinstance(ctx.deps.data, pd.Series):
            ctx.deps.data = ctx.deps.data.to_numpy()

        if ctx.deps.data_secondary is not None and isinstance(ctx.deps.data_secondary, pd.Series):
            ctx.deps.data_secondary = ctx.deps.data_secondary.to_numpy()
        if ctx.deps.data_secondary is not None:
            return test_function(ctx.deps.data, ctx.deps.data_secondary, **ctx.deps.test_params)
        return test_function(ctx.deps.data, **ctx.deps.test_params)

    return agent
