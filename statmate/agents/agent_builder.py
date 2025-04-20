"""Agent Builder for Statmate."""

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field
from pydantic_ai import Agent, RunContext
from pydantic_ai.models import Model, ModelSettings

from statmate.statistical_core.base import StatTestResult


def build_generic_system_prompt(**kwargs: str) -> str:
    """Build a detailed, structured system prompt for statistical test agents.

    The template uses Jinja-style placeholders for runtime substitution.
    Supported placeholders:
      - test_name
      - test_specifics.alpha
      - test_specifics.assumption1
      - test_specifics.alternative_test
      - suggestions
    Additional context can be injected via kwargs.

    """
    template = """
You are a statistical test agent.

Validation:
  - Ensure data shape and parameters are correct and complete.
  - Check for missing values and consistent data types.

Execution:
  - Perform {{ test_name }} using significance level = {{ test_specifics.alpha }}.

Assumptions & Diagnostics:
  - Verify assumption: {{ test_specifics.assumption1 }}.
  - If violated, suggest: {{ test_specifics.alternative_test }}.

Data Exploration:
  - Summarize distribution metrics (mean, median, variance, skewness).
  - Detect outliers and missing values; document handling decisions.

Precision & Effect Size:
  - Compute effect size (e.g., Cohen's d) and confidence intervals for estimate precision.

Screening & Recommendations:
  - Recommend visual diagnostics (histogram, boxplot, Q-Q plot).
  - Flag additional tests or data transformations if needed.
  - In the case of low number of samples, suggest using more data or different test whchich is more robust.

Output:
  - Return JSON containing:
      * test_name, statistic, p_value
      * null_hypothesis, alternative_hypothesis
      * interpretation and decision on H0
      * test_specifics (alpha, sample sizes, effect size, confidence intervals)

Suggestions:
  - {{ suggestions }}

Best Practices:
  - Log each step and parameter for reproducibility.
  - Seed random generators when applicable.
  - Version-control code and track parameter settings.
"""

    # Append any extra context provided
    if kwargs:
        extras = '\n'.join(f'- {k}: {v}' for k, v in kwargs.items())
        template += f'\nAdditional Context:\n{extras}\n'

    return template


class StatTestDeps(BaseModel):
    """Dependencies for the statistical test agent."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    data: np.ndarray | pd.Series = Field(
        description='Primary dataset (numpy array or pandas Series).',
    )
    data_secondary: np.ndarray | pd.Series | None = Field(
        description='Secondary dataset for two-sample tests.',
        default=None,
    )
    test_params: dict[str, Any] | None = Field(
        description='Parameters such as alpha, alternative, equal_var.',
        default=None,
    )


class AgentResult(BaseModel):
    """Result of the statistical test."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    statistical_test_result: StatTestResult = Field(description='Core test results object.')
    result: str = Field(description='Interpretation of null hypothesis decision.')
    comments: str = Field(description='Additional comments and recommendations.')

    def __str__(self) -> str:
        return (
            f'### Test Name: {self.statistical_test_result.test_name} ###\n'
            f'Statistic: {self.statistical_test_result.statistics}, p-value: {self.statistical_test_result.p_value}\n'
            f'Result: {self.result}\n'
            f'Comments: {self.comments}\n'
        )


def build_stat_test_agent(
    model: Model | str | None,
    test_name: str,
    test_function: Callable[..., StatTestResult],
    system_prompt: str | None = None,
    model_settings: ModelSettings | None = None,
    retries: int = 3,
    **prompt_kwargs: str,
) -> Agent[StatTestDeps, AgentResult]:
    """Constructs a statistical test agent with an extensible, precise prompt."""
    # Generate prompt if none provided
    if system_prompt is None:
        system_prompt = build_generic_system_prompt(**prompt_kwargs)

    agent = Agent(
        model=model,
        model_settings=model_settings,
        deps_type=StatTestDeps,
        result_type=AgentResult,
        name=f'Statistical Test Agent: {test_name}',
        system_prompt=system_prompt,
        result_tool_description=f'Performing {test_name} statistical test.',
        retries=retries,
    )

    @agent.tool
    async def run_test(ctx: RunContext[StatTestDeps]) -> StatTestResult:
        # Initialize parameters
        if ctx.deps.test_params is None:
            ctx.deps.test_params = {}
        # Convert pandas Series to numpy arrays
        if isinstance(ctx.deps.data, pd.Series):
            ctx.deps.data = ctx.deps.data.to_numpy()
        if ctx.deps.data_secondary is not None and isinstance(ctx.deps.data_secondary, pd.Series):
            ctx.deps.data_secondary = ctx.deps.data_secondary.to_numpy()
        # Execute two-sample or one-sample test
        if ctx.deps.data_secondary is not None:
            return test_function(
                ctx.deps.data,
                ctx.deps.data_secondary,
                **ctx.deps.test_params,
            )
        return test_function(ctx.deps.data, **ctx.deps.test_params)

    return agent


async def run_async_agent(
    agent: Agent[StatTestDeps, AgentResult],
    user_prompt: str,
    deps: StatTestDeps,
) -> AgentResult:
    """Run the agent asynchronously.

    Args:
        agent: The agent to run.
        user_prompt: The user prompt to provide.
        deps: The dependencies for the agent.

    Returns:
        The result of the agent run.
    """
    result = await agent.run(user_prompt=user_prompt, deps=deps)

    return result.data


def run_sync_agent(
    agent: Agent[StatTestDeps, AgentResult],
    user_prompt: str,
    deps: StatTestDeps,
) -> AgentResult:
    """Run the agent synchronously.

    Args:
        agent: The agent to run.
        user_prompt: The user prompt to provide.
        deps: The dependencies for the agent.

    Returns:
        The result of the agent run.
    """
    result = agent.run_sync(user_prompt=user_prompt, deps=deps)

    return result.data
