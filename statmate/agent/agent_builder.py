"""Agent Builder for Statmate."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from pydantic_ai import Agent, RunContext
from pydantic_ai.models import Model

from statmate.statistical_core.base import StatTestResult


@dataclass
class StatTestDeps:
    """Dependencies for statistical test."""

    data: np.ndarray | pd.Series
    test_params: dict[str, Any] | None = None


def build_stat_test_agent(
    model: Model | str | None,
    test_name: str,
    test_function: Callable[..., StatTestResult],
    system_prompt: str,
) -> Agent[StatTestDeps, StatTestResult]:
    """Builds a statistical test agent.

    Args:
        model (str): Model name.
        test_name (str): Name of the statistical test.
        test_function (Callable[..., StatTestResult]): Function to perform the test.
        system_prompt (str): System prompt for the agent.

    Returns:
        Agent: The built agent.
    """
    agent = Agent(
        model=model,
        deps_type=StatTestDeps,
        result_type=StatTestResult,
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
        return test_function(ctx.deps.data, **ctx.deps.test_params)

    return agent
