"""Workflow state management using Pydantic models.

This module defines the state structure for the statistical test workflow,
replacing the TypedDict approach with proper Pydantic models for better
validation and type safety.
"""

from typing import Literal

import pandas as pd
from langchain_core.messages import AIMessage
from pydantic import BaseModel, Field


class WorkflowState(BaseModel):
    """State for the statistical test workflow.

    This model uses Pydantic for validation instead of TypedDict,
    providing better type safety and automatic validation.
    """

    model_config = {'arbitrary_types_allowed': True}

    # Data fields
    df: pd.Series | pd.DataFrame = Field(description='Primary dataset for analysis')
    secondary_df: pd.Series | None = Field(default=None, description='Secondary dataset for two-sample tests')

    # Metadata fields
    target_columns: list[str] = Field(default_factory=list, description='Columns to analyze')
    paired: bool | None = Field(default=None, description='Whether data represents paired measurements')
    data_type: Literal['CONTINUOUS', 'CATEGORICAL'] | None = Field(
        default=None, description='Type of data being analyzed'
    )
    do_association: bool = Field(default=False, description='Whether to perform association tests')
    number_of_samples: int = Field(default=0, description='Total number of samples in the dataset')

    # Results fields
    results: list[AIMessage] = Field(default_factory=list, description='List of test results as AIMessages')
    probabilities: dict[str, float] = Field(default_factory=dict, description='P-values from executed tests')

    # Model configuration
    model_name: str | None = Field(default=None, description='AI model to use for analysis')
    provider: str | None = Field(default=None, description='Model provider (openai, anthropic, etc.)')

    def add_result(self, message: AIMessage) -> None:
        """Add a result message to the results list.

        Args:
            message: AIMessage containing test results.
        """
        self.results.append(message)

    def add_probability(self, test_name: str, p_value: float) -> None:
        """Add a test p-value to the probabilities dict.

        Args:
            test_name: Name of the test.
            p_value: P-value from the test.
        """
        self.probabilities[test_name] = p_value

    def get_probability(self, test_name: str, default: float = 0.0) -> float:
        """Get a test p-value from the probabilities dict.

        Args:
            test_name: Name of the test.
            default: Default value if test_name not found.

        Returns:
            P-value for the test, or default if not found.
        """
        return self.probabilities.get(test_name, default)


def create_initial_state(
    df: pd.DataFrame | pd.Series,
    target_columns: list[str] | None = None,
    paired: bool = False,
    do_association: bool = False,
) -> WorkflowState:
    """Create an initial workflow state.

    Args:
        df: Primary dataset.
        target_columns: Columns to analyze. If None, uses all columns.
        paired: Whether data represents paired measurements.
        do_association: Whether to perform association tests.

    Returns:
        Initialized WorkflowState.
    """
    return WorkflowState(
        df=df,
        secondary_df=None,
        target_columns=target_columns or [],
        paired=paired,
        data_type=None,
        do_association=do_association,
        number_of_samples=0,
        results=[],
        probabilities={},
    )
