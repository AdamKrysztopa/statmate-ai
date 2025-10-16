"""Abstract base classes and interfaces for StatMate AI.

This module defines the contracts that various components must implement,
ensuring consistent behavior across the application.
"""

from abc import ABC, abstractmethod
from typing import Any, Protocol

import numpy as np
import pandas as pd
from pydantic import BaseModel
from pydantic_ai import Agent

from statmate.statistical_core.base import StatTestResult


class StatisticalTest(Protocol):
    """Protocol for statistical test functions."""

    def __call__(
        self,
        data: np.ndarray,
        data_secondary: np.ndarray | None = None,
        alpha: float = 0.05,
        **kwargs: Any,
    ) -> StatTestResult:
        """Execute the statistical test.

        Args:
            data: Primary data array.
            data_secondary: Optional secondary data for two-sample tests.
            alpha: Significance level.
            **kwargs: Additional test-specific parameters.

        Returns:
            StatTestResult containing test outcomes.
        """
        ...


class DataTransformer(ABC):
    """Abstract base class for data transformations."""

    @abstractmethod
    def transform(self, data: pd.DataFrame, **kwargs: Any) -> pd.DataFrame | tuple[pd.Series, pd.Series]:
        """Transform the input data.

        Args:
            data: Input DataFrame to transform.
            **kwargs: Transformation-specific parameters.

        Returns:
            Transformed data.
        """

    @abstractmethod
    def validate_parameters(self, **kwargs: Any) -> dict[str, Any]:
        """Validate transformation parameters.

        Args:
            **kwargs: Parameters to validate.

        Returns:
            Validated parameters.

        Raises:
            DataValidationError: If parameters are invalid.
        """


class AgentDependencies(BaseModel):
    """Base class for agent dependencies."""


class AgentResult(BaseModel):
    """Base class for agent results."""


class StatMateAgent(ABC):
    """Abstract base class for StatMate agents."""

    def __init__(self, name: str):
        """Initialize the agent.

        Args:
            name: Name of the agent.
        """
        self.name = name

    @abstractmethod
    def build(self, **kwargs: Any) -> Agent:
        """Build the PydanticAI agent.

        Args:
            **kwargs: Agent-specific configuration.

        Returns:
            Configured Agent instance.
        """

    @abstractmethod
    def validate_dependencies(self, deps: Any) -> None:
        """Validate agent dependencies.

        Args:
            deps: Dependencies to validate.

        Raises:
            DataValidationError: If dependencies are invalid.
        """


class WorkflowNode(ABC):
    """Abstract base class for workflow nodes."""

    def __init__(self, name: str):
        """Initialize the workflow node.

        Args:
            name: Name of the node.
        """
        self.name = name

    @abstractmethod
    def execute(self, state: dict[str, Any]) -> dict[str, Any]:
        """Execute the node logic.

        Args:
            state: Current workflow state.

        Returns:
            Updated workflow state.

        Raises:
            NodeExecutionError: If node execution fails.
        """

    @abstractmethod
    def validate_state(self, state: dict[str, Any]) -> None:
        """Validate the workflow state before execution.

        Args:
            state: State to validate.

        Raises:
            DataValidationError: If state is invalid.
        """


class DataValidator(ABC):
    """Abstract base class for data validators."""

    @abstractmethod
    def validate(self, data: Any) -> None:
        """Validate the data.

        Args:
            data: Data to validate.

        Raises:
            DataValidationError: If validation fails.
        """

    @abstractmethod
    def get_validation_errors(self, data: Any) -> list[str]:
        """Get list of validation errors without raising.

        Args:
            data: Data to validate.

        Returns:
            List of validation error messages (empty if valid).
        """
