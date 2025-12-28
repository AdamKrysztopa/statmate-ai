"""Custom exceptions for StatMate AI.

This module defines specific exception types for better error handling
and debugging throughout the application.
"""


class StatMateError(Exception):
    """Base exception for all StatMate errors."""


class DataValidationError(StatMateError):
    """Raised when input data validation fails."""


class InsufficientDataError(DataValidationError):
    """Raised when there is insufficient data for a statistical test."""


class InvalidDataShapeError(DataValidationError):
    """Raised when data has an invalid shape for the requested operation."""


class MissingDataError(DataValidationError):
    """Raised when required data is missing."""


class InvalidDataTypeError(DataValidationError):
    """Raised when data type is invalid for the requested operation."""


class ConfigurationError(StatMateError):
    """Raised when there is a configuration error."""


class ModelError(StatMateError):
    """Raised when there is an error with the AI model."""


class ModelInitializationError(ModelError):
    """Raised when model initialization fails."""


class ModelInferenceError(ModelError):
    """Raised when model inference fails."""


class WorkflowError(StatMateError):
    """Raised when there is an error in workflow execution."""


class NodeExecutionError(WorkflowError):
    """Raised when a workflow node fails to execute."""

    def __init__(self, node_name: str, original_error: Exception):
        """Initialize NodeExecutionError.

        Args:
            node_name: Name of the failed node.
            original_error: The original exception that caused the failure.
        """
        self.node_name = node_name
        self.original_error = original_error
        super().__init__(f"Node '{node_name}' failed: {str(original_error)}")


class StatisticalTestError(StatMateError):
    """Raised when a statistical test fails."""


class TestAssumptionViolationError(StatisticalTestError):
    """Raised when statistical test assumptions are violated."""

class StatisticalAssumptionError(StatisticalTestError):
    """Raised when required statistical assumptions are not met for a function."""


class RoutingError(WorkflowError):
    """Raised when routing logic encounters an invalid path."""


class AgentError(StatMateError):
    """Raised when an agent encounters an error."""


class AgentToolError(AgentError):
    """Raised when an agent tool fails."""

    def __init__(self, tool_name: str, original_error: Exception):
        """Initialize AgentToolError.

        Args:
            tool_name: Name of the failed tool.
            original_error: The original exception that caused the failure.
        """
        self.tool_name = tool_name
        self.original_error = original_error
        super().__init__(f"Tool '{tool_name}' failed: {str(original_error)}")


class TransformationError(StatMateError):
    """Raised when data transformation fails."""


class InvalidTransformationError(TransformationError):
    """Raised when an invalid transformation is requested."""
