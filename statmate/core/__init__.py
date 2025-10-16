"""Core functionality for StatMate AI.

This module contains foundational components including:
- Configuration management
- Custom exceptions
- Logging setup
- Abstract base classes and interfaces
- Input validation utilities
"""

from statmate.core.base_interfaces import (
    AgentDependencies,
    AgentResult,
    DataTransformer,
    DataValidator,
    StatisticalTest,
    StatMateAgent,
    WorkflowNode,
)
from statmate.core.config import (
    Config,
    DataType,
    LoggingConfig,
    ModelConfig,
    NodeName,
    StatisticalTestConfig,
    TransformationType,
    WorkflowConfig,
    default_config,
)
from statmate.core.exceptions import (
    AgentError,
    AgentToolError,
    ConfigurationError,
    DataValidationError,
    InsufficientDataError,
    InvalidDataShapeError,
    InvalidDataTypeError,
    InvalidTransformationError,
    MissingDataError,
    ModelError,
    ModelInferenceError,
    ModelInitializationError,
    NodeExecutionError,
    StatisticalTestError,
    StatMateError,
    TestAssumptionViolationError,
    TransformationError,
    WorkflowError,
)
from statmate.core.logging_config import get_logger, setup_logging
from statmate.core.model_config import (
    SUPPORTED_MODELS,
    ModelCapability,
    ModelInfo,
    ModelProvider,
    ModelProviderConfig,
    MultiModelConfig,
    create_default_multi_model_config,
)
from statmate.core.model_provider import (
    ModelProviderError,
    ModelProviderSystem,
    create_model_provider_system,
)
from statmate.core.validation import (
    validate_alpha,
    validate_array_not_empty,
    validate_categorical_data,
    validate_contingency_table,
    validate_dataframe_columns,
    validate_independent_samples,
    validate_minimum_sample_size,
    validate_no_missing_values,
    validate_numeric_data,
    validate_paired_data,
    validate_same_length,
    validate_test_parameters,
)

__all__ = [
    # Config
    'Config',
    'ModelConfig',
    'StatisticalTestConfig',
    'WorkflowConfig',
    'LoggingConfig',
    'default_config',
    'NodeName',
    'DataType',
    'TransformationType',
    # Model Configuration
    'MultiModelConfig',
    'ModelProvider',
    'ModelProviderConfig',
    'ModelInfo',
    'ModelCapability',
    'SUPPORTED_MODELS',
    'create_default_multi_model_config',
    'ModelProviderSystem',
    'ModelProviderError',
    'create_model_provider_system',
    # Exceptions
    'StatMateError',
    'DataValidationError',
    'InsufficientDataError',
    'InvalidDataShapeError',
    'MissingDataError',
    'InvalidDataTypeError',
    'ConfigurationError',
    'ModelError',
    'ModelInitializationError',
    'ModelInferenceError',
    'WorkflowError',
    'NodeExecutionError',
    'StatisticalTestError',
    'TestAssumptionViolationError',
    'AgentError',
    'AgentToolError',
    'TransformationError',
    'InvalidTransformationError',
    # Logging
    'setup_logging',
    'get_logger',
    # Interfaces
    'StatisticalTest',
    'DataTransformer',
    'AgentDependencies',
    'AgentResult',
    'StatMateAgent',
    'WorkflowNode',
    'DataValidator',
    # Validation
    'validate_array_not_empty',
    'validate_minimum_sample_size',
    'validate_no_missing_values',
    'validate_numeric_data',
    'validate_same_length',
    'validate_dataframe_columns',
    'validate_categorical_data',
    'validate_contingency_table',
    'validate_paired_data',
    'validate_independent_samples',
    'validate_alpha',
    'validate_test_parameters',
]
