"""Configuration module for StatMate AI.

This module centralizes all configuration settings, constants, and defaults
used throughout the StatMate application.
"""

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class ModelConfig:
    """Configuration for AI model settings."""

    model_name: str = 'gpt-4o'
    temperature: float = 0.0
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    max_tokens: int | None = None
    retries: int = 3


@dataclass
class StatisticalTestConfig:
    """Configuration for statistical tests."""

    default_alpha: float = 0.05
    """Default significance level for statistical tests."""

    normality_threshold: float = 0.05
    """P-value threshold for normality tests."""

    variance_threshold: float = 0.05
    """P-value threshold for variance equality tests."""

    categorical_sample_size_threshold: int = 10
    """Minimum sample size for Chi-square test (vs Fisher's exact)."""

    meta_analysis_rejection_threshold: float = 0.5
    """Weighted rejection score threshold for meta-analysis."""


@dataclass
class WorkflowConfig:
    """Configuration for workflow execution."""

    enable_parallel_execution: bool = False
    """Whether to run tests in parallel where possible."""

    cache_results: bool = False
    """Whether to cache test results."""

    max_workflow_retries: int = 2
    """Maximum number of workflow retries on failure."""


@dataclass
class LoggingConfig:
    """Configuration for logging."""

    level: Literal['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'] = 'INFO'
    """Default logging level."""

    format: str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    """Log message format."""

    suppress_httpx: bool = True
    """Whether to suppress verbose HTTPX logging."""

    log_to_file: bool = False
    """Whether to log to file."""

    log_file_path: str = 'statmate.log'
    """Path to log file if log_to_file is True."""


@dataclass
class Config:
    """Main configuration class for StatMate AI."""

    model: ModelConfig = field(default_factory=ModelConfig)
    statistical: StatisticalTestConfig = field(default_factory=StatisticalTestConfig)
    workflow: WorkflowConfig = field(default_factory=WorkflowConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    @classmethod
    def from_dict(cls, config_dict: dict) -> 'Config':
        """Create Config from dictionary.

        Args:
            config_dict: Dictionary containing configuration values.

        Returns:
            Config instance.
        """
        return cls(
            model=ModelConfig(**config_dict.get('model', {})),
            statistical=StatisticalTestConfig(**config_dict.get('statistical', {})),
            workflow=WorkflowConfig(**config_dict.get('workflow', {})),
            logging=LoggingConfig(**config_dict.get('logging', {})),
        )


# Default global configuration instance
default_config = Config()


# Constants
class NodeName:
    """Node names for workflow graph."""

    INITIALIZATION = 'Initialization Agent'
    ASSESS_STUDY_DESIGN = 'Assess Study Design'
    TWO_INDEPENDENT_GROUPS = 'Two Independent Groups?'
    NONPARAMETRIC = 'Nonparametric Tests'
    SUMMARY = 'Summary'
    SHAPIRO = 'Shapiro-Wilk Test'
    LEVENE = "Levene's Test"
    PAIRED_T = 'Paired t-test'
    WILCOXON = 'Wilcoxon Signed-Rank test'
    INDEP_T = 'Independent t-test'
    WELCH = "Welch's t-test"
    MANN = 'Mann-Whitney U'
    CHI2 = 'Chi-square test'
    FISHER = 'Fisher exact test'
    NORMALITY_OF_DIFFERENCE = 'Parametric assumptions hold?'
    ANOVA_RM = 'ANOVA repeated measures'


class DataType:
    """Data type constants."""

    CONTINUOUS = 'CONTINUOUS'
    CATEGORICAL = 'CATEGORICAL'


class TransformationType:
    """Data transformation types."""

    TRANSFORM_INDEPENDENT = 'transform_independent'
    TRANSFORM_CATEGORICAL = 'transform_categorical'
    NONE = 'None'
