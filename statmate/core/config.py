"""Configuration module for StatMate AI.

This module centralizes all configuration settings, constants, and defaults
used throughout the StatMate application.
"""

from dataclasses import dataclass, field
from typing import Literal

from statmate.core.model_config import (
    MultiModelConfig,
    create_default_multi_model_config,
)


@dataclass
class StatisticalTestConfig:
    """Configuration for statistical tests."""

    default_alpha: float = 0.05
    """Default significance level for statistical tests."""

    normality_threshold: float = 0.05
    """P-value threshold for normality tests."""

    variance_threshold: float = 0.05
    """P-value threshold for variance equality tests."""

    skewness_threshold: float = 2.0
    """Absolute skewness above this value is flagged for assumption diagnostics."""

    kurtosis_threshold: float = 7.0
    """Kurtosis above this value (excess) is flagged as heavy-tailed."""

    variance_ratio_threshold: float = 4.0
    """Ratio of max/min variance above this value indicates heteroscedasticity."""

    sparsity_threshold: float = 0.2
    """Fraction of zero/empty values above this value is considered sparse."""

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

    model: MultiModelConfig = field(default_factory=create_default_multi_model_config)
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
        # Note: This simple conversion might not handle nested structures
        # in MultiModelConfig correctly if loaded from a plain dict (e.g., JSON).
        # It assumes the structure is already correct or that default values are sufficient.
        return cls(
            model=MultiModelConfig(**config_dict.get('model', {})),
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
    ANOVA_ASSUMPTIONS = 'ANOVA assumptions'
    ANOVA_ONE_WAY = 'One-way ANOVA'
    KRUSKAL_WALLIS = 'Kruskal-Wallis H-test'
    FRIEDMAN = 'Friedman test'
    MCNEMAR = 'McNemar test'
    COX_REGRESSION = 'Cox regression (placeholder)'
    CHOICE = 'Choice Node'
    INTENT = 'Intent Agent'
    METHODOLOGY_AUDITOR = 'Methodology Auditor'
    REVIEWER = 'Reviewer Agent'
    DESIGN_VERIFICATION = 'Design Verification'
    DESIGN_RECONCILIATION = 'Design Reconciliation'
    DESCRIPTIVE_SUMMARY = 'Descriptive Summary'
    USER_INTERVENTION = 'User Intervention Needed'


class DataType:
    """Data type constants."""

    CONTINUOUS = 'CONTINUOUS'
    CATEGORICAL = 'CATEGORICAL'
    SURVIVAL = 'SURVIVAL'


class TransformationType:
    """Data transformation types."""

    TRANSFORM_INDEPENDENT = 'transform_independent'
    TRANSFORM_CATEGORICAL = 'transform_categorical'
    NONE = 'None'
