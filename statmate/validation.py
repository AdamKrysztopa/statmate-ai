"""Data validation utilities for StatMate AI.

This module provides functions to validate input data for statistical tests
and workflows.
"""

from typing import Any

import numpy as np
import pandas as pd

from statmate.exceptions import (
    DataValidationError,
    InsufficientDataError,
    InvalidDataShapeError,
    InvalidDataTypeError,
    MissingDataError,
)


def validate_array_not_empty(data: np.ndarray, name: str = 'data') -> None:
    """Validate that an array is not empty.

    Args:
        data: Array to validate.
        name: Name of the data for error messages.

    Raises:
        InsufficientDataError: If array is empty.
    """
    if data.size == 0:
        raise InsufficientDataError(f'{name} is empty')


def validate_minimum_sample_size(
    data: np.ndarray | pd.Series | pd.DataFrame, min_size: int, name: str = 'data'
) -> None:
    """Validate that data has minimum sample size.

    Args:
        data: Data to validate.
        min_size: Minimum required sample size.
        name: Name of the data for error messages.

    Raises:
        InsufficientDataError: If data has fewer samples than required.
    """
    if isinstance(data, pd.DataFrame):
        size = len(data)
    elif isinstance(data, pd.Series):
        size = len(data)
    else:
        size = data.shape[0] if data.ndim > 0 else 1

    if size < min_size:
        raise InsufficientDataError(f'{name} has {size} samples, but {min_size} required')


def validate_no_missing_values(data: np.ndarray | pd.Series | pd.DataFrame, name: str = 'data') -> None:
    """Validate that data has no missing values.

    Args:
        data: Data to validate.
        name: Name of the data for error messages.

    Raises:
        MissingDataError: If data contains missing values.
    """
    if isinstance(data, pd.DataFrame | pd.Series):
        if data.isnull().any().any() if isinstance(data, pd.DataFrame) else data.isnull().any():
            raise MissingDataError(f'{name} contains missing values')
    elif isinstance(data, np.ndarray):
        if np.isnan(data).any():
            raise MissingDataError(f'{name} contains NaN values')


def validate_numeric_data(data: np.ndarray | pd.Series | pd.DataFrame, name: str = 'data') -> None:
    """Validate that data is numeric.

    Args:
        data: Data to validate.
        name: Name of the data for error messages.

    Raises:
        InvalidDataTypeError: If data is not numeric.
    """
    if isinstance(data, pd.DataFrame):
        if not data.select_dtypes(include=[np.number]).shape[1] == data.shape[1]:
            raise InvalidDataTypeError(f'{name} must contain only numeric values')
    elif isinstance(data, pd.Series):
        if not pd.api.types.is_numeric_dtype(data):
            raise InvalidDataTypeError(f'{name} must be numeric')
    elif isinstance(data, np.ndarray):
        if not np.issubdtype(data.dtype, np.number):
            raise InvalidDataTypeError(f'{name} must be numeric')


def validate_same_length(
    data1: np.ndarray | pd.Series, data2: np.ndarray | pd.Series, name1: str = 'data1', name2: str = 'data2'
) -> None:
    """Validate that two datasets have the same length.

    Args:
        data1: First dataset.
        data2: Second dataset.
        name1: Name of first dataset for error messages.
        name2: Name of second dataset for error messages.

    Raises:
        InvalidDataShapeError: If datasets have different lengths.
    """
    len1 = len(data1) if isinstance(data1, pd.Series) else data1.shape[0]
    len2 = len(data2) if isinstance(data2, pd.Series) else data2.shape[0]

    if len1 != len2:
        raise InvalidDataShapeError(f'{name1} has length {len1}, but {name2} has length {len2}')


def validate_dataframe_columns(data: pd.DataFrame, required_columns: list[str]) -> None:
    """Validate that DataFrame contains required columns.

    Args:
        data: DataFrame to validate.
        required_columns: List of required column names.

    Raises:
        MissingDataError: If required columns are missing.
    """
    missing = set(required_columns) - set(data.columns)
    if missing:
        raise MissingDataError(f'DataFrame is missing required columns: {missing}')


def validate_categorical_data(data: pd.Series | pd.DataFrame, min_categories: int = 2) -> None:
    """Validate categorical data.

    Args:
        data: Data to validate.
        min_categories: Minimum number of categories required.

    Raises:
        InvalidDataTypeError: If data is not suitable for categorical analysis.
    """
    if isinstance(data, pd.Series):
        n_unique = data.nunique()
        if n_unique < min_categories:
            raise InvalidDataTypeError(
                f'Categorical data must have at least {min_categories} categories, found {n_unique}'
            )
    elif isinstance(data, pd.DataFrame):
        for col in data.columns:
            n_unique = data[col].nunique()
            if n_unique < min_categories:
                raise InvalidDataTypeError(
                    f'Column {col} must have at least {min_categories} categories, found {n_unique}'
                )


def validate_contingency_table(table: np.ndarray | pd.DataFrame, min_cell_count: int = 5) -> None:
    """Validate contingency table for chi-square test.

    Args:
        table: Contingency table to validate.
        min_cell_count: Minimum expected count per cell (default 5 for chi-square).

    Raises:
        DataValidationError: If table is invalid.
        InvalidDataShapeError: If table doesn't have at least 2x2 shape.
    """
    if isinstance(table, pd.DataFrame):
        table = table.to_numpy()

    if table.ndim != 2:
        raise InvalidDataShapeError('Contingency table must be 2-dimensional')

    if table.shape[0] < 2 or table.shape[1] < 2:
        raise InvalidDataShapeError('Contingency table must be at least 2x2')

    if np.any(table < 0):
        raise DataValidationError('Contingency table cannot contain negative values')

    # Check for sufficient expected counts
    row_totals = table.sum(axis=1)
    col_totals = table.sum(axis=0)
    total = table.sum()

    if total == 0:
        raise InsufficientDataError('Contingency table is empty (sum is 0)')

    expected = np.outer(row_totals, col_totals) / total
    if np.any(expected < min_cell_count):
        low_count = np.sum(expected < min_cell_count)
        raise DataValidationError(
            f"{low_count} cells have expected count < {min_cell_count}. Consider using Fisher's exact test instead."
        )


def validate_paired_data(data1: np.ndarray | pd.Series, data2: np.ndarray | pd.Series, alpha: float = 0.05) -> None:
    """Validate data for paired tests.

    Args:
        data1: First dataset.
        data2: Second dataset.
        alpha: Significance level (for context).

    Raises:
        DataValidationError: If data is not valid for paired tests.
    """
    validate_array_not_empty(data1 if isinstance(data1, np.ndarray) else data1.to_numpy(), 'data1')
    validate_array_not_empty(data2 if isinstance(data2, np.ndarray) else data2.to_numpy(), 'data2')
    validate_same_length(data1, data2)
    validate_numeric_data(data1, 'data1')
    validate_numeric_data(data2, 'data2')
    validate_minimum_sample_size(data1, 3, 'data1')


def validate_independent_samples(
    data1: np.ndarray | pd.Series, data2: np.ndarray | pd.Series, alpha: float = 0.05
) -> None:
    """Validate data for independent samples tests.

    Args:
        data1: First dataset.
        data2: Second dataset.
        alpha: Significance level (for context).

    Raises:
        DataValidationError: If data is not valid for independent samples tests.
    """
    validate_array_not_empty(data1 if isinstance(data1, np.ndarray) else data1.to_numpy(), 'data1')
    validate_array_not_empty(data2 if isinstance(data2, np.ndarray) else data2.to_numpy(), 'data2')
    validate_numeric_data(data1, 'data1')
    validate_numeric_data(data2, 'data2')
    validate_minimum_sample_size(data1, 2, 'data1')
    validate_minimum_sample_size(data2, 2, 'data2')


def validate_alpha(alpha: float) -> None:
    """Validate significance level (alpha).

    Args:
        alpha: Significance level to validate.

    Raises:
        DataValidationError: If alpha is not in valid range.
    """
    if not 0 < alpha < 1:
        raise DataValidationError(f'Alpha must be between 0 and 1, got {alpha}')


def validate_test_parameters(**params: Any) -> None:
    """Validate common test parameters.

    Args:
        **params: Test parameters to validate.

    Raises:
        DataValidationError: If parameters are invalid.
    """
    if 'alpha' in params:
        validate_alpha(params['alpha'])

    if 'alternative' in params:
        valid_alternatives = ['two-sided', 'less', 'greater']
        if params['alternative'] not in valid_alternatives:
            raise DataValidationError(f'alternative must be one of {valid_alternatives}, got {params["alternative"]}')
