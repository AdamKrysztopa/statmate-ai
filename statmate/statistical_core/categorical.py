"""Helper utilities for categorical and contingency analyses."""

from __future__ import annotations

import numpy as np
import pandas as pd

from statmate.core.exceptions import DataValidationError, InvalidDataShapeError
from statmate.core.validation import validate_contingency_table


def prepare_contingency_table(table: np.ndarray | pd.DataFrame | list[list[int]]) -> np.ndarray:
    """Convert a contingency-like input to a validated 2D numpy array."""
    arr = table.to_numpy() if isinstance(table, pd.DataFrame) else np.asarray(table)
    if arr.ndim != 2:
        raise InvalidDataShapeError('Contingency table must be 2-dimensional')
    validate_contingency_table(arr, min_cell_count=0)
    return arr


def is_2x2_table(table: np.ndarray | pd.DataFrame | list[list[int]]) -> bool:
    """Check whether a contingency table is 2x2."""
    arr = table if isinstance(table, np.ndarray) else prepare_contingency_table(table)
    return arr.shape == (2, 2)


def expected_counts(table: np.ndarray | pd.DataFrame | list[list[int]]) -> np.ndarray:
    """Compute expected frequencies for a contingency table."""
    arr = prepare_contingency_table(table)
    total = arr.sum()
    row_totals = arr.sum(axis=1)
    col_totals = arr.sum(axis=0)
    return np.outer(row_totals, col_totals) / total


def has_small_expected_counts(table: np.ndarray | pd.DataFrame | list[list[int]], threshold: float = 5.0) -> bool:
    """Return True when any expected cell frequency falls below the threshold."""
    return bool(np.any(expected_counts(table) < threshold))


def cramers_v_from_chi2(chi2_statistic: float, table: np.ndarray | pd.DataFrame | list[list[int]]) -> float | None:
    """Compute Cramer's V effect size using a chi-square statistic."""
    arr = prepare_contingency_table(table)
    n = arr.sum()
    if n <= 0:
        return None
    r, c = arr.shape
    denom = min(r - 1, c - 1)
    if denom <= 0:
        return None
    return float(np.sqrt(chi2_statistic / (n * denom)))


def phi_coefficient(table: np.ndarray | pd.DataFrame | list[list[int]]) -> float | None:
    """Compute Phi coefficient for a 2x2 contingency table."""
    arr = prepare_contingency_table(table)
    if arr.shape != (2, 2):
        raise DataValidationError('Phi coefficient requires a 2x2 contingency table.')
    a, b = arr[0, 0], arr[0, 1]
    c, d = arr[1, 0], arr[1, 1]
    denom = np.sqrt((a + b) * (c + d) * (a + c) * (b + d))
    if denom == 0:
        return None
    return float((a * d - b * c) / denom)


def categorical_effect_size(
    table: np.ndarray | pd.DataFrame | list[list[int]],
    chi2_statistic: float | None = None,
) -> tuple[float | None, str | None]:
    """Return appropriate effect size (Phi or Cramer's V) for a contingency table."""
    arr = prepare_contingency_table(table)
    n = arr.sum()
    if n <= 0:
        return None, None

    if arr.shape == (2, 2):
        if chi2_statistic is not None:
            effect = float(np.sqrt(chi2_statistic / n))
        else:
            effect = phi_coefficient(arr)
        if effect is None:
            return None, None
        return effect, 'phi'

    if chi2_statistic is None:
        return None, None

    effect = cramers_v_from_chi2(chi2_statistic, arr)
    if effect is None:
        return None, None
    return effect, 'cramers_v'
