"""Data validation utilities for StatMate AI.

This module provides functions to validate input data for statistical tests
and workflows.
"""

from dataclasses import dataclass, field
from typing import Any, Literal, Sequence

import numpy as np
import pandas as pd

from statmate.core.config import default_config
from statmate.core.exceptions import (
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


def _flatten_numeric_array(data: np.ndarray | pd.Series | pd.DataFrame) -> np.ndarray:
    """Convert input data to a 1D numeric numpy array."""
    if isinstance(data, pd.DataFrame):
        numeric = data.select_dtypes(include=[np.number])
        arr = numeric.to_numpy().ravel()
    elif isinstance(data, pd.Series):
        arr = data.to_numpy().ravel()
    else:
        arr = np.asarray(data).ravel()
    try:
        return arr.astype(float, copy=False)
    except (TypeError, ValueError):
        coerced = pd.to_numeric(arr, errors='coerce')
        return np.asarray(coerced, dtype=float)


def validate_assumptions(
    data: np.ndarray | pd.Series | pd.DataFrame,
    test_type: str,
    secondary_data: np.ndarray | pd.Series | pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Compute standardized diagnostics for key statistical assumptions.

    The helper returns a structured payload instead of raising; callers can
    log and stream the failures for transparency.

    Args:
        data: Primary sample data.
        test_type: Identifier for the test being evaluated.
        secondary_data: Optional comparison sample (used for variance ratios).

    Returns:
        Dictionary containing diagnostics and any detected failures.
    """
    thresholds = {
        'skewness': default_config.statistical.skewness_threshold,
        'kurtosis': default_config.statistical.kurtosis_threshold,
        'variance_ratio': default_config.statistical.variance_ratio_threshold,
        'sparsity': default_config.statistical.sparsity_threshold,
    }

    primary_arr = _flatten_numeric_array(data)
    failures: list[str] = []

    skewness = float(pd.Series(primary_arr).skew()) if primary_arr.size else None
    if skewness is not None and abs(skewness) > thresholds['skewness']:
        failures.append(f'High skewness ({skewness:.2f}) exceeds |{thresholds["skewness"]}| threshold.')

    kurt = float(pd.Series(primary_arr).kurtosis()) if primary_arr.size else None
    if kurt is not None and kurt > thresholds['kurtosis']:
        failures.append(f'Heavy tails (kurtosis {kurt:.2f}) above {thresholds["kurtosis"]}.')

    variance_ratio = None
    if secondary_data is not None:
        secondary_arr = _flatten_numeric_array(secondary_data)
        var_a = float(np.nanvar(primary_arr, ddof=1)) if primary_arr.size > 1 else 0.0
        var_b = float(np.nanvar(secondary_arr, ddof=1)) if secondary_arr.size > 1 else 0.0
        if var_a > 0 and var_b > 0:
            high = max(var_a, var_b)
            low = min(var_a, var_b)
            variance_ratio = high / low if low > 0 else None
            if variance_ratio and variance_ratio > thresholds['variance_ratio']:
                failures.append(
                    f'Variance ratio {variance_ratio:.2f} exceeds threshold {thresholds["variance_ratio"]}; variances differ.'
                )

    zero_like = np.isnan(primary_arr) | (primary_arr == 0)
    sparsity = float(np.mean(zero_like)) if primary_arr.size else None
    if sparsity is not None and sparsity > thresholds['sparsity']:
        failures.append(
            f'Sparsity {sparsity:.2%} above threshold {thresholds["sparsity"]:.0%}; many zero/empty values.'
        )

    return {
        'test_type': test_type,
        'skewness': skewness,
        'kurtosis': kurt,
        'variance_ratio': variance_ratio,
        'sparsity': sparsity,
        'thresholds': thresholds,
        'failures': failures,
    }


@dataclass
class StatisticalDesign:
    """Structured output describing the detected study design."""

    design_type: Literal['independent', 'paired', 'mixed']
    is_paired: bool
    grouping_variable: str | None
    subject_id_column: str | None
    rationale: str
    overlap_summary: dict[str, Any] = field(default_factory=dict)
    comparison_matrix: dict[str, Any] = field(default_factory=dict)
    keyword_cues: dict[str, list[str]] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        """Convert to a JSON-serialisable dictionary."""
        return {
            'design_type': self.design_type,
            'is_paired': self.is_paired,
            'grouping_variable': self.grouping_variable,
            'subject_id_column': self.subject_id_column,
            'rationale': self.rationale,
            'overlap_summary': self.overlap_summary,
            'comparison_matrix': self.comparison_matrix,
            'keyword_cues': self.keyword_cues,
        }


def _keyword_cues(columns: Sequence[str]) -> dict[str, list[str]]:
    """Detect temporal vs grouping cues from column names."""
    temporal_tokens = ('time', 'visit', 'week', 'month', 'follow', 'day', 'year')
    grouping_tokens = ('group', 'arm', 'cohort', 'treatment', 'variant', 'condition')
    lower = {col: col.lower() for col in columns}
    temporal_like = [col for col, low in lower.items() if any(tok in low for tok in temporal_tokens)]
    group_like = [col for col, low in lower.items() if any(tok in low for tok in grouping_tokens)]
    return {'temporal_like': temporal_like, 'group_like': group_like}


def _candidate_subject_columns(df: pd.DataFrame, provided: Sequence[str] | None = None) -> list[str]:
    """Identify plausible subject ID columns."""
    if provided:
        return [col for col in provided if col in df.columns]
    keywords = ('id', 'subject', 'participant', 'patient', 'user')
    candidates = []
    for col in df.columns:
        low = str(col).lower()
        if any(k in low for k in keywords):
            candidates.append(col)
    if df.index.name:
        candidates.append(df.index.name)
    return list(dict.fromkeys(candidates))  # preserve order, deduplicate


def _candidate_group_columns(df: pd.DataFrame, provided: Sequence[str] | None = None) -> list[str]:
    """Identify columns likely representing grouping/arms."""
    if provided:
        return [col for col in provided if col in df.columns]
    max_cardinality = max(2, int(len(df) * 0.5))
    cat_cols = []
    for col in df.columns:
        nunique = df[col].nunique(dropna=True)
        if nunique <= max_cardinality and not pd.api.types.is_numeric_dtype(df[col]):
            cat_cols.append(col)
    return cat_cols


def _build_comparison_matrix(df: pd.DataFrame, subject_col: str, group_col: str) -> dict[str, Any]:
    """Enumerate cross-group overlaps for mixed designs."""
    subset = df[[subject_col, group_col]].dropna()
    levels = list(subset[group_col].unique())
    id_sets = {lvl: set(subset[subset[group_col] == lvl][subject_col]) for lvl in levels}
    comparisons: list[dict[str, Any]] = []
    for i, group_a in enumerate(levels):
        for group_b in levels[i + 1 :]:
            shared = len(id_sets[group_a] & id_sets[group_b])
            comparisons.append(
                {
                    'group_a': group_a,
                    'group_b': group_b,
                    'shared_ids': shared,
                    'comparison_type': 'paired' if shared > 0 else 'independent',
                }
            )
    return {'groups': levels, 'comparisons': comparisons}


def _overlap_summary(df: pd.DataFrame, subject_col: str, group_col: str) -> dict[str, Any]:
    """Compute overlap diagnostics for subject/group pairs."""
    overlaps = df.groupby(subject_col)[group_col].nunique(dropna=True)
    shared_ids = int((overlaps > 1).sum())
    repeated_within_group = int(df.duplicated(subset=[subject_col, group_col]).sum())
    total_ids = int(df[subject_col].nunique(dropna=True))
    overlap_rate = float(shared_ids / total_ids) if total_ids else 0.0
    comparison_matrix = _build_comparison_matrix(df, subject_col, group_col)
    return {
        'shared_ids_across_groups': shared_ids,
        'repeated_within_group': repeated_within_group,
        'total_subjects': total_ids,
        'overlap_rate': overlap_rate,
        'comparison_matrix': comparison_matrix,
    }


def infer_statistical_design(
    data: pd.DataFrame | pd.Series,
    group_candidates: Sequence[str] | None = None,
    subject_id_candidates: Sequence[str] | None = None,
) -> tuple[StatisticalDesign, dict[str, Any]]:
    """Infer whether the design is paired, independent, or mixed based on structure.

    Args:
        data: Input dataset.
        group_candidates: Optional candidate grouping columns.
        subject_id_candidates: Optional candidate subject ID columns.

    Returns:
        Tuple of (StatisticalDesign, structural_summary).
    """
    frame = data.to_frame() if isinstance(data, pd.Series) else data.copy()
    summary: dict[str, Any] = {
        'n_rows': int(len(frame)),
        'n_cols': int(frame.shape[1]),
        'nunique_by_column': {col: int(frame[col].nunique(dropna=True)) for col in frame.columns},
    }
    summary['keyword_cues'] = _keyword_cues(frame.columns)

    subject_cols = _candidate_subject_columns(frame, subject_id_candidates)
    group_cols = _candidate_group_columns(frame, group_candidates)
    summary['subject_id_candidates'] = subject_cols
    summary['group_candidates'] = group_cols

    best_pair: tuple[str, str] | None = None
    best_overlap: dict[str, Any] = {}
    best_score = -1

    for subj in subject_cols:
        subj_series = frame[subj] if subj in frame.columns else pd.Series(frame.index, name=subj)
        if subj_series.isna().all():
            continue
        for grp in group_cols:
            if grp == subj:
                continue
            candidate_df = frame[[subj, grp]].dropna()
            if candidate_df.empty:
                continue
            overlap = _overlap_summary(candidate_df, subj, grp)
            score = overlap['shared_ids_across_groups'] + overlap['repeated_within_group']
            if score > best_score:
                best_score = score
                best_overlap = overlap
                best_pair = (subj, grp)

    # Fallback: row-wise paired signals (wide format)
    paired_by_row = False
    if best_pair is None:
        numeric_cols = frame.select_dtypes(include=[np.number]).columns
        paired_by_row = len(numeric_cols) >= 2 and len(frame) > 1
        if paired_by_row:
            best_overlap = {'paired_by_row': True, 'shared_ids_across_groups': 0, 'repeated_within_group': 0}

    design_type: Literal['independent', 'paired', 'mixed'] = 'independent'
    grouping_variable: str | None = None
    subject_id_column: str | None = None
    rationale_parts: list[str] = []

    if best_pair:
        subject_id_column, grouping_variable = best_pair
        overlap = best_overlap
        comparison_matrix = overlap.get('comparison_matrix', {})
        shared = overlap.get('shared_ids_across_groups', 0)
        repeated = overlap.get('repeated_within_group', 0)
        if shared > 0 and repeated > 0:
            design_type = 'mixed'
            rationale_parts.append(
                f'{shared} subject IDs appear across groups and {repeated} repeated entries within at least one group.'
            )
        elif shared > 0 or repeated > 0:
            design_type = 'paired'
            if shared > 0:
                rationale_parts.append(f'{shared} subject IDs overlap across groups, implying paired/crossover data.')
            if repeated > 0:
                rationale_parts.append(f'{repeated} repeated IDs within a group suggest longitudinal tracking.')
        else:
            rationale_parts.append('No subject overlaps detected across groups; treating as independent samples.')
        best_overlap.setdefault('comparison_matrix', comparison_matrix)
    elif paired_by_row:
        design_type = 'paired'
        rationale_parts.append('Multiple measurement columns per row imply a paired/wide layout.')
    else:
        rationale_parts.append('No ID/group overlap found; defaulting to independent design.')

    design = StatisticalDesign(
        design_type=design_type,
        is_paired=design_type != 'independent',
        grouping_variable=grouping_variable,
        subject_id_column=subject_id_column,
        rationale=' '.join(rationale_parts).strip(),
        overlap_summary=best_overlap,
        comparison_matrix=best_overlap.get('comparison_matrix', {}),
        keyword_cues=summary['keyword_cues'],
    )
    summary['overlap_summary'] = best_overlap
    summary['design_type'] = design_type
    return design, summary
