"""Lightweight PII detection and masking utilities."""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

PII_PATTERNS: dict[str, re.Pattern[str]] = {
    'email': re.compile(r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}'),
    'phone': re.compile(r'(\+?\d{1,3}[\s-]?)?(\(?\d{3}\)?[\s-]?)?\d{3}[\s-]?\d{4}'),
    'ssn': re.compile(r'\b\d{3}-\d{2}-\d{4}\b'),
    'credit_card': re.compile(r'\b(?:\d[ -]*?){13,16}\b'),
}

POSSIBLE_PII_NAMES = {
    'email',
    'mail',
    'phone',
    'mobile',
    'contact',
    'ssn',
    'social',
    'address',
    'street',
    'zip',
    'name',
    'full_name',
    'patient',
    'user',
    'id',
    'identifier',
}


def _mask_value(value: Any, keep_numeric: bool = False) -> Any:
    """Mask a single value while keeping dtype compatibility."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return value
    if keep_numeric:
        return abs(hash(str(value))) % 10_000
    return '[MASKED]'


def _column_has_pii(name: str, series: pd.Series) -> tuple[bool, str]:
    """Check if a column appears to contain PII."""
    lower = name.lower()
    for token in POSSIBLE_PII_NAMES:
        if token in lower:
            return True, f'column name contains "{token}"'

    if series.dtype == object or pd.api.types.is_string_dtype(series):
        sample = series.dropna().astype(str).head(50)
        for label, pattern in PII_PATTERNS.items():
            if sample.str.contains(pattern).any():
                return True, f'matched {label} pattern'
    return False, ''


def mask_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Mask obvious PII in a DataFrame and return a report."""
    masked_df = df.copy()
    masked_columns: list[dict[str, Any]] = []

    for col in masked_df.columns:
        series = masked_df[col]
        has_pii, reason = _column_has_pii(str(col), series)
        if not has_pii:
            continue

        keep_numeric = pd.api.types.is_numeric_dtype(series)
        masked_series = series.map(lambda v: _mask_value(v, keep_numeric=keep_numeric))
        masked_df[col] = masked_series
        masked_columns.append(
            {
                'column': str(col),
                'reason': reason or 'potential identifier',
                'masked_values': int(series.notna().sum()),
            }
        )

    report = {
        'timestamp': datetime.utcnow().isoformat(),
        'masked_columns': masked_columns,
        'note': 'PII masking applied prior to LLM calls' if masked_columns else 'No obvious PII detected',
    }

    return masked_df, report


def sanitize_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Public entry point to run masking with a stable interface."""
    if not isinstance(df, pd.DataFrame):
        try:
            df = pd.DataFrame(df)
        except Exception:
            return df, {'note': 'Unable to coerce input to DataFrame for masking'}
    return mask_dataframe(df)
