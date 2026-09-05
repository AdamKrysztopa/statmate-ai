"""Phase 1: deterministic structural checks for initialization pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from statmate.core.validation import (
    StatisticalDesign,
    detect_wide_format_pairing,
    infer_statistical_design,
)


@dataclass
class StructuralCheckResult:
    """Deterministic schema and design pre-check output."""

    is_valid: bool
    design: StatisticalDesign | None
    structural_summary: dict[str, Any]
    row_count: int
    column_count: int
    wide_format_detected: bool
    errors: list[str] = field(default_factory=list)


def check_structural_validity(df: pd.DataFrame | pd.Series) -> StructuralCheckResult:
    """Run deterministic structural checks before any agent reasoning."""
    frame = df if isinstance(df, pd.DataFrame) else pd.DataFrame(df)
    errors: list[str] = []

    if frame.empty:
        errors.append('Dataset is empty.')

    if frame.shape[1] == 0:
        errors.append('Dataset has no columns.')

    design, structural_summary = infer_statistical_design(frame)
    wide_detection = detect_wide_format_pairing(frame.columns)

    return StructuralCheckResult(
        is_valid=not errors,
        design=design,
        structural_summary=structural_summary,
        row_count=frame.shape[0],
        column_count=frame.shape[1],
        wide_format_detected=bool(wide_detection),
        errors=errors,
    )


def build_structural_prompt_payload(result: StructuralCheckResult) -> dict[str, Any]:
    """Serialize structural results for downstream agent prompts."""
    return {
        'statistical_design': result.design.as_dict() if result.design else None,
        'structural_summary': result.structural_summary,
        'wide_format_detected': result.wide_format_detected,
        'row_count': result.row_count,
        'column_count': result.column_count,
        'errors': result.errors,
    }
