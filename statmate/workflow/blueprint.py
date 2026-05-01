"""Data blueprint schema for routing and audit decisions.

This module defines structured, typed metadata describing the dataset and
computed distribution diagnostics that downstream routing engines can use
without re-inspecting the raw data.
"""

from __future__ import annotations

from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field
from scipy import stats

from statmate.core import get_logger
from statmate.core.validation import detect_wide_format_pairing

logger = get_logger(__name__)


class VariableRole(BaseModel):
    """Describe the semantic role of a variable."""

    name: str
    role: Literal["independent", "dependent", "covariate", "group"]
    description: str | None = None


class DistributionMetric(BaseModel):
    """Distribution diagnostics for a variable."""

    skewness: float | None = None
    kurtosis: float | None = None
    normality_p_value: float | None = None


class SampleBalance(BaseModel):
    """Balance diagnostics across groups."""

    balanced: bool = True
    balance_ratio: float | None = None
    group_sizes: dict[str, int] = Field(default_factory=dict)


class DataBlueprint(BaseModel):
    """Immutable, machine-readable data blueprint."""

    model_config = ConfigDict(frozen=True)

    variable_roles: list[VariableRole] = Field(default_factory=list)
    distribution_metrics: dict[str, DistributionMetric] = Field(default_factory=dict)
    sample_balance: SampleBalance | None = None
    survival_data: bool = False
    is_paired: bool | None = Field(
        default=None,
        description="Whether the data layout is paired/repeated based on structural detection.",
    )
    group_samples: dict[str, int] = Field(
        default_factory=dict, description="Per-group sample counts for routing/sufficiency checks."
    )
    index_column: str | None = Field(
        default=None,
        description="Column or index name representing the pairing identifier (subject ID).",
    )
    target_column: str | None = Field(
        default=None,
        description="Primary dependent/target column captured during initialization.",
    )
    partition_report: dict[str, Any] | None = Field(
        default=None,
        description="Structured overlap report across groups/IDs emitted by initialization.",
    )
    raw: dict[str, Any] = Field(default_factory=dict, description="Original payload from agents if provided")


def _compute_distribution_metrics(series: pd.Series) -> DistributionMetric:
    """Compute skewness, kurtosis, and Shapiro-Wilk p-value when possible."""
    try:
        values = pd.to_numeric(series.dropna(), errors="coerce")
        skewness = float(values.skew()) if not values.empty else None
        kurtosis = float(values.kurtosis()) if not values.empty else None
        p_val = None
        if len(values) >= 3:
            _, p_val = stats.shapiro(values)
            p_val = float(p_val)
        return DistributionMetric(skewness=skewness, kurtosis=kurtosis, normality_p_value=p_val)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Could not compute distribution metrics: %s", exc)
        return DistributionMetric()


def _detect_survival_columns(df: pd.DataFrame) -> bool:
    """Heuristic detection for survival/time-to-event layouts."""
    cols = [str(col).lower() for col in df.columns]
    survival_tokens = ("event", "status", "censored", "survival", "time_to_event", "time")
    has_time = any("time" in col or "duration" in col for col in cols)
    has_event = any(tok in col for col in cols for tok in ("event", "status", "censor"))
    return has_time and has_event or any(tok in col for col in cols for tok in survival_tokens)


def _compute_sample_balance(df: pd.DataFrame, group_col: str | None) -> SampleBalance | None:
    """Compute balance ratio for grouping column if available."""
    if not group_col or group_col not in df.columns:
        return None
    counts = df[group_col].value_counts(dropna=False).to_dict()
    if not counts:
        return None
    max_size = max(counts.values())
    min_size = min(counts.values())
    ratio = float(min_size / max_size) if max_size else None
    balanced = ratio is None or ratio >= 0.8
    return SampleBalance(
        balanced=balanced, balance_ratio=ratio, group_sizes={str(k): int(v) for k, v in counts.items()}
    )


def build_data_blueprint(
    data: pd.DataFrame,
    *,
    dependent_vars: list[str] | None = None,
    group_var: str | None = None,
    covariates: list[str] | None = None,
    is_paired: bool | None = None,
    index_column: str | None = None,
    target_column: str | None = None,
    partition_report: dict[str, Any] | None = None,
    raw_payload: dict[str, Any] | None = None,
) -> DataBlueprint:
    """Create a DataBlueprint with deterministic, typed metadata."""
    df = data.copy()
    if isinstance(df, pd.Series):
        df = df.to_frame()
    total_n = len(df)

    dep_vars = dependent_vars or []
    covariate_vars = covariates or []
    roles: list[VariableRole] = []

    provided_roles = (raw_payload or {}).get("variable_roles") if raw_payload else None
    if provided_roles:
        for role in provided_roles:
            name = str(role.get("name"))
            declared_role = role.get("role")
            if declared_role and name:
                roles.append(VariableRole(name=name, role=declared_role))

    for dep in dep_vars:
        if dep not in {r.name for r in roles}:
            roles.append(VariableRole(name=str(dep), role="dependent"))
    if group_var and group_var not in {r.name for r in roles}:
        roles.append(VariableRole(name=str(group_var), role="group"))
    for cov in covariate_vars:
        if cov not in {r.name for r in roles}:
            roles.append(VariableRole(name=str(cov), role="covariate"))

    # Remaining columns default to independent roles
    used = {r.name for r in roles}
    for col in df.columns:
        if col not in used:
            roles.append(VariableRole(name=str(col), role="independent"))

    distribution_metrics: dict[str, DistributionMetric] = {}
    provided_metrics = (raw_payload or {}).get("distribution_metrics") if raw_payload else None
    if provided_metrics:
        for var, metrics in provided_metrics.items():
            distribution_metrics[str(var)] = DistributionMetric(**metrics)

    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]) and str(col) not in distribution_metrics:
            distribution_metrics[str(col)] = _compute_distribution_metrics(df[col])

    sample_balance_payload = (raw_payload or {}).get("sample_balance") if raw_payload else None
    sample_balance = None
    if sample_balance_payload:
        sample_balance = SampleBalance(**sample_balance_payload)
    sample_balance = sample_balance or _compute_sample_balance(df, group_var)

    # Group sample counts for sufficiency checks
    group_samples: dict[str, int] = {}
    if group_var and group_var in df.columns:
        counts = df[group_var].value_counts(dropna=False).to_dict()
        group_samples = {str(k): int(v) for k, v in counts.items()}
    elif total_n:
        group_samples = {"__all__": int(total_n)}

    # Resolve pairing + identifier/target metadata
    paired_flag = is_paired
    if paired_flag is None and raw_payload:
        design_hint = raw_payload.get("data_design")
        if design_hint == "paired":
            paired_flag = True
        elif design_hint == "independent":
            paired_flag = False
    if paired_flag is None:
        wide_detection = detect_wide_format_pairing(df.columns)
        if wide_detection.get("detected"):
            paired_flag = True
            if wide_detection.get("pairs"):
                first_pair = wide_detection["pairs"][0]
                if first_pair[0] in df.columns and first_pair[1] in df.columns:
                    aligned = df[list(first_pair)].dropna()
                    group_samples = {str(first_pair[0]): len(aligned), str(first_pair[1]): len(aligned)}
    idx_col = index_column or (raw_payload or {}).get("index_column") or df.index.name
    tgt_col = target_column or (raw_payload or {}).get("target_column")

    return DataBlueprint(
        variable_roles=roles,
        distribution_metrics=distribution_metrics,
        sample_balance=sample_balance,
        survival_data=_detect_survival_columns(df),
        is_paired=paired_flag,
        group_samples=group_samples,
        index_column=idx_col,
        target_column=tgt_col,
        partition_report=partition_report or (raw_payload or {}).get("partition_report"),
        raw=raw_payload or {},
    )
