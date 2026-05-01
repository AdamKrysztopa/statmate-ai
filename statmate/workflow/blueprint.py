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
    role: Literal['independent', 'dependent', 'covariate', 'group']
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


class InfluenceDiagnostics(BaseModel):
    """Outlier and influence diagnostics computed during blueprint construction."""

    cooks_distance_max: float | None = None
    high_leverage_count: int | None = None
    outlier_count_iqr: int | None = None
    vif_warnings: list[str] = Field(default_factory=list)


class DataBlueprint(BaseModel):
    """Immutable, machine-readable data blueprint."""

    model_config = ConfigDict(frozen=True)

    variable_roles: list[VariableRole] = Field(default_factory=list)
    distribution_metrics: dict[str, DistributionMetric] = Field(default_factory=dict)
    sample_balance: SampleBalance | None = None
    survival_data: bool = False
    is_paired: bool | None = Field(
        default=None,
        description='Whether the data layout is paired/repeated based on structural detection.',
    )
    group_samples: dict[str, int] = Field(
        default_factory=dict, description='Per-group sample counts for routing/sufficiency checks.'
    )
    index_column: str | None = Field(
        default=None,
        description='Column or index name representing the pairing identifier (subject ID).',
    )
    target_column: str | None = Field(
        default=None,
        description='Primary dependent/target column captured during initialization.',
    )
    partition_report: dict[str, Any] | None = Field(
        default=None,
        description='Structured overlap report across groups/IDs emitted by initialization.',
    )
    regression_intent: Literal['none', 'linear', 'logistic'] = Field(
        default='none',
        description='Detected regression intent: linear OLS, logistic, or none.',
    )
    raw: dict[str, Any] = Field(default_factory=dict, description='Original payload from agents if provided')
    influence_diagnostics: InfluenceDiagnostics | None = None


def _compute_distribution_metrics(series: pd.Series) -> DistributionMetric:
    """Compute skewness, kurtosis, and Shapiro-Wilk p-value when possible."""
    try:
        values = pd.to_numeric(series.dropna(), errors='coerce')
        skewness = float(values.skew()) if not values.empty else None
        kurtosis = float(values.kurtosis()) if not values.empty else None
        p_val = None
        if len(values) >= 3:
            _, p_val = stats.shapiro(values)
            p_val = float(p_val)
        return DistributionMetric(skewness=skewness, kurtosis=kurtosis, normality_p_value=p_val)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning('Could not compute distribution metrics: %s', exc)
        return DistributionMetric()


def _detect_survival_columns(df: pd.DataFrame) -> bool:
    """Heuristic detection for survival/time-to-event layouts."""
    cols = [str(col).lower() for col in df.columns]
    survival_tokens = ('event', 'status', 'censored', 'survival', 'time_to_event', 'time')
    has_time = any('time' in col or 'duration' in col for col in cols)
    has_event = any(tok in col for col in cols for tok in ('event', 'status', 'censor'))
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


def _compute_influence(df: pd.DataFrame, target_col: str) -> InfluenceDiagnostics:
    """Compute Cook's D, leverage, IQR outlier count, and VIF warnings."""
    import numpy as np
    from statsmodels.regression.linear_model import OLS
    from statsmodels.stats.outliers_influence import OLSInfluence, variance_inflation_factor
    from statsmodels.tools import add_constant

    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c != target_col]
    if len(numeric_cols) < 1 or target_col not in df.columns:
        return InfluenceDiagnostics()

    sub = df[[target_col] + numeric_cols].dropna()
    if len(sub) < len(numeric_cols) + 2:
        return InfluenceDiagnostics()

    y = sub[target_col]
    X = sub[numeric_cols]

    # IQR outlier count on target
    q1 = float(y.quantile(0.25))
    q3 = float(y.quantile(0.75))
    iqr = q3 - q1
    outlier_count_iqr = int(((y < q1 - 1.5 * iqr) | (y > q3 + 1.5 * iqr)).sum())

    try:
        X_const = add_constant(X)
        model = OLS(y, X_const).fit()
        influence = OLSInfluence(model)
        cooks_d, _ = influence.cooks_distance
        cooks_distance_max = float(np.max(cooks_d))
        n, k = len(sub), X.shape[1]
        hat = influence.hat_matrix_diag
        threshold = 2.0 * (k + 1) / n
        high_leverage_count = int((hat > threshold).sum())

        # VIF warnings for predictors with VIF > 5
        X_arr = X_const.values
        vif_warnings: list[str] = []
        for i, col in enumerate(X_const.columns):
            if col == 'const':
                continue
            vif_val = float(variance_inflation_factor(X_arr, i))
            if vif_val > 5.0:
                vif_warnings.append(str(col))
    except Exception:
        cooks_distance_max = None
        high_leverage_count = None
        vif_warnings = []

    return InfluenceDiagnostics(
        cooks_distance_max=cooks_distance_max,
        high_leverage_count=high_leverage_count,
        outlier_count_iqr=outlier_count_iqr,
        vif_warnings=vif_warnings,
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

    provided_roles = (raw_payload or {}).get('variable_roles') if raw_payload else None
    if provided_roles:
        for role in provided_roles:
            name = str(role.get('name'))
            declared_role = role.get('role')
            if declared_role and name:
                roles.append(VariableRole(name=name, role=declared_role))

    for dep in dep_vars:
        if dep not in {r.name for r in roles}:
            roles.append(VariableRole(name=str(dep), role='dependent'))
    if group_var and group_var not in {r.name for r in roles}:
        roles.append(VariableRole(name=str(group_var), role='group'))
    for cov in covariate_vars:
        if cov not in {r.name for r in roles}:
            roles.append(VariableRole(name=str(cov), role='covariate'))

    # Remaining columns default to independent roles
    used = {r.name for r in roles}
    for col in df.columns:
        if col not in used:
            roles.append(VariableRole(name=str(col), role='independent'))

    distribution_metrics: dict[str, DistributionMetric] = {}
    provided_metrics = (raw_payload or {}).get('distribution_metrics') if raw_payload else None
    if provided_metrics:
        for var, metrics in provided_metrics.items():
            distribution_metrics[str(var)] = DistributionMetric(**metrics)

    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]) and str(col) not in distribution_metrics:
            distribution_metrics[str(col)] = _compute_distribution_metrics(df[col])

    sample_balance_payload = (raw_payload or {}).get('sample_balance') if raw_payload else None
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
        group_samples = {'__all__': int(total_n)}

    # Resolve pairing + identifier/target metadata
    paired_flag = is_paired
    if paired_flag is None and raw_payload:
        design_hint = raw_payload.get('data_design')
        if design_hint == 'paired':
            paired_flag = True
        elif design_hint == 'independent':
            paired_flag = False
    if paired_flag is None:
        wide_detection = detect_wide_format_pairing(df.columns)
        if wide_detection.get('detected'):
            paired_flag = True
            if wide_detection.get('pairs'):
                first_pair = wide_detection['pairs'][0]
                if first_pair[0] in df.columns and first_pair[1] in df.columns:
                    aligned = df[list(first_pair)].dropna()
                    group_samples = {str(first_pair[0]): len(aligned), str(first_pair[1]): len(aligned)}
    idx_col = index_column or (raw_payload or {}).get('index_column') or df.index.name
    tgt_col = target_column or (raw_payload or {}).get('target_column')

    # Detect regression intent from data layout
    regression_intent_val: Literal['none', 'linear', 'logistic'] = 'none'
    if tgt_col and tgt_col in df.columns:
        numeric_predictors = [col for col in df.columns if col != tgt_col and pd.api.types.is_numeric_dtype(df[col])]
        target_series = df[tgt_col].dropna()
        unique_vals = target_series.nunique()
        if unique_vals == 2 and len(numeric_predictors) >= 1:
            regression_intent_val = 'logistic'
        elif pd.api.types.is_numeric_dtype(df[tgt_col]) and len(numeric_predictors) >= 2:
            regression_intent_val = 'linear'

    # Compute influence diagnostics when a numeric target is available
    influence_diag: InfluenceDiagnostics | None = None
    if tgt_col and tgt_col in df.columns and pd.api.types.is_numeric_dtype(df[tgt_col]):
        try:
            influence_diag = _compute_influence(df, tgt_col)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning('Influence diagnostics failed: %s', exc)
            influence_diag = None

    return DataBlueprint(
        variable_roles=roles,
        distribution_metrics=distribution_metrics,
        sample_balance=sample_balance,
        survival_data=_detect_survival_columns(df),
        is_paired=paired_flag,
        group_samples=group_samples,
        index_column=idx_col,
        target_column=tgt_col,
        partition_report=partition_report or (raw_payload or {}).get('partition_report'),
        regression_intent=regression_intent_val,
        raw=raw_payload or {},
        influence_diagnostics=influence_diag,
    )
