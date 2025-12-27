"""Simple visualization utilities for analysis results."""

from __future__ import annotations

import base64
import io
import math
from itertools import combinations
from typing import Any

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from pandas.api.types import is_numeric_dtype  # noqa: E402
from scipy import stats  # noqa: E402


class VisualizationService:
    """Generate lightweight diagnostic plots and effect sizes for analyses."""

    @staticmethod
    def _fig_to_base64(fig: plt.Figure) -> str:
        """Convert a Matplotlib figure to a base64-encoded PNG."""
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode('utf-8')

    @staticmethod
    def _cohens_d(group1: pd.Series, group2: pd.Series) -> float | None:
        """Compute Cohen's d for two groups when possible."""
        g1 = group1.dropna()
        g2 = group2.dropna()
        if len(g1) < 3 or len(g2) < 3:
            return None
        diff = g1.mean() - g2.mean()
        pooled_var = ((len(g1) - 1) * g1.var() + (len(g2) - 1) * g2.var()) / (len(g1) + len(g2) - 2)
        pooled_std = math.sqrt(pooled_var) if pooled_var > 0 else 0.0
        if pooled_std == 0:
            return None
        return diff / pooled_std

    @classmethod
    def generate_visualizations(
        cls,
        df: pd.DataFrame,
        *,
        selected_columns: list[str] | None = None,
        limit: int = 8,
    ) -> dict[str, Any]:
        """Create a small set of charts (distribution/box/scatter) and effect sizes."""
        if df is None or df.empty:
            return {'plots': [], 'effect_sizes': {}}

        df_for_plot = df.copy()
        if selected_columns:
            available = [c for c in selected_columns if c in df_for_plot.columns]
            if available:
                df_for_plot = df_for_plot[available]

        if len(df_for_plot) > 2000:
            df_for_plot = df_for_plot.sample(2000, random_state=42)

        numeric_cols = [c for c in df_for_plot.columns if is_numeric_dtype(df_for_plot[c])]
        categorical_cols = [c for c in df_for_plot.columns if c not in numeric_cols]

        plots: list[dict[str, Any]] = []
        effect_sizes: dict[str, float] = {}

        def _add_plot(payload: dict[str, Any]) -> None:
            if len(plots) < limit:
                plots.append(payload)

        # Identify a simple grouping column (binary/category) for effect size + box plots
        group_column = None
        group_levels: list[str] = []
        for col in categorical_cols:
            levels = list(df_for_plot[col].dropna().astype(str).unique())
            if 2 <= len(levels) <= 4:
                group_column = col
                group_levels = levels[:2]
                break

        # Effect sizes per numeric column (binary group only)
        if group_column and numeric_cols:
            level_a, level_b = group_levels[:2]
            for col in numeric_cols:
                g1 = df_for_plot[df_for_plot[group_column].astype(str) == level_a][col]
                g2 = df_for_plot[df_for_plot[group_column].astype(str) == level_b][col]
                d_val = cls._cohens_d(g1, g2)
                if d_val is not None:
                    effect_sizes[f'{col} ({level_a} vs {level_b})'] = round(float(d_val), 4)

        # Histograms / distributions for numeric columns
        for col in numeric_cols:
            series = df_for_plot[col].dropna()
            if series.empty:
                continue
            fig, ax = plt.subplots(figsize=(4, 3))
            ax.hist(series, bins=20, color='#22d3ee', alpha=0.85, edgecolor='#0b1223')
            ax.set_title(f'Distribution: {col}')
            ax.set_xlabel(col)
            ax.set_ylabel('Frequency')
            _add_plot(
                {
                    'title': f'Histogram · {col}',
                    'description': f'Value distribution for {col}.',
                    'image_base64': cls._fig_to_base64(fig),
                    'type': 'histogram',
                    'column': col,
                    'content_type': 'image/png',
                }
            )
            if len(plots) >= limit:
                return {'plots': plots, 'effect_sizes': effect_sizes}

        # Box plots (optionally grouped)
        for col in numeric_cols:
            series = df_for_plot[col].dropna()
            if series.empty:
                continue
            fig, ax = plt.subplots(figsize=(4, 3))
            if group_column:
                grouped = [df_for_plot[df_for_plot[group_column].astype(str) == level][col].dropna() for level in group_levels]
                labels = [f'{group_column}={lvl}' for lvl in group_levels]
                ax.boxplot(grouped, labels=labels, patch_artist=True, boxprops={'facecolor': '#22d3ee', 'alpha': 0.65})
                ax.set_title(f'Box: {col} by {group_column}')
            else:
                ax.boxplot(series, vert=True, patch_artist=True, boxprops={'facecolor': '#22d3ee', 'alpha': 0.65})
                ax.set_title(f'Box Plot: {col}')
                ax.set_xticklabels([col])
            ax.set_ylabel(col)
            _add_plot(
                {
                    'title': f'Box · {col}',
                    'description': f'IQR + outliers for {col}.',
                    'image_base64': cls._fig_to_base64(fig),
                    'type': 'box',
                    'column': col,
                    'content_type': 'image/png',
                }
            )
            if len(plots) >= limit:
                return {'plots': plots, 'effect_sizes': effect_sizes}

        # Scatter plots for leading numeric pairs
        for x_col, y_col in combinations(numeric_cols[:4], 2):
            subset = df_for_plot[[x_col, y_col]].dropna()
            if len(subset) < 10:
                continue
            fig, ax = plt.subplots(figsize=(4, 3))
            ax.scatter(subset[x_col], subset[y_col], alpha=0.7, color='#0ea5e9')
            ax.set_xlabel(x_col)
            ax.set_ylabel(y_col)
            ax.set_title(f'Scatter: {x_col} vs {y_col}')
            _add_plot(
                {
                    'title': f'Scatter · {x_col} vs {y_col}',
                    'description': f'Correlation view for {x_col} and {y_col}.',
                    'image_base64': cls._fig_to_base64(fig),
                    'type': 'scatter',
                    'column': f'{x_col},{y_col}',
                    'content_type': 'image/png',
                }
            )
            if len(plots) >= limit:
                return {'plots': plots, 'effect_sizes': effect_sizes}

        # Q-Q plots to inspect normality
        for col in numeric_cols:
            series = df_for_plot[col].dropna()
            if len(series) < 5:
                continue
            fig, ax = plt.subplots(figsize=(4, 3))
            stats.probplot(series, dist='norm', plot=ax)
            ax.set_title(f'Q-Q Plot: {col}')
            _add_plot(
                {
                    'title': f'Q-Q Plot · {col}',
                    'description': f'Normality check for {col} via quantile-quantile plot.',
                    'image_base64': cls._fig_to_base64(fig),
                    'type': 'qq',
                    'column': col,
                    'content_type': 'image/png',
                }
            )
            if len(plots) >= limit:
                return {'plots': plots, 'effect_sizes': effect_sizes}

        # Simple categorical count plots
        for col in categorical_cols:
            series = df_for_plot[col].astype(str).fillna('NA')
            top_counts = series.value_counts().head(8)
            if top_counts.empty:
                continue
            fig, ax = plt.subplots(figsize=(4, 3))
            top_counts.sort_values().plot(kind='barh', ax=ax, color='#22d3ee')
            ax.set_title(f'Category Counts: {col}')
            ax.set_xlabel('Count')
            _add_plot(
                {
                    'title': f'Counts · {col}',
                    'description': f'Top categories for {col}.',
                    'image_base64': cls._fig_to_base64(fig),
                    'type': 'bar',
                    'column': col,
                    'content_type': 'image/png',
                }
            )
            if len(plots) >= limit:
                return {'plots': plots, 'effect_sizes': effect_sizes}

        return {'plots': plots, 'effect_sizes': effect_sizes}
