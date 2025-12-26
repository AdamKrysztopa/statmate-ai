"""Simple visualization utilities for analysis results."""

from __future__ import annotations

import base64
import io
from typing import Any

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from pandas.api.types import is_numeric_dtype  # noqa: E402
from scipy import stats  # noqa: E402


class VisualizationService:
    """Generate lightweight diagnostic plots for analyses."""

    @staticmethod
    def _fig_to_base64(fig: plt.Figure) -> str:
        """Convert a Matplotlib figure to a base64-encoded PNG."""
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode('utf-8')

    @classmethod
    def generate_visualizations(
        cls,
        df: pd.DataFrame,
        *,
        selected_columns: list[str] | None = None,
        limit: int = 6,
    ) -> list[dict[str, Any]]:
        """Create a small set of charts (histograms, QQ plots, category bars)."""
        if df is None or df.empty:
            return []

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

        def _add_plot(payload: dict[str, Any]) -> None:
            if len(plots) < limit:
                plots.append(payload)

        # Histograms for numeric columns
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
                return plots

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
                return plots

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
                return plots

        return plots
