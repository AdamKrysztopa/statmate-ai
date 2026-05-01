"""Simple visualization utilities for analysis results."""

from __future__ import annotations

import base64
import io
import math
from itertools import combinations
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from pandas.api.types import is_numeric_dtype  # noqa: E402
from scipy import stats  # noqa: E402


class VisualizationService:
    """Generate lightweight diagnostic plots and effect sizes for analyses."""

    @staticmethod
    def _format_statistic(value: float) -> str:
        return f"{value:.2f}"

    @classmethod
    def _build_plot_payload(
        cls,
        *,
        title: str,
        description: str,
        image_base64: str,
        plot_type: str,
        column: str,
        caption: str,
    ) -> dict[str, Any]:
        return {
            "title": title,
            "description": description,
            "image_base64": image_base64,
            "type": plot_type,
            "column": column,
            "content_type": "image/png",
            "caption": caption,
        }

    @classmethod
    def _histogram_caption(cls, series: pd.Series, column: str) -> str:
        q1, median, q3 = series.quantile([0.25, 0.5, 0.75]).tolist()
        return (
            f"This histogram shows how {column} is distributed across the analysed observations. "
            f"The middle half of values lies between {cls._format_statistic(q1)} and {cls._format_statistic(q3)}, "
            f"with a median of {cls._format_statistic(median)}."
        )

    @classmethod
    def _box_caption(
        cls,
        *,
        series: pd.Series,
        column: str,
        group_column: str | None,
        grouped_series: dict[str, pd.Series] | None,
    ) -> str:
        if group_column and grouped_series:
            medians = {level: values.median() for level, values in grouped_series.items() if not values.empty}
            if medians:
                highest_group = max(medians, key=medians.get)
                highest_median = medians[highest_group]
                return (
                    f"This box plot compares the distribution of {column} across groups of {group_column}. "
                    f"The highest median appears in {group_column}={highest_group} "
                    f"({cls._format_statistic(float(highest_median))}), and each box shows the interquartile range."
                )
        q1, median, q3 = series.quantile([0.25, 0.5, 0.75]).tolist()
        return (
            f"This box plot summarises {column}. The median is {cls._format_statistic(median)}, "
            f"and the box spans the interquartile range from {cls._format_statistic(q1)} "
            f"to {cls._format_statistic(q3)}."
        )

    @classmethod
    def _scatter_caption(cls, subset: pd.DataFrame, x_col: str, y_col: str) -> str:
        if subset[x_col].nunique() < 2 or subset[y_col].nunique() < 2:
            return (
                f"Each point represents one observation for {x_col} and {y_col}. "
                f"Variation is limited in at least one variable, so a linear association is not informative here."
            )
        correlation = float(subset[x_col].corr(subset[y_col]))
        if correlation >= 0.2:
            direction = "positive"
        elif correlation <= -0.2:
            direction = "negative"
        else:
            direction = "minimal"
        return (
            f"Each point represents one observation. The pattern shows a {direction} association "
            f"between {x_col} and {y_col} (r = {correlation:.2f})."
        )

    @classmethod
    def _qq_caption(cls, series: pd.Series, column: str) -> str:
        skewness = float(series.skew())
        return (
            f"This Q-Q plot checks whether {column} follows a normal distribution. "
            f"The observed skewness is {cls._format_statistic(skewness)}, so points close to the reference line "
            f"would support approximate normality while large departures would suggest skew or heavy tails."
        )

    @classmethod
    def _bar_caption(cls, counts: pd.Series, column: str) -> str:
        top_category = str(counts.index[0])
        top_count = int(counts.iloc[0])
        total = int(counts.sum())
        proportion = top_count / total if total else 0.0
        return (
            f"This bar chart shows the most common categories for {column}. "
            f"{top_category} is the most frequent category "
            f"(n={top_count}, {proportion:.1%} of the displayed observations)."
        )

    @staticmethod
    def _fig_to_base64(fig: plt.Figure) -> str:
        """Convert a Matplotlib figure to a base64-encoded PNG."""
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("utf-8")

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
            return {"plots": [], "effect_sizes": {}}

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
                    effect_sizes[f"{col} ({level_a} vs {level_b})"] = round(float(d_val), 4)

        # Histograms / distributions for numeric columns
        for col in numeric_cols:
            series = df_for_plot[col].dropna()
            if series.empty:
                continue
            fig, ax = plt.subplots(figsize=(4, 3))
            ax.hist(series, bins=20, color="#22d3ee", alpha=0.85, edgecolor="#0b1223")
            ax.set_title(f"Distribution: {col}")
            ax.set_xlabel(col)
            ax.set_ylabel("Frequency")
            _add_plot(
                cls._build_plot_payload(
                    title=f"Histogram · {col}",
                    description=f"Value distribution for {col}.",
                    image_base64=cls._fig_to_base64(fig),
                    plot_type="histogram",
                    column=col,
                    caption=cls._histogram_caption(series, col),
                )
            )
            if len(plots) >= limit:
                return {"plots": plots, "effect_sizes": effect_sizes}

        # Box plots (optionally grouped)
        for col in numeric_cols:
            series = df_for_plot[col].dropna()
            if series.empty:
                continue
            fig, ax = plt.subplots(figsize=(4, 3))
            grouped_series: dict[str, pd.Series] | None = None
            if group_column:
                grouped_series = {
                    level: df_for_plot[df_for_plot[group_column].astype(str) == level][col].dropna()
                    for level in group_levels
                }
                grouped = [grouped_series[level] for level in group_levels]
                labels = [f"{group_column}={lvl}" for lvl in group_levels]
                ax.boxplot(
                    grouped,
                    tick_labels=labels,
                    patch_artist=True,
                    boxprops={"facecolor": "#22d3ee", "alpha": 0.65},
                )
                ax.set_title(f"Box: {col} by {group_column}")
            else:
                ax.boxplot(series, vert=True, patch_artist=True, boxprops={"facecolor": "#22d3ee", "alpha": 0.65})
                ax.set_title(f"Box Plot: {col}")
                ax.set_xticklabels([col])
            ax.set_ylabel(col)
            _add_plot(
                cls._build_plot_payload(
                    title=f"Box · {col}",
                    description=f"IQR + outliers for {col}.",
                    image_base64=cls._fig_to_base64(fig),
                    plot_type="box",
                    column=col,
                    caption=cls._box_caption(
                        series=series,
                        column=col,
                        group_column=group_column,
                        grouped_series=grouped_series,
                    ),
                )
            )
            if len(plots) >= limit:
                return {"plots": plots, "effect_sizes": effect_sizes}

        # Scatter plots for leading numeric pairs
        for x_col, y_col in combinations(numeric_cols[:4], 2):
            subset = df_for_plot[[x_col, y_col]].dropna()
            if len(subset) < 10:
                continue
            fig, ax = plt.subplots(figsize=(4, 3))
            ax.scatter(subset[x_col], subset[y_col], alpha=0.7, color="#0ea5e9")
            ax.set_xlabel(x_col)
            ax.set_ylabel(y_col)
            ax.set_title(f"Scatter: {x_col} vs {y_col}")
            _add_plot(
                cls._build_plot_payload(
                    title=f"Scatter · {x_col} vs {y_col}",
                    description=f"Correlation view for {x_col} and {y_col}.",
                    image_base64=cls._fig_to_base64(fig),
                    plot_type="scatter",
                    column=f"{x_col},{y_col}",
                    caption=cls._scatter_caption(subset, x_col, y_col),
                )
            )
            if len(plots) >= limit:
                return {"plots": plots, "effect_sizes": effect_sizes}

        # Q-Q plots to inspect normality
        for col in numeric_cols:
            series = df_for_plot[col].dropna()
            if len(series) < 5:
                continue
            fig, ax = plt.subplots(figsize=(4, 3))
            stats.probplot(series, dist="norm", plot=ax)
            ax.set_title(f"Q-Q Plot: {col}")
            _add_plot(
                cls._build_plot_payload(
                    title=f"Q-Q Plot · {col}",
                    description=f"Normality check for {col} via quantile-quantile plot.",
                    image_base64=cls._fig_to_base64(fig),
                    plot_type="qq",
                    column=col,
                    caption=cls._qq_caption(series, col),
                )
            )
            if len(plots) >= limit:
                return {"plots": plots, "effect_sizes": effect_sizes}

        # Simple categorical count plots
        for col in categorical_cols:
            series = df_for_plot[col].astype(str).fillna("NA")
            top_counts = series.value_counts().head(8)
            if top_counts.empty:
                continue
            fig, ax = plt.subplots(figsize=(4, 3))
            top_counts.sort_values().plot(kind="barh", ax=ax, color="#22d3ee")
            ax.set_title(f"Category Counts: {col}")
            ax.set_xlabel("Count")
            _add_plot(
                cls._build_plot_payload(
                    title=f"Counts · {col}",
                    description=f"Top categories for {col}.",
                    image_base64=cls._fig_to_base64(fig),
                    plot_type="bar",
                    column=col,
                    caption=cls._bar_caption(top_counts, col),
                )
            )
            if len(plots) >= limit:
                return {"plots": plots, "effect_sizes": effect_sizes}

        return {"plots": plots, "effect_sizes": effect_sizes}
