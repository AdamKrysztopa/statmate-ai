"""ANOVA module for statistical tests."""

from collections.abc import Iterable
from itertools import combinations

import numpy as np
import pandas as pd
import scipy.stats
from scipy.stats import norm, rankdata
from statsmodels.stats.anova import AnovaRM
from statsmodels.stats.multitest import multipletests

from statmate.core.config import default_config
from statmate.statistical_core.base import StatTestResult


def anova_one_way_test(*groups: np.ndarray, alpha: float | None = None) -> StatTestResult:
    if alpha is None:
        alpha = default_config.statistical.default_alpha

    statistic, p_value = scipy.stats.f_oneway(*groups)
    if p_value < alpha:
        result_text = (
            f'We must reject the null hypothesis (p = {p_value:.4f} < α = {alpha}); at least one group mean differs.'
        )
    else:
        result_text = (
            f'We cannot reject the null hypothesis (p = {p_value:.4f} ≥ α = {alpha}); all group means appear equal.'
        )

    return StatTestResult(
        test_name='One-way ANOVA',
        statistics=statistic,
        p_value=p_value,
        null_hypothesis='All groups have equal means.',
        alternative='At least one group mean is different.',
        statistical_test_results=result_text,
        test_specifics={
            'alpha': alpha,
            'sample_sizes': [len(g) for g in groups],
            'number_of_groups': len(groups),
        },
    )


def prepare_groups_from_frame(data: pd.DataFrame, group_column: str, value_column: str) -> tuple[list[np.ndarray], list[str]]:
    """Extract numeric groups and labels from a long-format DataFrame."""
    frame = data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
    if group_column not in frame.columns:
        raise ValueError(f'Grouping column "{group_column}" not found in DataFrame.')
    if value_column not in frame.columns:
        raise ValueError(f'Value column "{value_column}" not found in DataFrame.')

    groups: list[np.ndarray] = []
    labels: list[str] = []
    for label, series in frame.groupby(group_column)[value_column]:
        arr = pd.to_numeric(series, errors='coerce').dropna().to_numpy()
        if arr.size == 0:
            continue
        groups.append(arr)
        labels.append(str(label))

    if len(groups) < 2:
        raise ValueError('At least two non-empty groups are required.')

    return groups, labels


def dunn_posthoc_test(
    groups: Iterable[np.ndarray],
    *,
    labels: list[str] | None = None,
    p_adjust: str = 'holm',
) -> list[dict[str, float | str | bool]]:
    """Compute pairwise Dunn's test with multiple-comparison correction."""
    cleaned: list[np.ndarray] = []
    label_list: list[str] = []
    for idx, group in enumerate(groups):
        arr = np.asarray(group, dtype=float)
        arr = arr[~np.isnan(arr)]
        if arr.size == 0:
            continue
        cleaned.append(arr)
        label_list.append(labels[idx] if labels and idx < len(labels) else f'Group {idx + 1}')

    if len(cleaned) < 2:
        return []

    all_values = np.concatenate(cleaned)
    n_total = len(all_values)
    ranks = rankdata(all_values)

    # Tie correction factor
    _, tie_counts = np.unique(all_values, return_counts=True)
    tie_correction = 1 - (np.sum(tie_counts**3 - tie_counts) / (n_total**3 - n_total)) if n_total > 1 else 1.0
    rank_const = (n_total * (n_total + 1)) / 12.0

    grouped_ranks: list[np.ndarray] = []
    offset = 0
    for arr in cleaned:
        next_offset = offset + len(arr)
        grouped_ranks.append(ranks[offset:next_offset])
        offset = next_offset

    raw_results: list[dict[str, float | str]] = []
    for i, j in combinations(range(len(cleaned)), 2):
        r_i = grouped_ranks[i].sum()
        r_j = grouped_ranks[j].sum()
        n_i = len(cleaned[i])
        n_j = len(cleaned[j])
        se = np.sqrt(rank_const * tie_correction * (1 / n_i + 1 / n_j))
        if se == 0:
            continue
        z_score = (r_i / n_i - r_j / n_j) / se
        p_uncorrected = float(2 * norm.sf(abs(z_score)))
        raw_results.append(
            {
                'group1': label_list[i],
                'group2': label_list[j],
                'z': float(z_score),
                'p_uncorrected': p_uncorrected,
            }
        )

    if not raw_results:
        return []

    adjusted = multipletests([res['p_uncorrected'] for res in raw_results], method=p_adjust.lower())
    for res, reject, p_adj in zip(raw_results, adjusted[0], adjusted[1]):
        res['p_adjusted'] = float(p_adj)
        res['reject'] = bool(reject)

    return raw_results


def kruskal_wallis_test(
    *groups: np.ndarray,
    alpha: float | None = None,
    group_labels: list[str] | None = None,
    perform_dunn: bool = True,
    p_adjust: str = 'holm',
) -> StatTestResult:
    """Run the Kruskal-Wallis H-test with optional Dunn's post-hoc analysis."""
    if alpha is None:
        alpha = default_config.statistical.default_alpha

    cleaned: list[np.ndarray] = []
    labels: list[str] = []
    for idx, group in enumerate(groups):
        arr = np.asarray(group, dtype=float)
        arr = arr[~np.isnan(arr)]
        if arr.size == 0:
            continue
        cleaned.append(arr)
        labels.append(group_labels[idx] if group_labels and idx < len(group_labels) else f'Group {idx + 1}')

    if len(cleaned) < 2:
        raise ValueError('Kruskal-Wallis H-test requires at least two groups.')

    statistic, p_value = scipy.stats.kruskal(*cleaned, nan_policy='omit')

    if p_value < alpha:
        decision = f'We must reject the null hypothesis (p = {p_value:.4f} < α = {alpha}); at least one group differs.'
    else:
        decision = (
            f'We cannot reject the null hypothesis (p = {p_value:.4f} ≥ α = {alpha}); group distributions appear similar.'
        )

    posthoc_results = (
        dunn_posthoc_test(cleaned, labels=labels, p_adjust=p_adjust) if perform_dunn and p_value < alpha else None
    )

    return StatTestResult(
        test_name='Kruskal-Wallis H-test',
        statistics=float(statistic),
        p_value=float(p_value),
        null_hypothesis='All groups come from the same distribution.',
        alternative='At least one group distribution differs.',
        statistical_test_results=decision,
        test_specifics={
            'alpha': alpha,
            'sample_sizes': [len(g) for g in cleaned],
            'group_labels': labels,
            'posthoc': posthoc_results,
            'p_adjust': p_adjust,
        },
    )


def anova_rm_test(
    input_data: pd.DataFrame, dependent_variable: str, subject: str, within: list[str], alpha: float = 0.05
) -> StatTestResult:
    # … validation as before …

    # ensure factors are categorical
    input_data[subject] = input_data[subject].astype('category')
    for col in within:
        input_data[col] = input_data[col].astype('category')

    model = AnovaRM(
        data=input_data,
        depvar=dependent_variable,
        subject=subject,
        within=within,
    ).fit()

    anova_table = model.anova_table
    first = anova_table.iloc[0]
    f_value, p_value = first['F Value'], first['Pr > F']

    if p_value < alpha:
        result_text = (
            f'We must reject the null hypothesis (p = {p_value:.4f} < α = {alpha}); '
            "at least one condition's mean is different."
        )
    else:
        result_text = (
            f'We cannot reject the null hypothesis (p = {p_value:.4f} ≥ α = {alpha}); all condition means appear equal.'
        )

    return StatTestResult(
        test_name='Repeated Measures ANOVA',
        statistics=f_value,
        p_value=p_value,
        null_hypothesis='The means across conditions (within‑subject factors) are equal.',
        alternative="At least one condition's mean is different.",
        statistical_test_results=result_text,
        test_specifics={
            'alpha': alpha,
            'dependent_variable': dependent_variable,
            'subject': subject,
            'within_factors': within,
            'anova_table': anova_table.to_dict(),
        },
    )


def friedman_test(
    data: pd.DataFrame | Iterable[np.ndarray],
    dependent_variable: str | None = None,
    subject: str | None = None,
    within: list[str] | None = None,
    alpha: float | None = None,
) -> StatTestResult:
    """Run the Friedman test for repeated measures with optional post-hoc guidance."""
    if alpha is None:
        alpha = default_config.statistical.default_alpha

    condition_labels: list[str] = []
    arrays: list[np.ndarray] = []
    n_subjects = 0

    if isinstance(data, pd.DataFrame):
        if not dependent_variable or not subject or not within:
            raise ValueError('dependent_variable, subject, and within are required for Friedman test on DataFrames.')
        factor = within[0]
        subset = data[[dependent_variable, subject, factor]].dropna()
        pivot = subset.pivot(index=subject, columns=factor, values=dependent_variable)
        if pivot.shape[1] < 3:
            raise ValueError('Friedman test requires at least three related conditions.')
        arrays = [pivot[col].to_numpy() for col in pivot.columns]
        condition_labels = [str(col) for col in pivot.columns]
        n_subjects = pivot.shape[0]
    else:
        arrays = [np.asarray(group, dtype=float) for group in data]
        arrays = [arr[~np.isnan(arr)] for arr in arrays]
        condition_labels = [f'Condition {i + 1}' for i in range(len(arrays))]
        n_subjects = len(arrays[0]) if arrays else 0

    if len(arrays) < 3:
        raise ValueError('Friedman test requires at least three paired samples.')

    statistic, p_value = scipy.stats.friedmanchisquare(*arrays)
    if p_value < alpha:
        decision = (
            f'We must reject the null hypothesis (p = {p_value:.4f} < α = {alpha}); '
            'at least one condition differs.'
        )
    else:
        decision = (
            f'We cannot reject the null hypothesis (p = {p_value:.4f} ≥ α = {alpha}); '
            'no evidence of differences across conditions.'
        )

    posthoc_guidance = (
        'If significant, follow up with pairwise Wilcoxon signed-rank tests or Nemenyi tests with Holm correction.'
    )

    return StatTestResult(
        test_name='Friedman Test',
        statistics=float(statistic),
        p_value=float(p_value),
        null_hypothesis='All related samples come from the same distribution.',
        alternative='At least one paired condition differs.',
        statistical_test_results=decision,
        test_specifics={
            'alpha': alpha,
            'condition_labels': condition_labels,
            'subjects': n_subjects,
            'posthoc_guidance': posthoc_guidance,
        },
    )


if __name__ == '__main__':
    # Example usage
    print('Running ANOVA tests...')
    print('One-way ANOVA Test:')
    data = np.random.rand(10, 3)  # Example data
    result = anova_one_way_test(data[:, 0], data[:, 1], data[:, 2])
    print(result)

    print('Repeated Measures ANOVA Test:')
    n_subj, n_cond = 10, 3
    subjects = np.repeat(np.arange(n_subj), n_cond)
    conditions = np.tile([f'C{i + 1}' for i in range(n_cond)], n_subj)
    values = np.random.randn(n_subj * n_cond)  # or your real data

    rm_df = pd.DataFrame({'subject': subjects, 'condition': conditions, 'value': values})

    # make factors categorical
    rm_df['subject'] = rm_df['subject'].astype('category')
    rm_df['condition'] = rm_df['condition'].astype('category')
    print(rm_df)
    # now this will work:
    rm_result = anova_rm_test(
        input_data=rm_df,
        dependent_variable='value',
        subject='subject',
        within=['condition'],
    )
    print(rm_result)
