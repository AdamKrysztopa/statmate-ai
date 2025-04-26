from collections.abc import Callable

import numpy as np
from pydantic_ai import Agent
from pydantic_ai.models.openai import ModelSettings, OpenAIModel

from statmate.agents import AgentResult, StatTestDeps, build_stat_test_agent, run_sync_agent
from statmate.statistical_core import StatTestResult, pearson_corr, spearman_corr


def pearson_agent(
    model: OpenAIModel,
    model_settings: ModelSettings | None = None,
    test_name: str = 'Pearson Correlation',
    test_function: Callable[..., StatTestResult] = pearson_corr,
) -> Agent[StatTestDeps, AgentResult]:
    """Builds a Pearson correlation agent."""
    return build_stat_test_agent(
        model=model,
        model_settings=model_settings,
        test_name=test_name,
        test_function=test_function,
        potential_suggestions=(
            'Check a scatterplot to verify linearity before relying on Pearson’s r. '
            'Assess and, if needed, remove or winsorize outliers that may unduly influence the correlation. '
            'Confirm both variables are approximately normally distributed or use Fisher’s z-transform for CIs. '
            'If variances differ greatly, consider robust methods or transformations. '
            'For non-linear but monotonic relationships, switch to Spearman’s rho.'
        ),
    )


def spearman_agent(
    model: OpenAIModel,
    model_settings: ModelSettings | None = None,
    test_name: str = 'Spearman Correlation',
    test_function: Callable[..., StatTestResult] = spearman_corr,
) -> Agent[StatTestDeps, AgentResult]:
    """Builds a Spearman rank-correlation agent."""
    return build_stat_test_agent(
        model=model,
        model_settings=model_settings,
        test_name=test_name,
        test_function=test_function,
        potential_suggestions=(
            'Plot ranked values or a scatterplot of ranks to ensure a monotonic trend. '
            'Be mindful of ties—if many ties occur, consider Kendall’s tau as an alternative. '
            'Spearman is more robust to outliers, but extreme values can still distort rho. '
            'You can bootstrap confidence intervals for ρ when sample sizes are small. '
            'If data meet normality and linearity, Pearson’s r may offer more power.'
        ),
    )


if __name__ == '__main__':
    import numpy as np
    from pydantic_ai.models.openai import ModelSettings, OpenAIModel

    from statmate.agents import StatTestDeps, run_sync_agent

    # Reproducible random data
    np.random.seed(42)
    # Series 1: base signal
    series1 = np.random.normal(loc=0.0, scale=1.0, size=100)
    # Series 2: correlated with series1 (Pearson & Spearman)
    series2 = 0.8 * series1 + 0.2 * np.random.normal(loc=0.0, scale=1.0, size=100)
    # Series 3: no pearson correlation, spearman correlation

    # assume series1 already defined
    # 1) build a smooth odd‐power transform + small noise
    raw3 = series1**3 + 0.05 * np.random.normal(size=series1.shape)

    # 2) subtract the best linear fit so Cov(series1, series3)=0
    beta3 = np.dot(series1, raw3) / np.dot(series1, series1)
    resid3 = raw3 - beta3 * series1

    # 3) flip sign to make Spearman positive
    series3 = -resid3

    model = OpenAIModel('gpt-4o')
    model_settings = ModelSettings(
        temperature=0.0,
        max_tokens=500,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
    )

    # Build agents
    p_agent = pearson_agent(model=model, model_settings=model_settings)
    s_agent = spearman_agent(model=model, model_settings=model_settings)

    # # 1 vs 2: expect correlation
    # deps_12 = StatTestDeps(data=series1, data_secondary=series2, test_params={'alpha': 0.05})
    # print('=== Series 1 vs Series 2 (correlated) ===')
    # print('Pearson:', run_sync_agent(p_agent, '', deps_12))
    # print('Spearman:', run_sync_agent(s_agent, '', deps_12))

    # 1 vs 3: expect no correlation
    deps_13 = StatTestDeps(data=series1, data_secondary=series3, test_params={'alpha': 0.05})
    print('\n=== Series 1 vs Series 3 (Pearson uncorrelated) ===')
    print('Pearson:', run_sync_agent(p_agent, '', deps_13))
    print('Spearman:', run_sync_agent(s_agent, '', deps_13))
