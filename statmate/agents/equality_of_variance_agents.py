from collections.abc import Callable

import numpy as np
from pydantic_ai import Agent
from pydantic_ai.models.openai import ModelSettings, OpenAIModel

from statmate.agents import AgentResult, StatTestDeps, build_stat_test_agent, run_sync_agent
from statmate.statistical_core import StatTestResult, bartlett_test, levene_test


def bartlett_agent(
    model: OpenAIModel,
    model_settings: ModelSettings | None = None,
    test_name: str = 'Bartlett Test',
    test_function: Callable[..., StatTestResult] = bartlett_test,
) -> Agent[StatTestDeps, AgentResult]:
    """Builds a Bartlett equality of variance agent."""
    return build_stat_test_agent(
        model=model,
        model_settings=model_settings,
        test_name=test_name,
        test_function=test_function,
        potential_suggestions='Remember Bartlett’s test is sensitive to non-normality; if your data depart from normality, consider Levene’s or Brown–Forsythe tests.'
        'If group sizes are unequal, prefer Levene’s test for more robust variance comparison.'
        'Visualize group variances with boxplots or residual plots to detect heteroscedasticity patterns.'
        'Apply variance-stabilizing transformations (e.g., log, square-root, or Box–Cox) if variances increase with the mean.'
        'For multiple groups beyond two, use Bartlett’s test extension or consider ANOVA on absolute deviations (Brown–Forsythe).',
    )


def levene_agent(
    model: OpenAIModel,
    model_settings: ModelSettings | None = None,
    test_name: str = 'Levene Test',
    test_function: Callable[..., StatTestResult] = levene_test,
) -> Agent[StatTestDeps, AgentResult]:
    """Builds a Levene variance homogeneity agent."""
    return build_stat_test_agent(
        model=model,
        model_settings=model_settings,
        test_name=test_name,
        test_function=test_function,
        potential_suggestions="Levene’s test uses the median (default) for robustness; use `center='mean'` if you assume symmetric distributions."
        "For skewed data, consider the Brown–Forsythe variant (center='median') for improved robustness."
        'Plot group variances and absolute deviations to see where variances differ visually.'
        'If variances are unequal, consider Welch’s ANOVA or generalized least squares for downstream analyses.'
        'Fligner–Killeen test is another nonparametric option less sensitive to outliers; consider it when Levene’s assumptions fail.',
    )


if __name__ == '__main__':
    # Example usage
    model = OpenAIModel('gpt-4o')
    model_settings = ModelSettings(
        temperature=0.1,
        max_tokens=1500,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
    )
    b_agent = bartlett_agent(model=model, model_settings=model_settings)
    l_agent = levene_agent(model=model, model_settings=model_settings)
    # Generate example data: two groups with different variances
    np.random.seed(42)
    group1 = np.random.normal(loc=0.0, scale=1.0, size=100)  # mean=0, var=1
    group2 = np.random.normal(loc=1.0, scale=1.0, size=100)  # mean=1, var=1
    group3 = np.random.normal(loc=0.0, scale=2.0, size=100)  # mean=0, var=4

    deps1 = StatTestDeps(
        data=group1,
        data_secondary=group2,
        test_params={
            'alpha': 0.05,
        },
    )

    deps2 = StatTestDeps(
        data=group1,
        data_secondary=group3,
        test_params={
            'alpha': 0.05,
        },
    )
    print('--- ---- ---- ---')
    b_res = run_sync_agent(b_agent, '', deps1)
    print('### Bartlett test results for same variance:')
    print(b_res)
    l_res = run_sync_agent(l_agent, '', deps1)
    print('\n### Levene test results for same variance:')
    print(l_res)
    print('--- ---- ---- ---')
    b_res2 = run_sync_agent(b_agent, '', deps2)
    print('### Bartlett test results for differnet variance:')
    print(b_res2)
    l_res2 = run_sync_agent(l_agent, '', deps2)
    print('\n### Levene test results for different variance:')
    print(l_res2)
    print('--- ---- ---- ---')
