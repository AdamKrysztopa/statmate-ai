"""Categorical comparison agents for chi-square, Fisher, McNemar, and trend tests."""

from collections.abc import Callable

import numpy as np
import pandas as pd
from pydantic_ai import Agent
from pydantic_ai.models.openai import Model, ModelSettings, OpenAIModel

from statmate.agents import AgentResult, StatTestDeps, build_stat_test_agent, run_sync_agent
from statmate.core.config import default_config
from statmate.statistical_core import (
    StatTestResult,
    chi2_test,
    cochran_armitage_trend_test,
    fisher_exact_test,
    mcnemar_test,
)


def chi2_agent(
    model: Model,
    model_settings: ModelSettings | None = None,
    test_name: str = 'Chi-Squared Test',
    test_function: Callable[..., StatTestResult] = chi2_test,
) -> Agent[StatTestDeps, AgentResult]:
    """Builds a Chi-Squared test agent.

    Args:
        model: The AI model to use.
        model_settings: Optional model settings.
        test_name: Name of the statistical test.
        test_function: Function that performs the test.

    Returns:
        Configured Agent instance.
    """
    return build_stat_test_agent(
        model=model,
        model_settings=model_settings,
        test_name=test_name,
        test_function=test_function,
        potential_suggestions='Please suggest the best way to perform the test. '
        'If results are not clear, propose different tests.',
    )


def fisher_exact_agent(
    model: Model,
    model_settings: ModelSettings | None = None,
    test_name: str = 'Fisher Exact Test',
    test_function: Callable[..., StatTestResult] = fisher_exact_test,
) -> Agent[StatTestDeps, AgentResult]:
    """Builds a Fisher Exact test agent.

    Args:
        model: The AI model to use.
        model_settings: Optional model settings.
        test_name: Name of the statistical test.
        test_function: Function that performs the test.

    Returns:
        Configured Agent instance.
    """
    return build_stat_test_agent(
        model=model,
        model_settings=model_settings,
        test_name=test_name,
        test_function=test_function,
        potential_suggestions='Please suggest the best way to perform the test. '
        'If results are not clear, propose different tests.',
    )


def mcnemar_agent(
    model: Model,
    model_settings: ModelSettings | None = None,
    test_name: str = "McNemar's Test",
    test_function: Callable[..., StatTestResult] = mcnemar_test,
) -> Agent[StatTestDeps, AgentResult]:
    """Builds a McNemar test agent for paired categorical data."""
    return build_stat_test_agent(
        model=model,
        model_settings=model_settings,
        test_name=test_name,
        test_function=test_function,
        potential_suggestions='Use when the same subjects are measured twice (pre/post). '
        'Ensure categories are binary and aligned row-wise.',
    )


def cochran_armitage_agent(
    model: Model,
    model_settings: ModelSettings | None = None,
    test_name: str = 'Cochran-Armitage Trend Test',
    test_function: Callable[..., StatTestResult] = cochran_armitage_trend_test,
) -> Agent[StatTestDeps, AgentResult]:
    """Builds a Cochran-Armitage trend test agent for ordered categories."""
    return build_stat_test_agent(
        model=model,
        model_settings=model_settings,
        test_name=test_name,
        test_function=test_function,
        potential_suggestions='Apply when a binary outcome is measured across ordered groups (e.g., dose levels). '
        'Confirm the ordering of the predictor categories.',
    )


if __name__ == '__main__':
    # Example usage
    model = OpenAIModel(default_config.model.model_name)
    model_settings = ModelSettings(
        temperature=0.1,
        max_tokens=1500,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
    )
    chi2_agent_ = chi2_agent(model=model, model_settings=model_settings)
    fisher_exact_agent_ = fisher_exact_agent(model=model)

    # Example data - Categorical groups
    age_groups = ['<18', '18-24', '25-34', '35-44', '45-54', '55-64', '65-74', '75-84', '85-94', '95+']
    beverages = ['Coffee', 'Tea', 'Juice']

    np.random.seed(42)

    # DEPENDENT DATA: age groups prefer different beverages
    preference_profiles = {
        '<18': [0.2, 0.4, 0.4],
        '18-24': [0.4, 0.4, 0.2],
        '25-34': [0.6, 0.3, 0.1],
        '35-44': [0.7, 0.2, 0.1],
        '45-54': [0.7, 0.2, 0.1],
        '55-64': [0.6, 0.3, 0.1],
        '65-74': [0.5, 0.4, 0.1],
        '75-84': [0.4, 0.5, 0.1],
        '85-94': [0.3, 0.6, 0.1],
        '95+': [0.2, 0.7, 0.1],
    }

    # Generate dependent data
    dependent_data = np.array([np.random.multinomial(100, preference_profiles[age_group]) for age_group in age_groups])

    # INDEPENDENT DATA: all age groups have the same distribution
    independent_data = np.tile([33, 33, 34], (len(age_groups), 1))

    # Create DataFrames
    dep_df = pd.DataFrame(dependent_data, index=age_groups, columns=beverages)
    dep_df.index.name = 'Age Group'

    ind_df = pd.DataFrame(independent_data, index=age_groups, columns=beverages)
    ind_df.index.name = 'Age Group'

    # Run the agents
    dep_data_deps = StatTestDeps(
        data=dep_df,
        data_secondary=None,
        test_params={
            'alpha': 0.05,
        },
    )
    user_prompt = (
        'If possible, please describe the data in detail - names, types, and values. '
        'Perform a Chi-Squared test on the dependent data. Suggest the best way to perform the test, if '
        'results are not clear, propose different tests. Build the story around the test, '
        'and explain the results in detail. Analyze the data and provide insights - check the columns and rows '
        'of the table to understand the data better. '
        'Use the data to explain the results and provide a detailed analysis. '
    )
    chi2_result_dep = run_sync_agent(chi2_agent_, user_prompt=user_prompt, deps=dep_data_deps)
    print('\nChi-Squared Test Result (Dependent Data):')
    print(chi2_result_dep)

    ind_data_deps = StatTestDeps(
        data=ind_df,
        data_secondary=None,
        test_params={
            'alpha': 0.05,
        },
    )
    chi2_result_ind = run_sync_agent(chi2_agent_, user_prompt=user_prompt, deps=ind_data_deps)
    print('\nChi-Squared Test Result (Independent Data):')
    print(chi2_result_ind)

    df = pd.DataFrame(
        {
            'usr_id': np.arange(200),
            'smoker': np.random.choice(['Yes', 'No'], 200),
            'exercise_level': np.random.choice(['Low', 'Medium', 'High'], 200),
        }
    )
    deps = StatTestDeps(
        data=pd.crosstab(df['smoker'], df['exercise_level']),
        data_secondary=None,
        test_params={
            'alpha': 0.05,
        },
    )
    chi__ = run_sync_agent(chi2_agent_, user_prompt=user_prompt, deps=deps)
    print('\nChi-Squared Test Result:')
    print(chi__)
