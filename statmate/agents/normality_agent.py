from collections.abc import Callable, Iterable

import numpy as np
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import Model, ModelSettings, OpenAIModel

from statmate.agents import (
    AgentResult,
    StatTestDeps,
    build_generic_system_prompt,
    build_stat_test_agent,
    run_sync_agent,
)
from statmate.statistical_core import (
    StatTestResult,
    anderson_darling_test,
    cramer_von_mises_test,
    dagostino_pearson_test,
    jarque_bera_test,
    ks_normal_test,
    lilliefors_test,
    normality_of_difference,
    shapiro_wilk_test,
)


def shapiro_wilk_agent(
    model: Model,
    model_settings: ModelSettings | None = None,
    test_name: str = 'Shapiro-Wilk Test',
    test_function: Callable[..., StatTestResult] = shapiro_wilk_test,
) -> Agent[StatTestDeps, AgentResult]:
    """Builds a Shapiro-Wilk test agent.

    Args:
        model: the model to use for the agent.
        model_settings: the settings for the model.
        test_name: the name of the test.
        test_function: the function to perform the test.

    Returns:
        Agent: A Shapiro-Wilk test agent.
    """
    return build_stat_test_agent(
        model=model,
        model_settings=model_settings,
        test_name=test_name,
        test_function=test_function,
        potential_suggertions='Please suggest the best way to perform the test, '
        'if results are not clear, propose different tests.',
    )


def anderson_darling_agent(
    model: Model,
    model_settings: ModelSettings | None = None,
    test_name: str = 'Anderson-Darling Test',
    test_function: Callable[..., StatTestResult] = anderson_darling_test,
) -> Agent[StatTestDeps, AgentResult]:
    """Builds an Anderson-Darling test agent.

    Args:
        model: the model to use for the agent.
        model_settings: the settings for the model.
        test_name: the name of the test.
        test_function: the function to perform the test.

    Returns:
            Agent: A Anderson-Darling test agent.
    """
    return build_stat_test_agent(
        model=model,
        model_settings=model_settings,
        test_name=test_name,
        test_function=test_function,
        potential_suggertions='Please suggest the best way to perform the test, '
        'if results are not clear, propose different tests.',
    )


def ks_normal_agent(
    model: Model,
    model_settings: ModelSettings | None = None,
    test_name: str = 'Kolmogorov-Smirnov Test',
    test_function: Callable[..., StatTestResult] = ks_normal_test,
) -> Agent[StatTestDeps, AgentResult]:
    """Builds a Kolmogorov-Smirnov test agent.

    Args:
        model: the model to use for the agent.
        model_settings: the settings for the model.
        test_name: the name of the test.
        test_function: the function to perform the test.

    Returns:
            Agent: A Kolmogorov-Smirnov test agent.
    """
    return build_stat_test_agent(
        model=model,
        model_settings=model_settings,
        test_name=test_name,
        test_function=test_function,
        potential_suggertions='Please suggest the best way to perform the test, '
        'if results are not clear, propose different tests.',
    )


def normality_of_difference_agent(
    model: Model,
    model_settings: ModelSettings | None = None,
    test_name: str = 'Normality of Difference Test',
    test_function: Callable[..., StatTestResult] = normality_of_difference,
) -> Agent[StatTestDeps, AgentResult]:
    """Builds a Normality of Difference test agent.

    Args:
        model: the model to use for the agent.
        model_settings: the settings for the model.
        test_name: the name of the test.
        test_function: the function to perform the test.

    Returns:
            Agent: A Normality of Difference test agent.
    """
    return build_stat_test_agent(
        model=model,
        model_settings=model_settings,
        test_name=test_name,
        test_function=test_function,
        difference_with_standard_normality='Tests if difference between the two samples is drawn from a normal distribution.',
        extra_comments='The statistical test first calculates the difference between the two samples, '
        'and then performs the Shapiro-Wilk test on the resulting differences.',
    )


def dagostino_pearson_agent(
    model: Model,
    model_settings: ModelSettings | None = None,
    test_name: str = 'DAgostino-Pearson Test',
    test_function: Callable[..., StatTestResult] = dagostino_pearson_test,
) -> Agent[StatTestDeps, AgentResult]:
    """Builds a DAgostino-Pearson test agent.

    Args:
        model: the model to use for the agent.
        model_settings: the settings for the model.
        test_name: the name of the test.
        test_function: the function to perform the test.

    Returns:
            Agent: A DAgostino-Pearson test agent.
    """
    return build_stat_test_agent(
        model=model,
        model_settings=model_settings,
        test_name=test_name,
        test_function=test_function,
        potential_suggertions='Please suggest the best way to perform the test, '
        'if results are not clear, propose different tests.',
    )


def jarque_bera_agent(
    model: Model,
    model_settings: ModelSettings | None = None,
    test_name: str = 'Jarque-Bera Test',
    test_function: Callable[..., StatTestResult] = jarque_bera_test,
) -> Agent[StatTestDeps, AgentResult]:
    """Builds a Jarque-Bera test agent.

    Args:
        model: the model to use for the agent.
        model_settings: the settings for the model.
        test_name: the name of the test.
        test_function: the function to perform the test.

    Returns:
            Agent: A Jarque-Bera test agent.
    """
    return build_stat_test_agent(
        model=model,
        model_settings=model_settings,
        test_name=test_name,
        test_function=test_function,
        potential_suggertions='Please suggest the best way to perform the test, '
        'if results are not clear, propose different tests.',
    )


def cramer_von_mises_agent(
    model: Model,
    model_settings: ModelSettings | None = None,
    test_name: str = 'Cramer-von Mises Test',
    test_function: Callable[..., StatTestResult] = cramer_von_mises_test,
) -> Agent[StatTestDeps, AgentResult]:
    """Builds a Cramer-von Mises test agent.

    Args:
        model: the model to use for the agent.
        model_settings: the settings for the model.
        test_name: the name of the test.
        test_function: the function to perform the test.

    Returns:
            Agent: A Cramer-von Mises test agent.
    """
    return build_stat_test_agent(
        model=model,
        model_settings=model_settings,
        test_name=test_name,
        test_function=test_function,
        potential_suggertions='Please suggest the best way to perform the test, '
        'if results are not clear, propose different tests.',
    )


def lilliefors_agent(
    model: Model,
    model_settings: ModelSettings | None = None,
    test_name: str = 'Lilliefors Test',
    test_function: Callable[..., StatTestResult] = lilliefors_test,
) -> Agent[StatTestDeps, AgentResult]:
    """Builds a Lilliefors test agent.

    Args:
        model: the model to use for the agent.
        model_settings: the settings for the model.
        test_name: the name of the test.
        test_function: the function to perform the test.

    Returns:
            Agent: A Lilliefors test agent.
    """
    return build_stat_test_agent(
        model=model,
        model_settings=model_settings,
        test_name=test_name,
        test_function=test_function,
        potential_suggertions='Please suggest the best way to perform the test, '
        'if results are not clear, propose different tests.',
    )


def meta_normality_agent(
    model: Model,
    model_settings: ModelSettings | None = None,
) -> Agent[StatTestDeps, AgentResult]:
    """Builds a meta-analysis normality agent that combines results from multiple normality tests.

    Args:
        model: the model to use for the agent.
        model_settings: the settings for the model.

    Returns:
        Agent: A meta-analysis normality agent.
    """
    # Structured prompt with pros/cons and weights
    system_prompt = build_generic_system_prompt(
        overview=(
            'Run all normality-test agents, extract their '
            'result, p_value, and comments; then perform '
            'a weighted meta-analysis based on each test’s empirical power.'
        ),
        weights=(
            'Shapiro-Wilk=0.35, Anderson-Darling=0.20, Lilliefors=0.15, '
            'Jarque-Bera=0.10, Cramer-von Mises=0.10, Kolmogorov-Smirnov=0.10, '
        ),
        pros_cons=(
            'Shapiro-Wilk: highest overall power; Anderson-Darling: tail-sensitive; '
            'Lilliefors: parameter-estimation correction; Jarque-Bera: moment-based; '
            'Cramer-von Mises: balanced EDF; K-S: max-distance; D’Agostino-Pearson: omnibus skew/kurtosis.'
        ),
        recommendations=(
            'If weighted evidence conflicts, recommend graphical Q-Q plots '
            'and consider bootstrap or transformation methods.'
        ),
    )

    agent = Agent(
        model=model,
        model_settings=model_settings,
        deps_type=StatTestDeps,
        result_type=AgentResult,
        name='Meta-Analysis Normality Agent',
        system_prompt=system_prompt,
        result_tool_description='Aggregates AgentResults with weighted meta-analysis.',
        retries=3,
    )

    @agent.tool
    def run_meta(ctx: RunContext[StatTestDeps]) -> tuple[StatTestResult, str]:
        deps = ctx.deps
        # Initialize agents
        agents = [
            (shapiro_wilk_agent(model, model_settings=model_settings), 0.35),
            (anderson_darling_agent(model, model_settings=model_settings), 0.20),
            (lilliefors_agent(model, model_settings=model_settings), 0.15),
            (jarque_bera_agent(model, model_settings=model_settings), 0.10),
            (cramer_von_mises_agent(model, model_settings=model_settings), 0.10),
            (ks_normal_agent(model, model_settings=model_settings), 0.10),
        ]
        # Run each agent and collect results
        records = []
        weighted_score = 0.0
        for agent_fn, weight in agents:
            print(f'Running {agent_fn.name}...')
            result = agent_fn.run_sync(user_prompt='Answer POTENTAILLY_NORMAL or NOT_NORMAL', deps=deps).data
            alpha = deps.test_params['alpha'] if deps.test_params else 0.05
            rejected = (
                result.statistical_test_result.p_value < alpha
                if isinstance(result.statistical_test_result.p_value, float)
                else all(p < alpha for p in result.statistical_test_result.p_value)  # type: ignore according to the type of p_value
                # must be either float or list of floats, so this error does not happen
            )
            p_val = result.statistical_test_result.p_value
            weighted_score += weight * (1.0 if rejected else 0.0)
            records.append(
                {
                    'name': result.statistical_test_result.test_name,
                    'p_value': p_val,
                    'rejected': rejected,
                    'comments': result.comments,
                }
            )

        # Meta‑decision threshold at 0.5
        overall_decision = 'non-normal' if weighted_score > 0.5 else 'normal'
        summary = f'Weighted rejection score = {weighted_score:.2f}. Overall conclusion: data are {overall_decision}.'
        # Combine comments: highlight strongest signals
        commentary = '\n'.join(
            f'- {rec["name"]}: {"reject" if rec["rejected"] else "fail to reject"} '
            f'(p={rec["p_value"]}); {rec["comments"]}'
            for rec in records
        )
        final_comments = (
            f'### Summary of Individual Tests ###\n{commentary}\n\n### Meta-Analysis Conclusion ###\n{summary}'
        )
        statisics = []
        for rec in records:
            if isinstance(rec['p_value'], np.ndarray | list | Iterable):
                statisics += list(rec['p_value'])
            else:
                statisics.append(rec['p_value'])

        return StatTestResult(
            test_name='Meta-Analysis Normality Test',
            statistics=statisics,
            p_value=weighted_score,
            null_hypothesis='Data are normally distributed.',
            alternative='Data deviate from normal distribution.',
            statistical_test_results=summary,
            test_specifics={
                'weights': {ag.name: w for ag, w in agents},  # optional: agent names
                'individual_results': records,
            },
        ), final_comments

    return agent


if __name__ == '__main__':
    # Example usage
    model = OpenAIModel('gpt-4o')
    model_settings = ModelSettings(
        temperature=0.0,
        max_tokens=1000,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
    )

    agent = shapiro_wilk_agent(model=model, model_settings=model_settings)

    data_1 = np.random.normal(0, 1, 100)
    data_2 = np.random.uniform(0, 1, 10)
    data_3 = np.random.normal(1, 1, 100)

    test_params = {'alpha': 0.05}

    deps = StatTestDeps(data=data_2, test_params=test_params)
    user_prompt = """
Please let me know the result of the test. You must be sure about the result and you cannot say that the test 
is inconclusive. If you are not sure, please suggest the best way to perform the test.',
Be very concuise about the number of samples.
    """
    result = run_sync_agent(agent, user_prompt=user_prompt, deps=deps)
    print(result)

    agent_2 = normality_of_difference_agent(model=model, model_settings=model_settings)

    result_2 = agent_2.run_sync(
        user_prompt=user_prompt,
        deps=StatTestDeps(data=data_1, data_secondary=data_3, test_params=test_params),
    )
    print(result_2.data)
