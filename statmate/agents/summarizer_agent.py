# summarizer_agent.py

from langchain_core.messages import AIMessage
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import Model, ModelSettings

# --- Summariser agent definitions ---


class SummariserDeps(BaseModel):
    results: list[AIMessage] = Field(description='Structured AIMessage results from executed tests.')
    performed_tests: list[str] = Field(description='List of tests explicitly executed.')


class SummariserResults(BaseModel):
    summary: str = Field(description='Consolidated scientific summary of test outcomes.')
    recommendations: str = Field(description='Concise recommendations based on executed tests.')
    performed_tests: list[str] = Field(description='Tests actually performed.')
    power_interpretation: str | None = Field(
        default=None,
        description='Plain-language power note — populated when any p_value > 0.05; null otherwise.',
    )

    def __str__(self) -> str:
        tests = ', '.join(self.performed_tests)
        parts = [
            f'Summary:\n{self.summary}',
            f'Recommendations:\n{self.recommendations}',
            f'Performed Tests:\n{tests}',
        ]
        if self.power_interpretation:
            parts.append(f'Power Interpretation:\n{self.power_interpretation}')
        return '\n\n'.join(parts)


def get_summariser_agent(
    model: Model,
    model_settings: ModelSettings,
    retries: int = 2,
) -> Agent[SummariserDeps, SummariserResults]:
    system_prompt = """
    You are a statistical summarization specialist. You will receive exactly two inputs:
    - `results`: AIMessage objects for each performed test, formatted as JSON.
    - `performed_tests`: Names of tests explicitly performed.

    You have one available tool: `all_inputs_merged`, which merges the results and performed tests into a single string.

    Invoke this tool to get the merged string, but do not use it in your final output.

    STRICTLY adhere to these instructions:
    1. ONLY summarize tests explicitly listed in `performed_tests`.
    2. DO NOT infer or invent additional tests or outcomes.
    3. Explicitly reference test statistics and p-values ONLY from `results`.

    Provide EXACTLY the following JSON keys without additional text or formatting:
    {
      "summary": "<scientific paragraph summarizing ONLY provided tests, clearly mentioning statistics and p-values>",
      "recommendations": "<concise recommendations based on provided tests only>",
      "performed_tests": "<copy this directly from input>",
      "power_interpretation": "<plain-language note or null>"
    }

    When one or more reported p-values exceed the significance threshold (p > 0.05), populate
    `power_interpretation` with a plain-language note that: (a) states explicitly that a
    non-significant result does not confirm the null hypothesis; (b) references the observed effect
    size from the test result fields and explains qualitatively whether it suggests a clinically
    meaningful difference (e.g., "the observed Cohen's d of 0.15 is small, suggesting the groups
    differ by less than one-fifth of a standard deviation"); (c) advises the reader to interpret the
    result in the context of sample size and study power — do not compute power numerically but
    describe the implication of having a small sample relative to the observed effect.
    If ALL results have p <= 0.05, set `power_interpretation` to null.
    """

    agent = Agent(
        model=model,
        model_settings=model_settings,
        deps_type=SummariserDeps,
        result_type=SummariserResults,
        name='Summariser Agent',
        system_prompt=system_prompt,
        retries=retries,
    )

    @agent.tool
    def all_inputs_merged(ctx: RunContext[SummariserDeps]) -> str:
        """Merge all inputs into a single string for the model."""
        results = ctx.deps.results
        results = '\n'.join([result.__str__() for result in results])
        performed_tests = ctx.deps.performed_tests
        return f'Results: {results}, Performed Tests: {performed_tests}'

    return agent
