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

    def __str__(self) -> str:
        tests = ', '.join(self.performed_tests)
        return f'Summary:\n{self.summary}\n\nRecommendations:\n{self.recommendations}\n\nPerformed Tests:\n{tests}'


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
      "performed_tests": "<copy this directly from input>"
    }
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
