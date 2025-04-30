# summarizer_agent.py

from langchain_core.messages import AIMessage
from pydantic import BaseModel, Field
from pydantic_ai import Agent
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

    return Agent(
        model=model,
        model_settings=model_settings,
        deps_type=SummariserDeps,
        result_type=SummariserResults,
        name='Summariser Agent',
        system_prompt=system_prompt,
        retries=retries,
    )
