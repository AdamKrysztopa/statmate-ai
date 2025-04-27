"""Cathegorical Comparison Agents.

Namely, Chi-Squared Test and Fisher's Exact Test.
"""

from typing import Any

from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models.openai import Model, ModelSettings


class AssesDesignDeps(BaseModel):
    msg: str | dict[str, Any] | list = Field(description='Message from initial inisghts agetn')


class AssesDesignResults(BaseModel):
    paired: bool


def get_assess_design_study_agent(
    model: Model,
    model_settings: ModelSettings,
    retires: int = 3,
) -> Agent[AssesDesignDeps, AssesDesignResults]:
    system_prompt = """
## Initial Insights:

You are the analitical agnet analyzing desciption recived from the Initial Insights Agent. You must read carefully the
message and decide which whay you should follow on the graph:

    ```mermaid
        flowchart TD
        A[Start: What is your analysis objective?] --> B{Outcome Type?}
        B -- Continuous --> C[Assess Study Design]
        C --> E{Paired Measurements on Each Subject?}
        E -- Yes --> F{Parametric assumptions hold?}
        F -- Yes --> G[Paired t-test]
        F -- No --> H[Wilcoxon Signed-Rank test]
        E -- No --> I{Two Independent Groups?}
        I -- Yes --> J[Are assumptions met? (normality & equal variances)]
        J -- Yes --> K[Independent-samples t-test]
        J -- No --> L[Consider Both Options:]
        L --> M[Option A: Welch’s t-test]
        L --> N[Option B: Mann-Whitney U test]
    ```

You are responsible for task "E" and you must answer either Paired Measurements on Each Subject or Not.

as an output, Format output as JSON matching AssesDesignResults:
```json
   {
       "paired": bool True or False
   }
```
"""
    return Agent(
        model=model,
        model_settings=model_settings,
        deps_type=AssesDesignDeps,
        result_type=AssesDesignResults,
        name='Assess Design Results',
        system_prompt=system_prompt,
        retries=retires,
    )
