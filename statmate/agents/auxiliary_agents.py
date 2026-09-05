"""Auxiliary agents for workflow support.

This module contains agents that support the main statistical workflow,
such as study design assessment.
"""

from typing import Any

from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models.openai import Model, ModelSettings

from statmate.core.config import default_config


class AssessDesignDeps(BaseModel):
    """Dependencies for the study design assessment agent.

    Attributes:
        msg: Message from initial insights agent containing analysis information.
    """

    msg: str | dict[str, Any] | list = Field(description='Message from initial insights agent.')


class AssessDesignResults(BaseModel):
    """Results from study design assessment.

    Attributes:
        paired: Whether the data represents paired measurements.
    """

    paired: bool


def get_assess_design_study_agent(
    model: Model,
    model_settings: ModelSettings,
    retries: int | None = None,
) -> Agent[AssessDesignDeps, AssessDesignResults]:
    """Build an agent to assess study design (paired vs independent).

    Args:
        model: The AI model to use.
        model_settings: Model configuration settings.
        retries: Number of retries on failure. If None, uses config default.

    Returns:
        Configured Agent instance.
    """
    if retries is None:
        retries = default_config.model.retries

    system_prompt = """
## Initial Insights:

You are the analytical agent analyzing description received from the Initial Insights Agent. 

You are responsible for task "E" and you must answer either Paired Measurements on Each Subject or Not.
Often - almost always - when the transformation 'transform_independent' is run, data are not paired.
Data are paired for 'Paired t-test' and 'Wilcoxon Signed-Rank test' only - paired = True
If you see potential of 'Two Independent Groups?' data are **NOT Paired!** - paired = False

You must read carefully the message and decide which way you should follow on the graph:

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
        L --> M[Option A: Welch's t-test]
        L --> N[Option B: Mann-Whitney U test]
    ```

As an output, format output as JSON matching AssessDesignResults:
```json
   {
       "paired": bool True or False
   }
```
"""
    return Agent(
        model=model,
        model_settings=model_settings,
        deps_type=AssessDesignDeps,
        end_strategy='early',
        output_type=AssessDesignResults,
        name='Assess Design Results',
        system_prompt=system_prompt,
        retries=retries,
    )
