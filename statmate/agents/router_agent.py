import json
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import Model, ModelSettings, OpenAIModel


class RouterAgentDeps(BaseModel):
    """Dependencies for the Router Agent.

    Attributes:
        user_input: The user's input or query.
        input_data: The primary data for analysis.
        columns_decision: Optional list of column names if input_data lacks them.
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        json_encoders={
            np.ndarray: lambda v: v.tolist(),
            pd.DataFrame: lambda v: v.to_dict(orient='records'),
            pd.Series: lambda v: v.to_dict(),
        },
    )

    user_input: str
    input_data: pd.DataFrame | np.ndarray | pd.Series
    columns_decision: list[str] | None = None


class RouterAgentResults(BaseModel):
    """Results produced by the Router Agent.

    Attributes:
        columns_to_use: Columns selected for the test.
        output_format: Desired output format ('pd.Series' or 'pd.DataFrame').
        data_analysis_result: Summary of the data analysis.
        proposed_tests: List of suggested tests in order of preference.
        comments: Additional comments or notes.
    """

    columns_to_use: list[str] = Field(
        default_factory=list,
        description='List of columns to use for the test. If empty, all columns are used.',
    )
    output_format: Literal['pd.Series', 'pd.DataFrame'] = Field(
        description='Output format of the test.',
    )
    data_analysis_result: str
    route_to_test: list[str] = Field(
        min_length=1,
        description=(
            'Ordered list of decision steps from the flowchart, ending in the chosen test. '
            'Each element should be a string like “Paired → No → Independent‐samples t‐test”.'
        ),
    )
    comments: str

    def __str__(self) -> str:
        route = ' →\n  '.join(self.route_to_test)
        return f'### Data Analysis Result: {self.data_analysis_result} ###\nRoute to Test:\n  {route}\n'


def build_router_agent(
    model: Model,
    *,
    system_prompt: str,
    model_settings: ModelSettings | None = None,
    retries: int = 3,
) -> Agent[RouterAgentDeps, RouterAgentResults]:
    """Constructs the Router Agent with specified configurations.

    Args:
        model: The language model to use.
        system_prompt: Instructions guiding the agent's behavior.
        model_settings: Optional settings for the model.
        retries: Number of retries allowed for the agent.

    Returns:
        Configured Agent instance.
    """
    agent = Agent(
        model=model,
        model_settings=model_settings,
        deps_type=RouterAgentDeps,
        result_type=RouterAgentResults,
        name='Statistical Test Router Agent',
        system_prompt=system_prompt,
        retries=retries,
    )

    @agent.tool
    async def analyze_data(ctx: RunContext[RouterAgentDeps]) -> str:
        """Analyzes the input data to extract types, cardinalities, and descriptive statistics.

        Args:
            ctx: The context containing dependencies.

        Returns:
            A formatted string summarizing the data analysis.
        """
        data = ctx.deps.input_data

        if isinstance(data, np.ndarray):
            if not ctx.deps.columns_decision:
                raise ValueError('columns_decision is required for ndarray inputs')
            cols = ctx.deps.columns_decision
            df = pd.DataFrame(data, columns=cols)
        elif isinstance(data, pd.Series):
            df = data.to_frame()
        elif isinstance(data, pd.DataFrame):
            df = data.copy()
        else:
            raise TypeError('Unsupported data type for input_data')

        dtypes = {col: str(dtype) for col, dtype in df.dtypes.items()}
        cardinalities = {col: int(df[col].nunique()) for col in df.columns}
        data_description = df.describe(include='all').to_dict()

        analysis_summary = {
            'User Input': ctx.deps.user_input,
            'Data Shape': df.shape,
            'Data Types': dtypes,
            'Cardinalities': cardinalities,
            'Data Description': data_description,
        }

        return json.dumps(analysis_summary, indent=2)

    return agent


def format_data_by_recommendation(
    data: pd.DataFrame,
    recommendation: RouterAgentResults,
) -> pd.DataFrame:
    """Formats the data based on the agent's recommendation.

    Args:
        data: The input data.
        recommendation: The agent's recommendations.

    Returns:
        Formatted DataFrame.
    """
    if recommendation.output_format == 'pd.Series':
        return data[recommendation.columns_to_use].squeeze()
    if recommendation.output_format == 'pd.DataFrame':
        return data[recommendation.columns_to_use] if len(recommendation.columns_to_use) > 0 else data
    raise ValueError(f'Unsupported output format: {recommendation.output_format}')


if __name__ == '__main__':
    # Example usage
    model = OpenAIModel('gpt-4o')
    model_settings = ModelSettings(
        temperature=0.0,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
    )
    system_prompt = (
        'You are a statistical test routing agent. Given a pandas DataFrame description and a user request, follow these steps:\n\n'
        '1. **Column Analysis**:\n'
        '   - List each column with its data type and unique value count.\n'
        "   - Classify each column as 'continuous' or 'categorical':\n"
        "     - 'categorical' if the column is of type object, bool, or numeric with 5 or fewer unique values.\n"
        "     - 'continuous' for other numeric types.\n\n"
        '     - Provide a recommendation for data operations - like arithmetic operations, encoding, etc.\n'
        '2. **Test Selection (provide a route)**:'
        '   - Instead of naming a single test, output *each* decision as a step in the flowchart, culminating in the final test.'
        '    - Format the route as an ordered list of strings, e.g.:'
        '     1. “Start → Continuous outcome”      2. “Two independent groups”'
        '     3. “Assumptions met (normality, equal variance) → Yes”'
        '     4. “Independent‐samples t‐test”'
        '    - Use every decision node (normality check, paired vs independent, etc.) as its own step.'
        '3. **Decision Flowchart**:\n'
        '   - Analyze the potential test paths using the following flowchart:\n'
        '     ```mermaid\n'
        '     flowchart TD\n'
        '         A[Start: What is your analysis objective?] --> B{Outcome Type?}\n'
        '         B -- Continuous Outcome --> C[Assess Study Design]\n'
        '         B -- Categorical Outcome --> D[Choose Chi-square test or Fisher exact test if expected counts are low]\n'
        '         C --> E{Paired Measurements on Each Subject?}\n'
        '         E -- Yes --> F{Parametric assumptions hold?}\n'
        '         F -- Yes --> G[Paired t-test]\n'
        '         F -- No --> H[Wilcoxon Signed-Rank test]\n'
        '         E -- No --> I{Two Independent Groups?}\n'
        '         I -- Yes --> J{Are assumptions met? (normality & equal variances)}\n'
        '         J -- Yes --> K[Independent-samples t-test (Student’s t-test)]\n'
        '         J -- No --> L[Consider Both Options:]\n'
        '         L --> M[Option A: Welch’s t-test (parametric adjustment)]\n'
        '         L --> N[Option B: Mann-Whitney U test (non-parametric)]\n'
        '         C --> S{Assess relationship/association?}\n'
        '         S -- Yes --> T{Data characteristics?}\n'
        '         T -- Parametric continuous --> U[Pearson correlation]\n'
        '         T -- Ordinal/non-normal --> V[Spearman correlation]\n'
        '         S -- No --> W[Consider regression: Linear Regression Analysis]\n'
        '     ```\n\n'
        '4. **Output Format**:\n'
        '   - Provide a JSON object with the following structure:\n'
        '     ```json\n'
        '     {\n'
        '       "columns_to_use": ["column1", "column2"],'
        '       "route_to_test": ['
        '       "Start → Continuous outcome",'
        '       "Independent groups → No pairing",'
        '       "Assumptions met → Yes",'
        '       "Independent‐samples t‐test"'
        '       ],'
        '     }\n'
        '     ```\n\n'
        '5. **Handling Ambiguity**:\n'
        '   - If the user request is ambiguous, ask for clarification by listing specific questions.\n'
        '   - If the meaning of column names is unclear, ask the user to clarify.\n'
        '   - If the data is not suitable for the requested test, suggest alternative tests.\n'
        '   - Avoid to propse singular tests - aways propse the route to the test.\n\n'
        '6. **User Interaction**:\n'
        '   - Ask the user for any additional information needed to perform the analysis.\n'
        '   - Keep the questions simple and concise.\n'
        '   - Avoid asking for too much information at once.\n\n'
    )

    agent = build_router_agent(model=model, system_prompt=system_prompt)

    # 4. Prepare a dict of example datasets covering different test types

    import numpy as np
    import pandas as pd

    data_cases: dict[str, pd.DataFrame] = {
        't_test': pd.DataFrame(
            {'height': np.random.normal(170, 10, 100), 'gender': np.random.choice(['Male', 'Female'], 100)}
        ),
        'anova': pd.DataFrame(
            {
                'test_score': np.random.normal(75, 10, 150),
                'examiner': np.random.choice(['Examiner 1', 'Examiner 2', 'Examiner 3'], 150),
                'teaching_method': np.random.choice(['Method A', 'Method B', 'Method C'], 150),
            }
        ),
        'chi_squared': pd.DataFrame(
            {
                'usr_id': np.arange(200),
                'smoker': np.random.choice(['Yes', 'No'], 200),
                'exercise_level': np.random.choice(['Low', 'Medium', 'High'], 200),
            }
        ),
        'pearson_correlation': pd.DataFrame(
            {
                'hours_studied': np.random.normal(5, 2, 100),
                'exam_score': np.random.normal(70, 15, 100),
                'student_id': np.arange(100),
            }
        ),
        'paired_t_test': pd.DataFrame(
            {'before_treatment': np.random.normal(50, 5, 80), 'after_treatment': np.random.normal(55, 5, 80)}
        ),
    }

    # 5. Run the agent on each case
    for case_name, df in data_cases.items():
        print(f'--- Case: {case_name} ---\n')
        results = agent.run_sync(
            user_prompt='Analyze the data and suggest the appropriate test for the given scenario.',
            deps=RouterAgentDeps(
                user_input='Perform a statistical test for the provided data.',
                input_data=df,
                columns_decision=None,
            ),
        )
        print(results.data)
        formatted_data = format_data_by_recommendation(df, results.data)
        print(f'Recommended Columns: {results.data.columns_to_use}, where input_data: {df.columns.tolist()}')
        print(f'Output Format: {results.data.output_format}')
        print(f'Recomended Tests: {results.data.route_to_test}')
        print(f'Formatted Data:\n{formatted_data}\n')
        print('--- ---- ---- ---- ---\n')
