from enum import Enum
from typing import Any, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import Model, ModelSettings, OpenAIModel

INITIAL_INSIGHTS_PROMPT = """
    You are a deterministic statistical-test routing agent.  
Follow these numbered instructions exactly—do not add or omit steps:

1. **Available Tools**  
   • analyze_data(ctx) → dict[str, Any]  
   • calculate_columns_diff(ctx, cols_to_diff: list[str]) → RouterAgentDeps  

2. **Column Analysis**  - verify the columns names, and use take them into acount for path selection.
   2.1 List each column with its data type and unique-value count.  
   2.2 Classify each column as ‘continuous’ or ‘categorical’.  
   2.3 Provide encoding or transformation recommendations.  
        **Do NOT** calculate any "diff" column here.
   2.4 Provide a list of columns to use for the test
       2.4.1 Make sure that not meaningful columns are NOT included in the list.
       2.4.2 Whenever applies, give description why you decide to drop the column.
   2.5 use data_type == CATEGORICAL only for the cases, when test is ither Chi squared of Fisher. CONTINOUS - elswerere
       data_dype is refering to the target data.
   2.6 group_column you deliver, will hold info on anayzed group. It will be stored as pd.Series or pd.DataFrame index
       it is e.g. Age, sex, or anything that recognize analysed objec, but it is not the matter of test.
       group_column cannot be a subset of analysis_columns

3. **Column-Difference Calculation**  
   - Compute the diff `col1−col2` if potentiall test requies it - paired t-test or Wilcoxon Signed-Rank test.
   - If the input data is a DataFrame, add a new column named `diff` to the input data.
   - To compute, call the `calculate_columns_diff` tool; do not infer or invent differences.
   - For significanlty different mean values (like more than few sigma) skip the diff - and look for correlation.

4. **Test-Selection Route**  
   4.1 Use exactly the `NodeName` enum values for each decision step.  
   4.2 Emit **each** flowchart node visited, in order, ending with the test node.  
   4.3 Format output as JSON matching RouterAgentResults:
   ```json
   {
       "analysis_columns": ["col1", "col2"] or ["diff"]
       "group_column": "sex", "used_method"
       "output_format": "pd.Series",
       "data_analysis_result": "Data analysis summary",
       "route_to_test": ["NodeName1", "NodeName2", "FinalTest"],
       "comments": "Any additional comments"
       "data_type": 'CATEGORICAL' or 'CONTINUOUS' based on column analysis.
       "data_size": Total number of rows in the dataset.
       "number_of_columns": Count of features (columns) in the dataset.
    }
    ```
    4.4. Use following decision tree to select the test:
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
5. **Handling Ambiguity**
    - If a required detail is missing or the user’s request conflicts with these rules, ask a concise clarifying question rather than guessing.

6. **Revisit**
    - come back to step 2 only, and decide which columns are necessary for a given path.
    """


class NodeName(str, Enum):
    START = 'Start'
    OUTCOME_TYPE = 'Outcome Type'
    ASSESS_STUDY_DESIGN = 'Assess Study Design'
    PAIRED_MEASUREMENTS = 'Paired Measurements?'
    PARAMETRIC_ASSUMPTIONS = 'Parametric assumptions hold?'
    TWO_INDEPENDENT_GROUPS = 'Two Independent Groups?'
    CONSIDER_BOTH_OPTIONS = 'Consider Both Options'
    ASSESS_ASSOCIATION = 'Assess association?'
    DATA_CHARACTERISTICS = 'Data characteristics?'
    # Test nodes
    SHAPIRO = 'Shapiro-Wilk Test'
    LEVENE = "Levene's Test"
    PAIRED_T = 'Paired t-test'
    WILCOXON = 'Wilcoxon Signed-Rank test'
    INDEP_T = 'Independent t-test'
    WELCH = 'Welch’s t-test'
    MANN = 'Mann-Whitney U'
    PEARSON = 'Pearson correlation'
    SPEARMAN = 'Spearman correlation'
    CHI2 = 'Chi-square test'
    FISHER = 'Fisher exact test'
    ANOVA_RM = 'ANOVA repeated measures'


class InitialInsightsAgentDeps(BaseModel):
    """Dependencies for the Initial Insights Agent.

    Attributes:
        user_input: user insights to the data or analysis.
        input_data: input data for the analysis as pd.Series, pd.DataFrame, or np.ndarray
        columns_decsion: colums for making furhter decisions, if np.ndarray provided - not empty.
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


class InitialInsightsAgentResults(BaseModel):
    """Results produced by the Router Agent.

    Attributes:
        analysis_columns: Columns selected for the test.
        output_format: Desired output format ('pd.Series' or 'pd.DataFrame').
        data_analysis_result: Summary of the data analysis.
        route_to_test: List of suggested tests in order of preference.
        comments: Additional comments or notes.
        data_type: 'CATEGORICAL' or 'CONTINUOUS' targrt column analysis.
        data_size: Total number of rows in the dataset.
        number_of_columns: Count of features (columns) in the dataset.
    """

    analysis_columns: list[str] = Field(
        default_factory=list,
        description='List of max columns to use for the test. If empty, all columns are used. For pd.Series single column.',
    )
    group_column: str | None = Field(description='Column name for the subject of thest info, but not the results.')
    output_format: Literal['pd.Series', 'pd.DataFrame'] = Field(
        description='Output format of the test.',
    )
    data_analysis_result: str
    route_to_test: list[NodeName] = Field(
        min_length=1,
        description=(
            'Ordered list of decision steps from the flowchart, ending in the chosen test. '
            'each element should be an NodeName element.'
        ),
    )
    comments: str

    data_type: Literal['CATEGORICAL', 'CONTINUOUS']
    data_size: int
    number_of_columns: int

    def __str__(self) -> str:
        """String representation of RouterAgentResults for easy readability."""
        route = ' →\n  '.join(self.route_to_test)
        return (
            f'### Data Analysis Result ###\n'
            f'{self.data_analysis_result}\n\n'
            f'### Route to Test ###\n'
            f'  {route}\n\n'
            f'### Additional Information ###\n'
            f'- Data Type: {self.data_type}\n'
            f'- Data Size: {self.data_size}\n'
            f'- Number of Columns: {self.number_of_columns}\n'
            f'- Columns to Use: {", ".join(self.analysis_columns) if self.analysis_columns else "All Columns"}\n'
            f'- Group Column: {self.group_column} - the index column\n'
            f'- Output Format: {self.output_format}\n'
            f'- Comments: {self.comments}\n'
        )


def build_initial_insights_agent(
    model: Model,
    *,
    system_prompt: str,
    model_settings: ModelSettings | None = None,
    retries: int = 3,
) -> Agent[InitialInsightsAgentDeps, InitialInsightsAgentResults]:
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
        deps_type=InitialInsightsAgentDeps,
        result_type=InitialInsightsAgentResults,
        name='Initial Insights Agent',
        system_prompt=system_prompt,
        retries=retries,
    )

    @agent.tool
    async def analyze_data(ctx: RunContext[InitialInsightsAgentDeps]) -> dict[str, Any]:
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
            ctx.deps.columns_decision = [str(data.name)] if data.name else []
        elif isinstance(data, pd.DataFrame):
            df = data.copy()
            ctx.deps.columns_decision = [str(col) for col in df.columns]
        else:
            raise TypeError('Unsupported data type for input_data')

        dtypes = {col: str(dtype) for col, dtype in df.dtypes.items()}
        cardinalities = {col: int(df[col].nunique()) for col in df.columns}
        data_description = df.describe(include='all').to_dict()

        return {
            'User Input': ctx.deps.user_input,
            'Data Shape': df.shape,
            'Data Types': dtypes,
            'Cardinalities': cardinalities,
            'Data Description': data_description,
        }

    @agent.tool
    async def calculate_columns_diff(
        ctx: RunContext[InitialInsightsAgentDeps], cols_to_diff: list[str]
    ) -> InitialInsightsAgentDeps:
        """Calculates the difference between two specified columns and updates the deps.

        Args:
            ctx: The agent run context containing input_data and columns_decision.
            cols_to_diff: Exactly two column names to diff.

        Returns:
            RouterAgentDeps: the same deps object, with 'diff' column added and columns_decision updated.
        """
        if len(cols_to_diff) == 2:
            col1, col2 = cols_to_diff
            if isinstance(ctx.deps.input_data, pd.DataFrame):
                if col1 in ctx.deps.input_data.columns and col2 in ctx.deps.input_data.columns:
                    if ctx.deps.input_data[col1].dtype != ctx.deps.input_data[col2].dtype:
                        ctx.deps.user_input += '\nBoth columns must have the same data type to calculate differences'
                        return ctx.deps
                    ctx.deps.input_data['diff'] = ctx.deps.input_data[col1] - ctx.deps.input_data[col2]
                    if ctx.deps.columns_decision is not None:
                        ctx.deps.columns_decision.append('diff')
                    ctx.deps.user_input += f'\nDifference between {col1} and {col2} calculated is now available in the input_data DataFrame in the column "diff"'
                    return ctx.deps
                ctx.deps.user_input += '\nBoth columns must exist in input_data'
            ctx.deps.user_input += '\ninput_data must be a pandas DataFrame to calculate column differences'
        ctx.deps.user_input += '\ncolumns_decision must contain exactly two column names'
        return ctx.deps

    return agent


def format_data_by_recommendation(
    data: pd.DataFrame,
    recommendation: InitialInsightsAgentResults,
) -> pd.DataFrame:
    """Formats the data based on the agent's recommendation.

    Args:
        data: The input data.
        recommendation: The agent's recommendations.

    Returns:
        Formatted DataFrame.
    """
    if recommendation.group_column is not None and recommendation.group_column != 'None':
        data = data.set_index(recommendation.group_column)
    if recommendation.output_format == 'pd.Series':
        return data[recommendation.analysis_columns].squeeze()
    if recommendation.output_format == 'pd.DataFrame':
        return data[recommendation.analysis_columns] if len(recommendation.analysis_columns) > 0 else data
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
    system_prompt = INITIAL_INSIGHTS_PROMPT

    agent = build_initial_insights_agent(model=model, system_prompt=system_prompt)

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
            deps=InitialInsightsAgentDeps(
                user_input='Perform a statistical test for the provided data.',
                input_data=df,
                columns_decision=None,
            ),
        )
        print(results.data)
        formatted_data = format_data_by_recommendation(df, results.data)
        print(f'Formatted Data:\n{formatted_data}\n')
        print('--- ---- ---- ---- ---\n')
