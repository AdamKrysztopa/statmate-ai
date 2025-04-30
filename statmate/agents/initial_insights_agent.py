import inspect
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
   • transform_independent_tool(ctx, value_col, group_col) → deps
   • transform_categorical_tool(ctx, row_category, col_category) → deps
   

2. **Column Analysis**  - verify the columns names, and use take them into acount for path selection.
   2.1 List each column with its data type and unique-value count.  
   2.2 Classify each column as ‘continuous’ or ‘categorical’.  
   2.3 Provide encoding or transformation recommendations.  
   2.4 Provide a list of columns to use for the test
       2.4.1 Make sure that not meaningful columns are NOT included in the list.
       2.4.2 Whenever applies, give description why you decide to drop the column.
   2.5 use data_type == CATEGORICAL only for the cases, when test is ither Chi squared of Fisher. CONTINOUS - elswerere
       data_dype is refering to the target data.
   2.6 group_column you deliver, will hold info on anayzed group. It will be stored as pd.Series or pd.DataFrame index
       it is e.g. Age, sex, or anything that recognize analysed objec, but it is not the matter of test.
       group_column cannot be a subset of analysis_columns

3. Data formatting depends on the test type:
   3.1. Transformation requirements:
       • **Independent-group tests** (e.g. Student’s t-test, Welch’s t-test, Mann–Whitney U):  
         Input is often in “long” format. Apply the `transform_independent_tool` tool with  
         `value_col` and `group_col` to pivot into wide form—producing one column per group named `<value_col>_<group>`.
       • **Categorical tests** (e.g. Chi-square, Fisher’s exact):  
         These require a numerical contingency table. Use the `transform_categorical_tool` tool with  
         `row_category` and `column_category` to build a counts DataFrame suitable for 
         `chi2_contingency` or `fisher_exact`.
   3.2. **Table-based tests** (ANOVA, Chi-square, Fisher’s exact) must receive a DataFrame input—specify which columns
   define rows and columns of the table.
   3.3. **Series-based tests** (correlations, paired t-test, Wilcoxon, etc.) take one or two `pd.Series`.
   Propose the appropriate column(s) for `x1` and `x2`, following SciPy’s convention of separate-series inputs.
   3.4. If data are transformed give name of data transformation tool:  'transform_independent', 'transform_categorical'.
   Give 'None' if no data transformation are needed.

4. **Test-Selection Route**  
   4.1 Use exactly the `NodeName` enum values for each decision step.  
   4.2 Emit **each** flowchart node visited, in order, ending with the test node.  
4.3 Format output as JSON matching RouterAgentResults (and **always** include `"tool_arguments": {...}`):
    ```json
    {
        "analysis_columns": ["<col1>", "..."],
        "group_column": "<col_or_null>",
        "output_format": "<pd.Series|pd.DataFrame>",
        "data_analysis_result": "<…>",
        "route_to_test": ["<NodeName>", "...", "<FinalTest>"],
        "comments": "<…>",
        "data_type": "<CONTINUOUS|CATEGORICAL>",
        "data_transformation": "<transform_independent|transform_categorical|None>",
        "tool_arguments": {
                "value_col": "<col_or_null>",
                "group_col": "<col_or_null>",
                "row_category": "<col_or_null>",
                "column_category": "<col_or_null>"
            },
        "data_size": <int>,
        "number_of_columns": <int>
    }
    ```
    Data Validation:
       - When data are transformed - make sure the analysis_columns are after the transformation.
       - Emit a "tool_arguments" object—even if no transform is needed, it must be {}.
       - **Make Sure** the tool_arguments are delivered and exactly correct when data_transformation is not None.
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
    data_transformation: Literal['transform_independent', 'transform_categorical', 'None']
    tool_arguments: dict[str, Any] = Field(
        default_factory=dict, description='Arguments to pass to the chosen data transformation tool'
    )
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
            f'- Data Transformations: {self.data_transformation}\n'
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
        index_name = df.index.name

        return {
            'User Input': ctx.deps.user_input,
            'Data Shape': df.shape,
            'Data Types': dtypes,
            'Cardinalities': cardinalities,
            'Data Description': data_description,
            'Index Name': index_name,
        }

    @agent.tool
    def transform_independent_tool(
        ctx: RunContext[InitialInsightsAgentDeps], value_col: str, group_col: str
    ) -> InitialInsightsAgentDeps:
        """Pivot a long-format DataFrame into wide format for two independent groups.

        Column names become "<value_col>_<group>".

        Args:
            ctx: The context containing dependencies.
            value_col: Name of the numeric column.
            group_col: Name of the binary‐categorical grouping column.

        Returns:
            Wide-format DataFrame with one column per group.
        """
        df = ctx.deps.input_data.copy()
        if isinstance(df, pd.Series):
            df = df.to_frame()
        elif isinstance(df, np.ndarray):
            df = pd.DataFrame(df, columns=ctx.deps.columns_decision)
        ctx.deps.input_data = transform_independent(df, value_col, group_col)
        ctx.deps.columns_decision = list(ctx.deps.input_data.columns)
        return ctx.deps

    @agent.tool
    def transform_categorical_tool(
        ctx: RunContext[InitialInsightsAgentDeps],
        row_category: str,
        column_category: str,
    ) -> InitialInsightsAgentDeps:
        """Build a contingency table from two categorical columns for a Chi-squared test.

        Args:
            ctx: The RunContext holding the current dependencies.
            row_category: Name of the column whose values will form the table’s rows.
            column_category: Name of the column whose values will form the table’s columns.

        Returns:
            The same InitialInsightsAgentDeps, with `input_data` replaced by
            a pandas.DataFrame contingency table (counts) suitable for chi2_contingency.
        """
        df = ctx.deps.input_data.copy()
        if isinstance(df, pd.Series):
            df = df.to_frame()
        elif isinstance(df, np.ndarray):
            df = pd.DataFrame(df, columns=ctx.deps.columns_decision)
        ctx.deps.input_data = transform_categorical(df, row_category, column_category)
        ctx.deps.columns_decision = list(ctx.deps.input_data.columns)
        return ctx.deps

    return agent


def format_data_by_recommendation(
    data: pd.DataFrame,
    rec: InitialInsightsAgentResults,
) -> pd.Series | tuple[pd.Series, pd.Series] | pd.DataFrame:
    """Formats the data based on the agent's recommendation.

    Args:
        data: The input data.
        recommendation: The agent's recommendations.

    Returns:
        Formatted DataFrame.
    """
    if rec.group_column and rec.group_column in data.columns:
        data = data.set_index(rec.group_column, drop=False)

    route = set(rec.route_to_test)

    # 3) Only these tests get a DataFrame back
    df_tests = {NodeName.CHI2, NodeName.FISHER, NodeName.ANOVA_RM}

    if route & df_tests:
        return data[rec.analysis_columns] if set(rec.analysis_columns).issubset(set(data.columns)) else data

    # 4) Otherwise always return two Series
    if set(rec.analysis_columns).issubset(set(data.columns)):
        col1, col2 = rec.analysis_columns[:2]
    else:
        col1, col2 = data.columns[:2]
    s1 = data[col1].dropna()
    s2 = data[col2].dropna()
    return s1, s2


def transform_independent(data: pd.DataFrame, value_col: str, group_col: str) -> pd.DataFrame:
    wide = data.pivot(columns=group_col, values=value_col)
    wide = wide.add_prefix(f'{value_col}_')
    return wide


def transform_categorical(data: pd.DataFrame, row_category: str, column_category: str) -> pd.DataFrame:
    return pd.crosstab(data[row_category], data[column_category])


# Map tool names to actual functions
TOOL_FUNCS = {
    'transform_independent': transform_independent,
    'transform_categorical': transform_categorical,
}


def validate_tool_args(tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
    func = TOOL_FUNCS[tool_name]
    sig = inspect.signature(func)
    expected = set(sig.parameters) - {'data'}
    provided = set(args)
    if not expected.issubset(provided):
        missing = expected - provided
        raise ValueError(f'Bad args for {tool_name}: missing {missing}')
    extra = provided - expected
    if len(extra) > 0:
        print(f'Need to remove the extra columns: {extra}')

    return {arg: arg_val for arg, arg_val in args.items() if arg in expected}


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
        # 'anova': pd.DataFrame(
        #     {
        #         'test_score': np.random.normal(75, 10, 150),
        #         'examiner': np.random.choice(['Examiner 1', 'Examiner 2', 'Examiner 3'], 150),
        #         'teaching_method': np.random.choice(['Method A', 'Method B', 'Method C'], 150),
        #     }
        # ),
        'chi_squared': pd.DataFrame(
            {
                'usr_id': np.arange(200),
                'smoker': np.random.choice(['Yes', 'No'], 200),
                'exercise_level': np.random.choice(['Low', 'Medium', 'High'], 200),
            }
        ),
        'pearson_correlation': pd.DataFrame(
            {
                'hours_studied': np.random.normal(20, 2, 100),
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
        res = results.data
        print(res)
        tool_args = res.tool_arguments or {}

        if res.data_transformation != 'None':
            tool_args = validate_tool_args(res.data_transformation, tool_args)
            df = TOOL_FUNCS[res.data_transformation](df, **tool_args)

        formatted = format_data_by_recommendation(df, res)
        print(formatted)
        print('--- ---- ---- ---- ---\n')
