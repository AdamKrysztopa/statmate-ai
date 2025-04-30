import logging
from collections.abc import Callable
from typing import Annotated, Literal

import numpy as np
import pandas as pd
from langchain_core.messages import AIMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from pydantic_ai.models.openai import ModelSettings, OpenAIModel
from typing_extensions import TypedDict

from statmate.agents import (
    INITIAL_INSIGHTS_PROMPT,
    build_initial_insights_agent,
    chi2_agent,
    fisher_exact_agent,
    levene_agent,
    mannwhitneyu_agent,
    normality_of_difference_agent,
    shapiro_wilk_agent,
    ttest_ind_agent,
    ttest_rel_agent,
    welch_t_agent,
    wilcoxon_agent,
)
from statmate.agents.agent_builder import StatTestDeps, run_sync_agent
from statmate.agents.auxilary_agents import AssesDesignDeps, get_assess_design_study_agent
from statmate.agents.initial_insights_agent import (
    TOOL_FUNCS,
    InitialInsightsAgentDeps,
    NodeName,
    format_data_by_recommendation,
    validate_tool_args,
)
from statmate.agents.summarizer_agent import SummariserDeps, get_summariser_agent

# ----------------------------------------------------------------------------
# Append strict tool_arguments requirement to prompt
# ----------------------------------------------------------------------------
INITIAL_INSIGHTS_PROMPT = (
    INITIAL_INSIGHTS_PROMPT
    + "\nData Validation:\n  - Always include a non-null 'tool_arguments' dict (empty if no transform)."
)

# ----------------------------------------------------------------------------
# Logging setup
# ----------------------------------------------------------------------------
# 1. Configure the root logger to INFO (affects everything by default)
logging.basicConfig(level=logging.INFO)

# 2. Create your own logger object
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# 3. Silence HTTPX’s INFO spam
logging.getLogger('httpx').setLevel(logging.WARNING)

# Now:
logger.info('This is your info log')  # will show
logging.getLogger('httpx').info('…')


class WorkflowState(TypedDict):
    """Shared context for the statistical-test workflow."""

    df: pd.Series | pd.DataFrame
    secondary_df: Annotated[pd.Series, 'Secondary data if needed'] | None
    target_columns: Annotated[list[str], 'Columns to test']
    paired: bool | None
    data_type: Literal['CONTINUOUS', 'CATEGORICAL'] | None
    do_association: bool
    number_of_samples: int
    results: Annotated[list, add_messages]
    probabilities: dict[str, float]


def call_test_agent(test_agent: Callable, state: WorkflowState) -> WorkflowState:
    """Call a statistical agent and append its result."""
    try:
        model = OpenAIModel('gpt-4o')
        settings = ModelSettings(temperature=0.0, top_p=1.0, frequency_penalty=0.0, presence_penalty=0.0)
        deps = StatTestDeps(data=state['df'], data_secondary=state.get('secondary_df'))
        result = run_sync_agent(test_agent(model=model, model_settings=settings), user_prompt='', deps=deps)
        state['results'].append(AIMessage(content=str(result)))
        p_val = result.statistical_test_result.p_value
        state['probabilities'][test_agent.__name__] = (
            float(p_val) if isinstance(p_val, float) else float(np.mean(p_val))
        )
        return state
    except Exception as e:
        logger.error(f'Error in call_test_agent {test_agent.__name__}: {e}')
        state['results'].append(AIMessage(content=f'Error: {e}'))
        return state


def call_initialization_agent(state: WorkflowState) -> WorkflowState:
    """Run the initialization agent to analyze data and suggest tests."""
    try:
        model = OpenAIModel('gpt-4o')
        settings = ModelSettings(temperature=0.0, top_p=1.0, frequency_penalty=0.0, presence_penalty=0.0)
        agent = build_initial_insights_agent(
            model=model, system_prompt=INITIAL_INSIGHTS_PROMPT, model_settings=settings
        )
        results = agent.run_sync(
            user_prompt='Analyze the data and suggest the appropriate test.',
            deps=InitialInsightsAgentDeps(
                user_input='Perform a statistical test.',
                input_data=state['df'],
                columns_decision=None,
            ),
        )
        # update sample count
        state['number_of_samples'] = int(results.data.data_size)

        # ensure tool_arguments exists
        tool_args = results.data.tool_arguments or {}
        # apply transformation if any
        if results.data.data_transformation != 'None':
            validated = validate_tool_args(results.data.data_transformation, tool_args)
            state['df'] = TOOL_FUNCS[results.data.data_transformation](state['df'], **validated)
        inp_df = state['df'] if isinstance(state['df'], pd.DataFrame) else pd.DataFrame(state['df'])
        # format for downstream tests
        formatted = format_data_by_recommendation(inp_df, results.data)
        if isinstance(formatted, tuple):
            state['df'], state['secondary_df'] = formatted
        else:
            state['df'] = formatted

        # set metadata
        state['data_type'] = results.data.data_type
        cols = results.data.analysis_columns
        state['target_columns'] = cols if set(cols).issubset(set(inp_df.columns)) else list(inp_df.columns)
        state['results'].append(AIMessage(content=str(results.data)))
        logger.info(f'Data type set: {state["data_type"]}')
        return state
    except Exception as e:
        logger.error(f'Error in call_initialization_agent: {e}')
        state['results'].append(AIMessage(content=f'Error: {e}'))
        return state


def decide_outcome(state: WorkflowState) -> str:
    """Determine continuous vs categorical path."""
    try:
        if state['data_type'] == 'CONTINUOUS':
            return NodeName.ASSESS_STUDY_DESIGN.value
        return NodeName.CHI2.value if state['number_of_samples'] > 10 else NodeName.FISHER.value
    except Exception as e:
        logger.error(f'Error in decide_outcome: {e}')
        return END


def asses_study_design_node(state: WorkflowState) -> WorkflowState:
    model = OpenAIModel('gpt-4o')
    settings = ModelSettings(temperature=0.0, top_p=1.0, frequency_penalty=0.0, presence_penalty=0.0)
    agent = get_assess_design_study_agent(model=model, model_settings=settings)
    logger.info('asses_study_design_node\nmofel is fed with those data:')
    logger.info(state['results'])
    res = agent.run_sync(deps=AssesDesignDeps(msg=state['results']))
    state['paired'] = res.data.paired
    msg = '~~~Paired comparison~~~' if res.data.paired else '~~~Two independent groups~~~'
    logger.info(msg)
    return state


# summariser node usage example (in your workflow)
def summariser_node(state: WorkflowState) -> WorkflowState:
    model = OpenAIModel('gpt-4o')
    settings = ModelSettings(temperature=0.0)

    deps = SummariserDeps(results=state['results'], performed_tests=list(state['probabilities'].keys()))

    agent = get_summariser_agent(model, settings)
    res = agent.run_sync(deps=deps)

    state['results'].append(AIMessage(content=str(res.data)))
    logger.info(f'Summariser output: {res.data}')

    return state


def assess_study_design(state: WorkflowState) -> str:
    """Branch on paired vs independent groups."""
    try:
        return NodeName.NORMALITY_OF_DIFFERENCE.value if state['paired'] else NodeName.TWO_INDEPENDENT_GROUPS.value
    except Exception as e:
        logger.error(f'Error in assess_study_design: {e}')
        return END


def parametric_assumptions(state: WorkflowState) -> str:
    """Run normality & variance checks."""
    try:
        p_normality_of_difference = state['probabilities'].get('normality_of_difference', 0)
        return NodeName.PAIRED_T.value if p_normality_of_difference > 0.05 else NodeName.WILCOXON.value
    except Exception as e:
        logger.error(f'Error in parametric_assumptions: {e}')
        return END


def two_independent_node(state: WorkflowState) -> WorkflowState:
    """Node: run Shapiro–Wilk on each group, then Levene’s test, updating state.probabilities."""
    try:
        # 1) Shapiro–Wilk on group 1

        secondary_df = state['secondary_df']
        orig_df = state['df']
        state['secondary_df'] = None

        state = call_test_agent(shapiro_wilk_agent, state)
        p1 = state['probabilities'].pop('shapiro_wilk_agent', 0)

        # 2) Shapiro–Wilk on group 2

        state['df'] = secondary_df
        state = call_test_agent(shapiro_wilk_agent, state)

        p2 = state['probabilities'].pop('shapiro_wilk_agent', 0)
        state['df'] = orig_df
        state['secondary_df'] = secondary_df

        # store distinct keys
        state['probabilities']['shapiro_group1'] = p1
        state['probabilities']['shapiro_group2'] = p2

        # 3) Levene’s test for equal variances
        state = call_test_agent(levene_agent, state)
        # call_test_agent already stores p under 'levene_agent'
    except Exception as e:
        logger.error(f'Error in two_independent_node: {e}')
    return state


def decide_two_independent(state: WorkflowState) -> str:
    """Edge: choose parametric vs. Welch’s t based on both normals & Levene."""
    try:
        p1 = state['probabilities'].get('shapiro_group1', 0)
        p2 = state['probabilities'].get('shapiro_group2', 0)
        p_leve = state['probabilities'].get('levene_agent', 0)
        if p1 > 0.05 and p2 > 0.05 and p_leve > 0.05:
            return NodeName.INDEP_T.value
        return NodeName.NONPARAMETRIC.value
    except Exception as e:
        logger.error(f'Error in decide_two_independent: {e}')
        return END


def nonparametric_node(state: WorkflowState) -> WorkflowState:
    """Run both Welch’s t‐test and Mann–Whitney U."""
    try:
        state = call_test_agent(welch_t_agent, state)
        state = call_test_agent(mannwhitneyu_agent, state)
    except Exception as e:
        logger.error(f'Error in nonparametric_node: {e}')
    return state


# Build graph
graph = StateGraph(WorkflowState)
graph.add_node('Initialization Agent', call_initialization_agent)
graph.add_conditional_edges(
    'Initialization Agent',
    decide_outcome,
    {
        NodeName.ASSESS_STUDY_DESIGN.value: NodeName.ASSESS_STUDY_DESIGN.value,
        NodeName.CHI2.value: NodeName.CHI2.value,
        NodeName.FISHER.value: NodeName.FISHER.value,
    },
)
graph.add_node(NodeName.ASSESS_STUDY_DESIGN.value, asses_study_design_node)
graph.add_conditional_edges(
    NodeName.ASSESS_STUDY_DESIGN.value,
    assess_study_design,
    {
        NodeName.NORMALITY_OF_DIFFERENCE.value: NodeName.NORMALITY_OF_DIFFERENCE.value,
        NodeName.TWO_INDEPENDENT_GROUPS.value: NodeName.TWO_INDEPENDENT_GROUPS.value,
        END: END,
    },
)
graph.add_conditional_edges(
    NodeName.NORMALITY_OF_DIFFERENCE.value,
    parametric_assumptions,
    {
        NodeName.PAIRED_T.value: NodeName.PAIRED_T.value,
        NodeName.WILCOXON.value: NodeName.WILCOXON.value,
        END: END,
    },
)
graph.add_node(NodeName.TWO_INDEPENDENT_GROUPS.value, two_independent_node)
graph.add_conditional_edges(
    NodeName.TWO_INDEPENDENT_GROUPS.value,
    decide_two_independent,
    {
        NodeName.INDEP_T.value: NodeName.INDEP_T.value,
        NodeName.NONPARAMETRIC.value: NodeName.NONPARAMETRIC.value,
        END: END,
    },
)
tag_list = [
    (NodeName.PAIRED_T.value, ttest_rel_agent),
    (NodeName.WILCOXON.value, wilcoxon_agent),
    (NodeName.INDEP_T.value, ttest_ind_agent),
    (NodeName.FISHER.value, fisher_exact_agent),
    (NodeName.CHI2.value, chi2_agent),
    (NodeName.NORMALITY_OF_DIFFERENCE.value, normality_of_difference_agent),
]
for name, agent_fn in tag_list:
    graph.add_node(name, lambda state, a=agent_fn: call_test_agent(a, state))

graph.add_node(NodeName.NONPARAMETRIC.value, nonparametric_node)
graph.add_edge(NodeName.NONPARAMETRIC.value, NodeName.SUMMARY.value)

graph.add_node(NodeName.SUMMARY.value, summariser_node)
# Add edges from each terminal node to END
for name, _ in tag_list:
    if name not in [NodeName.SHAPIRO.value, NodeName.NORMALITY_OF_DIFFERENCE.value]:
        graph.add_edge(name, NodeName.SUMMARY.value)

graph.add_edge(NodeName.SUMMARY.value, END)

graph.set_entry_point('Initialization Agent')
compiled = graph.compile()
if __name__ == '__main__':
    # df = pd.DataFrame(
    #     {
    #         'usr_id': np.arange(200),
    #         'smoker': np.random.choice(['Yes', 'No'], 200),
    #         'exercise_level': np.random.choice(['Low', 'Medium', 'High'], 200),
    #     }
    # )
    # df = pd.DataFrame({'before_treatment': np.random.normal(50, 5, 80), 'after_treatment': np.random.normal(55, 5, 80)})
    # df = pd.DataFrame(
    #     {
    #         'height': np.random.normal(170, 10, 500),
    #         'gender': np.random.choice(['Male', 'Female'], 500),
    #     }
    # )
    n = 250

    # generate performance: same mean (50) but different σ
    male_performance = np.random.normal(loc=50, scale=5, size=n)
    female_performance = np.random.normal(loc=50, scale=20, size=n)

    # assemble long-format DataFrame
    df = pd.DataFrame(
        {'gender': ['Male'] * n + ['Female'] * n, 'performance': np.concatenate([male_performance, female_performance])}
    )

    initial_state: WorkflowState = {
        'df': df,
        'secondary_df': None,
        'target_columns': [],
        'paired': False,
        'data_type': None,
        'do_association': False,
        'number_of_samples': 0,
        'results': [],
        'probabilities': {},
    }

    try:
        mermaid_src = compiled.get_graph().draw_mermaid()
        md = f'```mermaid\n{mermaid_src}\n```'
        with open('workflow_graph.md', 'w') as f:
            f.write(md)
        logger.info('Saved Mermaid diagram to workflow_graph.md')
    except Exception:
        logger.info('Something is wrong with Graphviz/Mermaid API', exc_info=True)

    prev_len = 0
    msgs = []
    for msg, meta in compiled.stream(initial_state, stream_mode='messages'):
        # msg is your AIMessage, meta["langgraph_node"] is the node name
        # logger.info(f'[{meta["langgraph_node"]}] {msg.content}')
        msgs.append(str(msg.content))

    print('\n\n'.join(msgs))
