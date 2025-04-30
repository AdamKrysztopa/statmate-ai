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
    shapiro_wilk_agent,
    ttest_ind_agent,
    ttest_rel_agent,
    welch_t_agent,
    wilcoxon_agent,
)
from statmate.agents.agent_builder import StatTestDeps, run_sync_agent
from statmate.agents.auxilary_agents import AssesDesignDeps, get_assess_design_study_agent
from statmate.agents.initial_insights_agent import InitialInsightsAgentDeps, NodeName, format_data_by_recommendation

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
        model_settings = ModelSettings(temperature=0.0, top_p=1.0, frequency_penalty=0.0, presence_penalty=0.0)
        deps = StatTestDeps(data=state['df'], data_secondary=state.get('secondary_df'))
        result = run_sync_agent(test_agent(model=model, model_settings=model_settings), user_prompt='', deps=deps)
        logger.info(f'Agent result: {result}')
        state['results'].append(AIMessage(content=str(result)))
        p_value = result.statistical_test_result.p_value
        p_value = p_value if isinstance(p_value, float) else np.mean(p_value)
        state['probabilities'][test_agent.__name__] = float(p_value)
        return state
    except Exception as e:
        logger.error(f'Error in call_test_agent {test_agent.__name__}: {e}')
        state['results'].append(AIMessage(content=f'Error: {e}'))
        return state


def call_initialization_agent(state: WorkflowState) -> WorkflowState:
    """Run the initialization agent to analyze data and suggest tests."""
    try:
        model = OpenAIModel('gpt-4o')
        model_settings = ModelSettings(temperature=0.0, top_p=1.0, frequency_penalty=0.0, presence_penalty=0.0)
        initial_insights_agent = build_initial_insights_agent(
            model=model, system_prompt=INITIAL_INSIGHTS_PROMPT, model_settings=model_settings
        )
        results = initial_insights_agent.run_sync(
            user_prompt='Analyze the data and suggest the appropriate test for the given scenario.',
            deps=InitialInsightsAgentDeps(
                user_input='Perform a statistical test for the provided data.',
                input_data=state['df'],
                columns_decision=None,
            ),
        )
        state['number_of_samples'] = int(results.data.data_size)
        if isinstance(state['df'], pd.DataFrame):
            res = format_data_by_recommendation(state['df'], results.data)
            if isinstance(res, tuple):
                state['df'], state['secondary_df'] = res
            else:
                state['df'] = res
        state['data_type'] = results.data.data_type
        logger.info(f'Data type set to: {state["data_type"]}')
        state['target_columns'] = results.data.analysis_columns
        state['results'].append(AIMessage(content=str(results.data)))
        return state
    except Exception as e:
        logger.error(f'Error in call_initialization_agent: {e}')
        state['results'].append(AIMessage(content=f'Error: {e}'))
        return state


def decide_outcome(state: WorkflowState) -> str:
    """Determine continuous vs categorical path."""
    try:
        if not state.get('data_type'):
            logger.error('Data type not set in state')
            return END
        next_node = (
            NodeName.ASSESS_STUDY_DESIGN.value
            if state['data_type'] == 'CONTINUOUS'
            else NodeName.CHI2.value
            if state['number_of_samples'] > 10
            else NodeName.FISHER.value
        )
        logger.info(f'Next node: {next_node}')
        return next_node
    except Exception as e:
        logger.error(f'Error in decide_outcome: {e}')
        return END


def asses_study_design_node(state: WorkflowState) -> WorkflowState:
    model = OpenAIModel('gpt-4o')
    model_settings = ModelSettings(temperature=0.0, top_p=1.0, frequency_penalty=0.0, presence_penalty=0.0)

    agent = get_assess_design_study_agent(model=model, model_settings=model_settings)

    res = agent.run_sync(model=model, model_settings=model_settings, deps=AssesDesignDeps(msg=state['results']))

    state['paired'] = res.data.paired

    return state


# it is edge - we need node to make sure if paired is in the name
def assess_study_design(state: WorkflowState) -> str:
    """Branch on paired vs independent groups."""
    try:
        next_node = NodeName.SHAPIRO.value if state['paired'] else NodeName.TWO_INDEPENDENT_GROUPS.value
        logger.info(f'Assess study design next node: {next_node}')
        return next_node
    except Exception as e:
        logger.error(f'Error in assess_study_design: {e}')
        return END


def parametric_assumptions(state: WorkflowState) -> str:
    """Run normality & variance checks."""
    try:
        norm_ok = state['probabilities']['shapiro_wilk_agent'] > 0.05
        next_node = NodeName.PAIRED_T.value if norm_ok else NodeName.WILCOXON.value
        logger.info(f'Parametric assumptions next node: {next_node}')
        return next_node
    except Exception as e:
        logger.error(f'Error in parametric_assumptions: {e}')
        return END


def two_independent(state: WorkflowState) -> str:
    """Run normality & variance checks for two independent samples."""
    try:
        state = call_test_agent(shapiro_wilk_agent, state)
        state = call_test_agent(levene_agent, state)
        norm_ok = state['results'][-2].statistical_test_result.p_value > 0.05
        var_ok = state['results'][-1].statistical_test_result.p_value > 0.05
        next_node = NodeName.INDEP_T.value if norm_ok and var_ok else NodeName.WELCH.value
        logger.info(f'Two independent next node: {next_node}')
        return next_node
    except Exception as e:
        logger.error(f'Error in two_independent: {e}')
        return END


# Build the graph
graph = StateGraph(WorkflowState)
graph.add_node('Initialization Agent', call_initialization_agent)
# graph.add_node(NodeName.OUTCOME_TYPE.value, decide_outcome)
graph.add_node(NodeName.ASSESS_STUDY_DESIGN.value, asses_study_design_node)
graph.add_node(NodeName.PARAMETRIC_ASSUMPTIONS.value, parametric_assumptions)
graph.add_node(NodeName.TWO_INDEPENDENT_GROUPS.value, two_independent)

for name, agent in [
    (NodeName.PAIRED_T.value, ttest_rel_agent),
    (NodeName.WILCOXON.value, wilcoxon_agent),
    (NodeName.INDEP_T.value, ttest_ind_agent),
    (NodeName.WELCH.value, welch_t_agent),
    (NodeName.FISHER.value, fisher_exact_agent),
    (NodeName.CHI2.value, chi2_agent),
    (NodeName.SHAPIRO.value, shapiro_wilk_agent),
]:
    graph.add_node(name, lambda state, a=agent: call_test_agent(a, state))

graph.set_entry_point('Initialization Agent')
# Add missing edge

graph.add_conditional_edges(
    'Initialization Agent',
    decide_outcome,
    {
        NodeName.ASSESS_STUDY_DESIGN.value: NodeName.ASSESS_STUDY_DESIGN.value,
        NodeName.CHI2.value: NodeName.CHI2.value,
        NodeName.FISHER.value: NodeName.FISHER.value,
    },
)
graph.add_conditional_edges(
    NodeName.ASSESS_STUDY_DESIGN.value,
    assess_study_design,
    {
        NodeName.SHAPIRO.value: NodeName.SHAPIRO.value,
        NodeName.TWO_INDEPENDENT_GROUPS.value: NodeName.TWO_INDEPENDENT_GROUPS.value,
        END: END,
    },
)
graph.add_conditional_edges(
    NodeName.SHAPIRO.value,
    parametric_assumptions,
    {
        NodeName.PAIRED_T.value: NodeName.PAIRED_T.value,
        NodeName.WILCOXON.value: NodeName.WILCOXON.value,
        END: END,
    },
)
# graph.add_conditional_edges(
#     NodeName.TWO_INDEPENDENT_GROUPS.value,
#     two_independent,
#     {
#         NodeName.INDEP_T.value: NodeName.INDEP_T.value,
#         NodeName.WELCH.value: NodeName.WELCH.value,
#         END: END,
#     },
# )
# graph.add_edge(NodeName.PAIRED_T.value, END)
# graph.add_edge(NodeName.WILCOXON.value, END)
# graph.add_edge(NodeName.INDEP_T.value, END)
# graph.add_edge(NodeName.WELCH.value, END)
# graph.add_edge(NodeName.CHI2.value, END)
# graph.add_edge(NodeName.FISHER.value, END)

compiled = graph.compile()

if __name__ == '__main__':
    # Create sample DataFrame
    # df = pd.DataFrame({'before_treatment': np.random.normal(50, 5, 80), 'after_treatment': np.random.normal(55, 5, 80)})
    df = pd.DataFrame({'before_treatment': np.random.normal(50, 5, 80), 'after_treatment': np.random.normal(55, 5, 80)})
    initial_state: WorkflowState = {
        'df': df,
        'secondary_df': None,
        'target_columns': ['height', 'gender'],  # Fixed to match DataFrame
        'paired': False,
        'data_type': None,
        'do_association': False,
        'number_of_samples': 0,
        'results': [],
        'probabilities': {},  # Added to match the WorkflowState type
    }
    logger.info('\nGraph Structure:')
    graph_representation = compiled.get_graph().draw_ascii()
    print(graph_representation)

    # Stream the workflow
    for evt in compiled.stream(initial_state):
        current_node = evt.get('__node__', 'Unknown')
        state = evt.get(current_node, {})
        logger.info(f'\nProcessing node: {current_node}')
        if 'results' in state and state['results']:
            logger.info('\nCurrent Messages in Workflow:')
            for i, message in enumerate(state['results'], 1):
                logger.info(f'Message {i}: {message.content}')
        if current_node == END:
            logger.info('\nWorkflow completed.')
            logger.info('\nFinal Messages in Workflow:')
            for i, message in enumerate(state['results'], 1):
                logger.info(f'Message {i}: {message.content}')
            break
