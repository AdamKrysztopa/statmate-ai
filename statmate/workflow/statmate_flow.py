from collections.abc import Callable
from typing import Annotated, Literal

import pandas as pd
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from numpy import ndarray
from pydantic_ai.models import Model, ModelSettings
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
from statmate.agents.initial_insights_agent import InitialInsightsAgentDeps, NodeName


class WorkflowState(TypedDict):
    """Shared context for the statistical-test workflow."""

    df: ndarray | pd.Series | pd.DataFrame  # Full dataset
    secondary_df: Annotated[ndarray | pd.Series, 'Secondary data if needed']
    target_columns: Annotated[list[str], 'Columns to test']
    paired: bool | None
    data_type: Literal['CONTINUOUS', 'CATEGORICAL'] | None
    do_association: bool
    results: Annotated[list, add_messages]


# Initialize model\ nmodel = OpenAIModel('gpt-4o')


def call_test_agent(
    test_agent: Callable,
    state: WorkflowState,
    model: Model,
    model_settings: ModelSettings | None = None,
) -> WorkflowState:
    """Call a statistical agent and append its result."""
    deps = StatTestDeps(data=state['df'], data_secondary=state.get('secondary_df'))
    result = run_sync_agent(
        test_agent(model=model, model_settings=model_settings),
        user_prompt='',
        deps=deps,
    )
    state['results'].append(result)
    return state


# Node handlers
def call_initialization_agent(state: WorkflowState, model: Model, model_settings: ModelSettings) -> WorkflowState:
    initial_insights_agent = build_initial_insights_agent(
        model=model,
        system_prompt=INITIAL_INSIGHTS_PROMPT,
        model_settings=model_settings,
    )
    results = initial_insights_agent.run_sync(
        user_prompt='Analyze the data and suggest the appropriate test for the given scenario.',
        deps=InitialInsightsAgentDeps(
            user_input='Perform a statistical test for the provided data.',
            input_data=state['df'],
            columns_decision=None,
        ),
    )


def decide_outcome(state: WorkflowState) -> tuple[str, WorkflowState]:
    """Determine continuous vs categorical path."""
    values = state['df'][state['target_columns'][0]]
    state['data_type'] = 'CONTINUOUS' if pd.api.types.is_numeric_dtype(values) else 'CATEGORICAL'
    next_node = NodeName.ASSESS_STUDY_DESIGN.value if state['data_type'] == 'CONTINUOUS' else NodeName.CHI2.value
    return next_node, state


def assess_study_design(state: WorkflowState) -> tuple[str, WorkflowState]:
    """Branch on paired vs independent groups."""
    return (
        (NodeName.PARAMETRIC_ASSUMPTIONS.value, state)
        if state['paired']
        else (NodeName.TWO_INDEPENDENT_GROUPS.value, state)
    )


def parametric_assumptions(state: WorkflowState) -> tuple[str, WorkflowState]:
    """Run normality & variance checks."""
    state = call_test_agent(shapiro_wilk_agent, state, model)
    state = call_test_agent(levene_agent, state, model)
    norm_ok = state['results'][-2].statistical_test_result.p_value > 0.05
    var_ok = state['results'][-1].statistical_test_result.p_value > 0.05
    return (NodeName.PAIRED_T.value, state) if norm_ok and var_ok else (NodeName.WILCOXON.value, state)


def two_independent(state: WorkflowState) -> tuple[str, WorkflowState]:
    """Run normality & variance checks for two independent samples."""
    state = call_test_agent(shapiro_wilk_agent, state, model)
    state = call_test_agent(levene_agent, state, model)
    norm_ok = state['results'][-2].statistical_test_result.p_value > 0.05
    var_ok = state['results'][-1].statistical_test_result.p_value > 0.05
    return (NodeName.INDEP_T.value, state) if norm_ok and var_ok else (NodeName.WELCH.value, state)


def end_node(state: WorkflowState) -> tuple[str, WorkflowState]:
    """Terminate workflow (summary agent can be hooked here)."""
    return END, state


# Build the graph
graph = StateGraph(WorkflowState)

# Add nodes
graph.add_node(START, decide_outcome)
graph.add_node(NodeName.OUTCOME_TYPE.value, decide_outcome)
graph.add_node(NodeName.ASSESS_STUDY_DESIGN.value, assess_study_design)
graph.add_node(NodeName.PARAMETRIC_ASSUMPTIONS.value, parametric_assumptions)
graph.add_node(NodeName.TWO_INDEPENDENT_GROUPS.value, two_independent)

# Final test nodes
for name, agent in [
    (NodeName.PAIRED_T.value, ttest_rel_agent),
    (NodeName.WILCOXON.value, wilcoxon_agent),
    (NodeName.INDEP_T.value, ttest_ind_agent),
    (NodeName.WELCH.value, welch_t_agent),
    (NodeName.FISHER.value, fisher_exact_agent),
    (NodeName.CHI2.value, chi2_agent),
]:
    graph.add_node(name, lambda state, a=agent: (END, call_test_agent(a, state, model)))

graph.add_node(END, end_node)

# Define edges
graph.set_entry_point(START)
graph.add_edge(START, NodeName.OUTCOME_TYPE.value)
graph.add_edge(NodeName.OUTCOME_TYPE.value, NodeName.ASSESS_STUDY_DESIGN.value)
graph.add_edge(NodeName.OUTCOME_TYPE.value, NodeName.CHI2.value)
graph.add_edge(NodeName.OUTCOME_TYPE.value, NodeName.FISHER.value)
graph.add_edge(NodeName.ASSESS_STUDY_DESIGN.value, NodeName.PARAMETRIC_ASSUMPTIONS.value)
graph.add_edge(NodeName.ASSESS_STUDY_DESIGN.value, NodeName.TWO_INDEPENDENT_GROUPS.value)
graph.add_edge(NodeName.PARAMETRIC_ASSUMPTIONS.value, NodeName.PAIRED_T.value)
graph.add_edge(NodeName.PARAMETRIC_ASSUMPTIONS.value, NodeName.WILCOXON.value)
graph.add_edge(NodeName.TWO_INDEPENDENT_GROUPS.value, NodeName.INDEP_T.value)
graph.add_edge(NodeName.TWO_INDEPENDENT_GROUPS.value, NodeName.WELCH.value)

# Leaf-to-END transitions
graph.add_edge(NodeName.PAIRED_T.value, END)
graph.add_edge(NodeName.WILCOXON.value, END)
graph.add_edge(NodeName.INDEP_T.value, END)
graph.add_edge(NodeName.WELCH.value, END)
graph.add_edge(NodeName.CHI2.value, END)
graph.add_edge(NodeName.FISHER.value, END)

# Compile workflow for execution
compiled = graph.compile()


# Example run
if __name__ == '__main__':
    df = pd.DataFrame({'before': [1, 2, 3], 'after': [2, 3, 4]})
    initial_state: WorkflowState = {
        'df': df,
        'secondary_df': None,
        'target_columns': ['before', 'after'],
        'paired': True,
        'data_type': None,
        'do_association': False,
        'results': [],
    }
    for evt in compiled.stream(initial_state):
        # process each step until END
        if evt.node == END:
            print('Workflow completed.')
            break
