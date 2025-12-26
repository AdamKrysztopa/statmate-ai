"""Graph builder for the statistical test workflow.

This module provides functions to construct the workflow graph,
replacing the global graph construction with a proper factory pattern.
"""

from langgraph.graph import END, StateGraph

from statmate.agents import (
    chi2_agent,
    fisher_exact_agent,
    normality_of_difference_agent,
    ttest_ind_agent,
    ttest_rel_agent,
    wilcoxon_agent,
)
from statmate.core import get_logger
from statmate.core.config import NodeName
from statmate.workflow.edges import (
    assess_study_design,
    decide_outcome,
    decide_two_independent,
    parametric_assumptions,
)
from statmate.workflow.model_factory import create_model, create_model_settings
from statmate.workflow.nodes import (
    assess_study_design_node,
    call_initialization_agent,
    call_test_agent,
    nonparametric_node,
    summariser_node,
    two_independent_node,
)
from statmate.workflow.state import WorkflowState

logger = get_logger(__name__)


class WorkflowGraphBuilder:
    """Builder for constructing the statistical test workflow graph."""

    def __init__(self):
        """Initialize the graph builder."""
        self.graph = StateGraph(WorkflowState)

    def add_initialization_node(self) -> 'WorkflowGraphBuilder':
        """Add the initialization agent node.

        Returns:
            Self for method chaining.
        """
        self.graph.add_node('Initialization Agent', call_initialization_agent)
        self.graph.set_entry_point('Initialization Agent')
        return self

    def add_initial_routing(self) -> 'WorkflowGraphBuilder':
        """Add conditional routing from initialization to test selection.

        Returns:
            Self for method chaining.
        """
        self.graph.add_conditional_edges(
            'Initialization Agent',
            decide_outcome,
            {
                NodeName.ASSESS_STUDY_DESIGN: NodeName.ASSESS_STUDY_DESIGN,
                NodeName.CHI2: NodeName.CHI2,
                NodeName.FISHER: NodeName.FISHER,
            },
        )
        return self

    def add_study_design_assessment(self) -> 'WorkflowGraphBuilder':
        """Add study design assessment node and edges.

        Returns:
            Self for method chaining.
        """
        self.graph.add_node(NodeName.ASSESS_STUDY_DESIGN, assess_study_design_node)
        self.graph.add_conditional_edges(
            NodeName.ASSESS_STUDY_DESIGN,
            assess_study_design,
            {
                NodeName.NORMALITY_OF_DIFFERENCE: NodeName.NORMALITY_OF_DIFFERENCE,
                NodeName.TWO_INDEPENDENT_GROUPS: NodeName.TWO_INDEPENDENT_GROUPS,
                END: END,
            },
        )
        return self

    def add_paired_test_path(self) -> 'WorkflowGraphBuilder':
        """Add nodes and edges for paired tests.

        Returns:
            Self for method chaining.
        """

        # Normality of difference test
        def normality_wrapper(state: WorkflowState) -> WorkflowState:
            model = create_model(model_name=state.model_name, provider=state.provider)
            settings = create_model_settings(model_name=state.model_name)
            agent = normality_of_difference_agent(model=model, model_settings=settings)
            return call_test_agent(agent, state, probability_key='normality_of_difference')

        self.graph.add_node(NodeName.NORMALITY_OF_DIFFERENCE, normality_wrapper)
        self.graph.add_conditional_edges(
            NodeName.NORMALITY_OF_DIFFERENCE,
            parametric_assumptions,
            {
                NodeName.PAIRED_T: NodeName.PAIRED_T,
                NodeName.WILCOXON: NodeName.WILCOXON,
                END: END,
            },
        )

        # Paired t-test
        def paired_t_wrapper(state: WorkflowState) -> WorkflowState:
            model = create_model(model_name=state.model_name, provider=state.provider)
            settings = create_model_settings(model_name=state.model_name)
            agent = ttest_rel_agent(model=model, model_settings=settings)
            return call_test_agent(agent, state, probability_key='paired_t_test')

        self.graph.add_node(NodeName.PAIRED_T, paired_t_wrapper)

        # Wilcoxon test
        def wilcoxon_wrapper(state: WorkflowState) -> WorkflowState:
            model = create_model(model_name=state.model_name, provider=state.provider)
            settings = create_model_settings(model_name=state.model_name)
            agent = wilcoxon_agent(model=model, model_settings=settings)
            return call_test_agent(agent, state, probability_key='wilcoxon_signed_rank')

        self.graph.add_node(NodeName.WILCOXON, wilcoxon_wrapper)

        return self

    def add_independent_test_path(self) -> 'WorkflowGraphBuilder':
        """Add nodes and edges for independent groups tests.

        Returns:
            Self for method chaining.
        """
        # Two independent groups node
        self.graph.add_node(NodeName.TWO_INDEPENDENT_GROUPS, two_independent_node)
        self.graph.add_conditional_edges(
            NodeName.TWO_INDEPENDENT_GROUPS,
            decide_two_independent,
            {
                NodeName.INDEP_T: NodeName.INDEP_T,
                NodeName.NONPARAMETRIC: NodeName.NONPARAMETRIC,
                END: END,
            },
        )

        # Independent t-test
        def indep_t_wrapper(state: WorkflowState) -> WorkflowState:
            model = create_model(model_name=state.model_name, provider=state.provider)
            settings = create_model_settings(model_name=state.model_name)
            agent = ttest_ind_agent(model=model, model_settings=settings)
            return call_test_agent(agent, state, probability_key='independent_t_test')

        self.graph.add_node(NodeName.INDEP_T, indep_t_wrapper)

        # Nonparametric node (Welch + Mann-Whitney)
        self.graph.add_node(NodeName.NONPARAMETRIC, nonparametric_node)

        return self

    def add_categorical_tests(self) -> 'WorkflowGraphBuilder':
        """Add categorical test nodes.

        Returns:
            Self for method chaining.
        """

        # Chi-square test
        def chi2_wrapper(state: WorkflowState) -> WorkflowState:
            model = create_model(model_name=state.model_name, provider=state.provider)
            settings = create_model_settings(model_name=state.model_name)
            agent = chi2_agent(model=model, model_settings=settings)
            return call_test_agent(agent, state, probability_key='chi_square')

        self.graph.add_node(NodeName.CHI2, chi2_wrapper)

        # Fisher exact test
        def fisher_wrapper(state: WorkflowState) -> WorkflowState:
            model = create_model(model_name=state.model_name, provider=state.provider)
            settings = create_model_settings(model_name=state.model_name)
            agent = fisher_exact_agent(model=model, model_settings=settings)
            return call_test_agent(agent, state, probability_key='fisher_exact')

        self.graph.add_node(NodeName.FISHER, fisher_wrapper)

        return self

    def add_summary_node(self) -> 'WorkflowGraphBuilder':
        """Add summary node and edges from all terminal nodes.

        Returns:
            Self for method chaining.
        """
        self.graph.add_node(NodeName.SUMMARY, summariser_node)

        # Add edges from all terminal test nodes to summary
        terminal_nodes = [
            NodeName.PAIRED_T,
            NodeName.WILCOXON,
            NodeName.INDEP_T,
            NodeName.CHI2,
            NodeName.FISHER,
        ]
        for node in terminal_nodes:
            self.graph.add_edge(node, NodeName.SUMMARY)

        # Nonparametric node also goes to summary
        self.graph.add_edge(NodeName.NONPARAMETRIC, NodeName.SUMMARY)

        # Summary goes to END
        self.graph.add_edge(NodeName.SUMMARY, END)

        return self

    def build(self):
        """Build and compile the graph.

        Returns:
            Compiled graph ready for execution.
        """
        return self.graph.compile()


def build_workflow_graph():
    """Build the complete statistical test workflow graph.

    Returns:
        Compiled workflow graph.
    """
    logger.info('Building workflow graph...')

    builder = WorkflowGraphBuilder()
    graph = (
        builder.add_initialization_node()
        .add_initial_routing()
        .add_study_design_assessment()
        .add_paired_test_path()
        .add_independent_test_path()
        .add_categorical_tests()
        .add_summary_node()
        .build()
    )

    logger.info('Workflow graph built successfully')
    return graph


# Create default graph instance
default_graph = build_workflow_graph()
