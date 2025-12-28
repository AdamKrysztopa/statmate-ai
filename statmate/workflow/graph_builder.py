"""Graph builder for the statistical test workflow.

This module provides functions to construct the workflow graph,
replacing the global graph construction with a proper factory pattern.
"""

from langgraph.graph import END, StateGraph

from statmate.agents import (
    chi2_agent,
    fisher_exact_agent,
    mannwhitneyu_agent,
    normality_of_difference_agent,
    get_reviewer_agent,
    ttest_ind_agent,
    ttest_rel_agent,
    wilcoxon_agent,
    welch_t_agent,
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
    choice_node,
    cox_regression_node,
    design_verification_node,
    intent_discovery_node,
    methodology_auditor_node,
    mcnemar_node,
    nonparametric_node,
    resolve_choice,
    reviewer_node,
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
        self.graph.add_node(NodeName.INITIALIZATION, call_initialization_agent)
        self.graph.set_entry_point(NodeName.INITIALIZATION)
        return self

    def add_intent_node(self) -> 'WorkflowGraphBuilder':
        """Add intent discovery node between initialization and verification."""
        self.graph.add_node(NodeName.INTENT, intent_discovery_node)
        self.graph.add_edge(NodeName.INITIALIZATION, NodeName.INTENT)
        return self

    def add_initial_routing(self) -> 'WorkflowGraphBuilder':
        """Add conditional routing from initialization to test selection.

        Returns:
            Self for method chaining.
        """
        self.graph.add_conditional_edges(
            NodeName.DESIGN_VERIFICATION,
            decide_outcome,
            {
                NodeName.ASSESS_STUDY_DESIGN: NodeName.ASSESS_STUDY_DESIGN,
                NodeName.CHI2: NodeName.CHI2,
                NodeName.FISHER: NodeName.FISHER,
                NodeName.NONPARAMETRIC: NodeName.NONPARAMETRIC,
                NodeName.MCNEMAR: NodeName.MCNEMAR,
                NodeName.CHOICE: NodeName.CHOICE,
                NodeName.COX_REGRESSION: NodeName.COX_REGRESSION,
            },
        )
        return self

    def add_choice_node(self) -> 'WorkflowGraphBuilder':
        """Add choice node to allow elastic user-in-the-loop routing."""
        self.graph.add_node(NodeName.CHOICE, choice_node)
        self.graph.add_conditional_edges(
            NodeName.CHOICE,
            resolve_choice,
            {
                NodeName.PAIRED_T: NodeName.PAIRED_T,
                NodeName.WILCOXON: NodeName.WILCOXON,
                NodeName.INDEP_T: NodeName.INDEP_T,
                NodeName.WELCH: NodeName.WELCH,
                NodeName.MANN: NodeName.MANN,
                NodeName.NONPARAMETRIC: NodeName.NONPARAMETRIC,
                NodeName.CHI2: NodeName.CHI2,
                NodeName.FISHER: NodeName.FISHER,
                NodeName.MCNEMAR: NodeName.MCNEMAR,
                NodeName.COX_REGRESSION: NodeName.COX_REGRESSION,
                NodeName.ASSESS_STUDY_DESIGN: NodeName.ASSESS_STUDY_DESIGN,
            },
        )
        return self

    def add_design_verification(self) -> 'WorkflowGraphBuilder':
        """Add a post-initialization design checkpoint."""
        self.graph.add_node(NodeName.DESIGN_VERIFICATION, design_verification_node)
        # Run sequentially: Initialization -> Intent -> Design Verification
        # Avoids concurrent writes to shared state keys (e.g., df) that LangGraph forbids.
        self.graph.add_edge(NodeName.INTENT, NodeName.DESIGN_VERIFICATION)
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
        def welch_wrapper(state: WorkflowState) -> WorkflowState:
            model = create_model(model_name=state.model_name, provider=state.provider)
            settings = create_model_settings(model_name=state.model_name)
            agent = welch_t_agent(model=model, model_settings=settings)
            return call_test_agent(agent, state, probability_key='welch_t_test')

        self.graph.add_node(NodeName.WELCH, welch_wrapper)

        def mann_wrapper(state: WorkflowState) -> WorkflowState:
            model = create_model(model_name=state.model_name, provider=state.provider)
            settings = create_model_settings(model_name=state.model_name)
            agent = mannwhitneyu_agent(model=model, model_settings=settings)
            return call_test_agent(agent, state, probability_key='mann_whitney_u')

        self.graph.add_node(NodeName.MANN, mann_wrapper)

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
        self.graph.add_node(NodeName.MCNEMAR, mcnemar_node)

        return self

    def add_survival_tests(self) -> 'WorkflowGraphBuilder':
        """Add survival-analysis placeholders."""
        self.graph.add_node(NodeName.COX_REGRESSION, cox_regression_node)
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
            NodeName.WELCH,
            NodeName.MANN,
            NodeName.CHI2,
            NodeName.FISHER,
            NodeName.MCNEMAR,
            NodeName.COX_REGRESSION,
        ]
        for node in terminal_nodes:
            self.graph.add_edge(node, NodeName.SUMMARY)

        # Nonparametric node also goes to summary
        self.graph.add_edge(NodeName.NONPARAMETRIC, NodeName.SUMMARY)

        return self

    def add_reviewer_node(self) -> 'WorkflowGraphBuilder':
        """Add the reviewer/consensus node after summary."""
        self.graph.add_node(NodeName.REVIEWER, reviewer_node)
        self.graph.add_edge(NodeName.METHODOLOGY_AUDITOR, NodeName.REVIEWER)
        self.graph.add_edge(NodeName.REVIEWER, END)
        return self

    def add_methodology_auditor_node(self) -> 'WorkflowGraphBuilder':
        """Add the methodology auditor node before reviewer."""
        self.graph.add_node(NodeName.METHODOLOGY_AUDITOR, methodology_auditor_node)
        self.graph.add_edge(NodeName.SUMMARY, NodeName.METHODOLOGY_AUDITOR)
        return self

    def build(self, checkpointer=None):
        """Build and compile the graph.

        Returns:
            Compiled graph ready for execution.
        """
        if checkpointer:
            return self.graph.compile(checkpointer=checkpointer)
        return self.graph.compile()


def build_workflow_graph(checkpointer=None):
    """Build the complete statistical test workflow graph.

    Returns:
        Compiled workflow graph.
    """
    logger.info('Building workflow graph...')

    builder = WorkflowGraphBuilder()
    graph = (
        builder.add_initialization_node()
        .add_intent_node()
        .add_design_verification()
        .add_initial_routing()
        .add_choice_node()
        .add_study_design_assessment()
        .add_paired_test_path()
        .add_independent_test_path()
        .add_categorical_tests()
        .add_survival_tests()
        .add_summary_node()
        .add_methodology_auditor_node()
        .add_reviewer_node()
        .build(checkpointer=checkpointer)
    )

    logger.info('Workflow graph built successfully')
    return graph


# Create default graph instance
try:
    from config.settings import settings
    from langgraph.checkpoint.sqlite import SqliteSaver
    from pathlib import Path

    def create_default_checkpointer():
        """Create a persistent checkpointer backed by SQLite."""
        checkpoint_dir = settings.DATA_DIR / 'checkpoints'
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
        return SqliteSaver(str(checkpoint_dir / 'workflow.sqlite'))

except Exception:  # pragma: no cover - langgraph optional fallback
    def create_default_checkpointer():
        try:
            from langgraph.checkpoint.memory import MemorySaver

            return MemorySaver()
        except Exception:
            return None


default_graph = build_workflow_graph(checkpointer=create_default_checkpointer())
