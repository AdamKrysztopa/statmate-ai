"""Summary and audit workflow nodes.

Covers: summariser_node, reviewer_node, methodology_auditor_node.
"""

from langchain_core.messages import AIMessage

from statmate.agents.reviewer_agent import ReviewerDeps, get_reviewer_agent
from statmate.agents.summarizer_agent import SummariserDeps, get_summariser_agent
from statmate.core import NodeExecutionError, get_logger
from statmate.core.config import NodeName
from statmate.core.model_provider import execute_with_backoff
from statmate.workflow.methodology_auditor import MethodologyAuditor, StructureAuditor
from statmate.workflow.model_factory import create_model, create_model_settings
from statmate.workflow.state import WorkflowState

logger = get_logger(__name__)


def summariser_node(state: WorkflowState) -> WorkflowState:
    """Generate a summary of all performed tests.

    Args:
        state: Current workflow state.

    Returns:
        Updated workflow state with summary.

    Raises:
        NodeExecutionError: If summary generation fails.
    """
    try:
        model = create_model(model_name=state.model_name, provider=state.provider)
        settings = create_model_settings(model_name=state.model_name)

        deps = SummariserDeps(
            results=state.results,
            performed_tests=list(state.probabilities.keys()),
        )

        agent = get_summariser_agent(model, settings)
        res = execute_with_backoff(
            lambda: agent.run_sync(deps=deps),
            on_retry=lambda attempt, delay, exc: state.add_step(
                step="Rate limit backoff",
                detail=f"Retrying summary in {delay:.1f}s (attempt {attempt})",
                data={"error": str(exc)},
            ),
        )

        state.add_result(AIMessage(content=str(res.data)))
        logger.info(f"Summariser output: {res.data}")
        state.add_step(
            step="Summary",
            detail=res.data.summary if hasattr(res, "data") and hasattr(res.data, "summary") else str(res.data),
            data={"performed_tests": deps.performed_tests},
        )

        return state
    except Exception as e:
        logger.error(f"Error in summariser_node: {e}")
        raise NodeExecutionError(node_name="summariser_node", original_error=e) from e


def reviewer_node(state: WorkflowState) -> WorkflowState:
    """Review the generated summary against raw outputs to catch hallucinations."""
    try:
        model = create_model(model_name=state.model_name, provider=state.provider)
        settings = create_model_settings(model_name=state.model_name)

        summary_text = ""
        if state.results:
            summary_text = str(state.results[-1].content)

        agent = get_reviewer_agent(model, settings)
        deps = ReviewerDeps(summary=summary_text, results=state.results, probabilities=state.probabilities)
        res = execute_with_backoff(
            lambda: agent.run_sync(deps=deps),
            on_retry=lambda attempt, delay, exc: state.add_step(
                step="Rate limit backoff",
                detail=f"Retrying reviewer in {delay:.1f}s (attempt {attempt})",
                data={"error": str(exc)},
            ),
        )

        state.reviewer_report = res.data.model_dump()
        adjusted_summary = res.data.adjusted_summary or summary_text

        # Track reviewer decision for downstream clients
        state.add_result(AIMessage(content=str(res.data)))
        state.add_step(
            step="Reviewer",
            detail="Approved summary" if res.data.approved else "Adjusted summary to match evidence",
            data={
                "approved": res.data.approved,
                "risk_score": res.data.risk_score,
                "flags": res.data.hallucination_flags,
                "adjusted_summary": adjusted_summary,
            },
        )

        # Preserve vetted summary for API consumers
        state.test_hierarchy = state.test_hierarchy or {}
        state.test_hierarchy["reviewer"] = state.reviewer_report
        return state
    except Exception as e:
        logger.error(f"Error in reviewer_node: {e}")
        raise NodeExecutionError(node_name="reviewer_node", original_error=e) from e


def methodology_auditor_node(state: WorkflowState) -> WorkflowState:
    """Audit executed tests and propose corrections when assumptions fail."""
    structure_auditor = StructureAuditor()
    structural = structure_auditor.audit(state)
    auditor = MethodologyAuditor()
    result = auditor.audit(state)
    payload = {
        "executed": result.executed,
        "recommended": result.recommended,
        "correction_step": result.correction_step,
        "conflicts": result.conflicts,
    }
    if structural:
        payload["structural_recommended"] = structural.recommended
        payload["structural_correction"] = structural.correction_step
    state.test_hierarchy = state.test_hierarchy or {}
    state.test_hierarchy["auditor"] = payload
    state.add_step(step=NodeName.METHODOLOGY_AUDITOR, detail="Auditor review complete", data=payload)
    return state
