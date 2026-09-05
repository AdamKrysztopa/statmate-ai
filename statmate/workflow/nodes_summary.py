"""Summary and audit workflow nodes.

Covers: summariser_node, reviewer_node, methodology_auditor_node.
"""

from typing import Any

from langchain_core.messages import AIMessage

from statmate.agents.reviewer_agent import (
    ReviewerDeps,
    ReviewerEvidence,
    apply_reviewer_informational_flags,
    get_reviewer_agent,
)
from statmate.agents.summarizer_agent import SummariserDeps, get_summariser_agent
from statmate.core import NodeExecutionError, get_logger
from statmate.core.config import NodeName
from statmate.core.model_provider import execute_with_backoff
from statmate.workflow.methodology_auditor import MethodologyAuditor, StructureAuditor
from statmate.workflow.model_factory import create_model, create_model_settings
from statmate.workflow.state import WorkflowState

logger = get_logger(__name__)

_DIAGNOSTIC_TEST_TOKENS = (
    'anderson',
    'assumption',
    'bartlett',
    'cramer',
    'dagostino',
    'jarque',
    'kolmogorov',
    'levene',
    'lilliefors',
    'normality',
    'shapiro',
)


def _get_summarizer_payload(state: WorkflowState) -> dict[str, Any] | None:
    if not state.test_hierarchy:
        return None
    payload = state.test_hierarchy.get('summarizer')
    return payload if isinstance(payload, dict) else None


def _normalize_test_name(test_name: str) -> str:
    return test_name.casefold().replace('_', ' ').replace('-', ' ')


def _is_diagnostic_test_name(test_name: str) -> bool:
    normalized_test_name = _normalize_test_name(test_name)
    return any(token in normalized_test_name for token in _DIAGNOSTIC_TEST_TOKENS)


def _get_reported_test_names(state: WorkflowState) -> list[str]:
    reported_tests: list[str] = []
    seen_tests: set[str] = set()
    for entry in state.execution_trace:
        data = entry.get('data')
        if not isinstance(data, dict):
            continue
        test_name = data.get('test_name')
        if not isinstance(test_name, str) or _is_diagnostic_test_name(test_name) or test_name in seen_tests:
            continue
        reported_tests.append(test_name)
        seen_tests.add(test_name)
    if reported_tests:
        return reported_tests
    return [test_name for test_name in state.probabilities if not _is_diagnostic_test_name(test_name)]


def _build_reported_probabilities(state: WorkflowState) -> dict[str, float]:
    reported_probabilities: dict[str, float] = {}
    for entry in state.execution_trace:
        data = entry.get('data')
        if not isinstance(data, dict):
            continue
        test_name = data.get('test_name')
        if not isinstance(test_name, str) or _is_diagnostic_test_name(test_name):
            continue
        p_value_raw = entry.get('p_value')
        if not isinstance(p_value_raw, (float, int)):
            continue
        reported_probabilities[test_name] = float(p_value_raw)
    if reported_probabilities:
        return reported_probabilities
    return {
        test_name: p_value
        for test_name, p_value in state.probabilities.items()
        if not _is_diagnostic_test_name(test_name)
    }


def _build_reviewer_result_context(state: WorkflowState) -> list[ReviewerEvidence]:
    result_context: list[ReviewerEvidence] = []
    for entry in state.execution_trace:
        data = entry.get('data')
        if not isinstance(data, dict):
            continue
        test_name = data.get('test_name')
        if not isinstance(test_name, str) or _is_diagnostic_test_name(test_name):
            continue
        if 'statistics' not in data and 'effect_size_type' not in data and 'confidence_interval' not in data:
            continue
        confidence_interval_raw = data.get('confidence_interval')
        confidence_interval = confidence_interval_raw if isinstance(confidence_interval_raw, list) else None
        p_value_raw = entry.get('p_value')
        p_value = float(p_value_raw) if isinstance(p_value_raw, (float, int)) else None
        result_context.append(
            ReviewerEvidence(
                test_name=test_name,
                effect_size_type=data.get('effect_size_type')
                if isinstance(data.get('effect_size_type'), str)
                else None,
                confidence_interval=confidence_interval,
                p_value=p_value,
            )
        )
    return result_context


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
        reported_tests = _get_reported_test_names(state)

        deps = SummariserDeps(
            results=state.results,
            performed_tests=reported_tests,
        )

        agent = get_summariser_agent(model, settings)
        res = execute_with_backoff(
            lambda: agent.run_sync(deps=deps),
            on_retry=lambda attempt, delay, exc: state.add_step(
                step='Rate limit backoff',
                detail=f'Retrying summary in {delay:.1f}s (attempt {attempt})',
                data={'error': str(exc)},
            ),
        )

        state.add_result(AIMessage(content=str(res.output)))
        state.test_hierarchy = state.test_hierarchy or {}
        state.test_hierarchy['summarizer'] = res.output.model_dump()
        logger.info(f'Summariser output: {res.output}')
        state.add_step(
            step='Summary',
            detail=res.output.summary if hasattr(res, 'output') and hasattr(res.output, 'summary') else str(res.output),
            data={
                'performed_tests': deps.performed_tests,
                'findings_count': len(res.output.findings),
            },
        )

        return state
    except Exception as e:
        logger.error(f'Error in summariser_node: {e}')
        raise NodeExecutionError(node_name='summariser_node', original_error=e) from e


def reviewer_node(state: WorkflowState) -> WorkflowState:
    """Review the generated summary against raw outputs to catch hallucinations."""
    try:
        model = create_model(model_name=state.model_name, provider=state.provider)
        settings = create_model_settings(model_name=state.model_name)

        summary_text = ''
        if state.results:
            summary_text = str(state.results[-1].content)

        summarizer_payload = _get_summarizer_payload(state) or {}
        reported_tests = _get_reported_test_names(state)
        reported_probabilities = _build_reported_probabilities(state)
        findings = (
            summarizer_payload.get('findings', []) if isinstance(summarizer_payload.get('findings', []), list) else []
        )
        performed_tests = summarizer_payload.get('performed_tests', reported_tests)
        if not isinstance(performed_tests, list):
            performed_tests = reported_tests
        performed_tests = [
            str(test_name)
            for test_name in performed_tests
            if isinstance(test_name, str) and not _is_diagnostic_test_name(test_name)
        ]
        if not performed_tests:
            performed_tests = reported_tests
        raw_results = state.results[:-1] if state.results else []

        agent = get_reviewer_agent(model, settings)
        deps = ReviewerDeps(
            summary=summary_text,
            results=raw_results,
            probabilities=reported_probabilities,
            performed_tests=[str(test_name) for test_name in performed_tests],
            findings=findings,
            result_context=_build_reviewer_result_context(state),
        )
        res = execute_with_backoff(
            lambda: agent.run_sync(deps=deps),
            on_retry=lambda attempt, delay, exc: state.add_step(
                step='Rate limit backoff',
                detail=f'Retrying reviewer in {delay:.1f}s (attempt {attempt})',
                data={'error': str(exc)},
            ),
        )

        reviewer_result = apply_reviewer_informational_flags(res.output, deps)
        state.reviewer_report = reviewer_result.model_dump()
        adjusted_summary = reviewer_result.adjusted_summary or summary_text

        # Track reviewer decision for downstream clients
        state.add_result(AIMessage(content=str(reviewer_result)))
        state.add_step(
            step='Reviewer',
            detail='Approved summary' if reviewer_result.approved else 'Adjusted summary to match evidence',
            data={
                'approved': reviewer_result.approved,
                'risk_score': reviewer_result.risk_score,
                'flags': reviewer_result.hallucination_flags,
                'missing_structure_flags': reviewer_result.missing_structure_flags,
                'missing_effect_size_flags': reviewer_result.missing_effect_size_flags,
                'missing_ci_flags': reviewer_result.missing_ci_flags,
                'multiplicity_warning': reviewer_result.multiplicity_warning,
                'adjusted_summary': adjusted_summary,
            },
        )

        # Preserve vetted summary for API consumers
        state.test_hierarchy = state.test_hierarchy or {}
        state.test_hierarchy['reviewer'] = state.reviewer_report
        return state
    except Exception as e:
        logger.error(f'Error in reviewer_node: {e}')
        raise NodeExecutionError(node_name='reviewer_node', original_error=e) from e


def methodology_auditor_node(state: WorkflowState) -> WorkflowState:
    """Audit executed tests and propose corrections when assumptions fail."""
    structure_auditor = StructureAuditor()
    structural = structure_auditor.audit(state)
    auditor = MethodologyAuditor()
    result = auditor.audit(state)
    payload = {
        'executed': result.executed,
        'recommended': result.recommended,
        'correction_step': result.correction_step,
        'conflicts': result.conflicts,
    }
    if structural:
        payload['structural_recommended'] = structural.recommended
        payload['structural_correction'] = structural.correction_step
    state.test_hierarchy = state.test_hierarchy or {}
    state.test_hierarchy['auditor'] = payload
    state.add_step(step=NodeName.METHODOLOGY_AUDITOR, detail='Auditor review complete', data=payload)
    return state
