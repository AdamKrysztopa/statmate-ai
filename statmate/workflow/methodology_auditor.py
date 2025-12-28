"""Deterministic auditor that compares executed tests with registry suggestions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from statmate.core.config import NodeName
from statmate.workflow.edges import DecisionEngine, decision_engine
from statmate.workflow.state import WorkflowState


@dataclass
class AuditResult:
    """Outcome of auditing a workflow run."""

    executed: str | None
    recommended: str | None
    correction_step: dict[str, Any] | None
    conflicts: list[str]


class MethodologyAuditor:
    """Lightweight auditor that reroutes based on assumption outcomes."""

    def __init__(self, engine: DecisionEngine | None = None):
        self.engine = engine or decision_engine

    def _last_executed_test(self, state: WorkflowState) -> str | None:
        for entry in reversed(state.execution_trace):
            name = entry.get('step')
            if name and name not in (
                NodeName.SUMMARY,
                NodeName.REVIEWER,
                NodeName.DESIGN_VERIFICATION,
                NodeName.INITIALIZATION,
                NodeName.CHOICE,
                NodeName.INTENT,
                'Structural Validation',
            ):
                return name
        return None

    def _latest_assumption_status(self, state: WorkflowState) -> dict[str, Any] | None:
        if not state.assumption_log:
            return None
        return state.assumption_log[-1]

    def audit(self, state: WorkflowState) -> AuditResult:
        executed = self._last_executed_test(state)
        assumption_status = self._latest_assumption_status(state)
        recommended = self.engine.evaluate_routing(
            state,
            assumption_status=assumption_status,
            prefer_terminal=True,
        )
        correction: dict[str, Any] | None = None
        conflicts: list[str] = []

        if recommendation_is_conflict := (
            assumption_status
            and assumption_status.get('status') == 'fail'
            and recommended != executed
            and recommended != NodeName.CHOICE
        ):
            correction = {
                'suggested_node': recommended,
                'reason': 'Assumption failure detected',
                'failures': assumption_status.get('failures'),
            }
            state.correction_steps.append(correction)
            state.pending_routing_decision = state.pending_routing_decision or {}
            state.pending_routing_decision.update({'selected': recommended})

        statuses = {entry.get('status') for entry in state.assumption_log if entry.get('status')}
        if 'pass' in statuses and 'fail' in statuses:
            conflicts.append('Assumption diagnostics disagree (pass vs fail).')

        # Resolve CHOICE to a concrete node for the audit summary
        if recommended == NodeName.CHOICE and state.pending_routing_decision:
            recommended = state.pending_routing_decision.get('selected') or state.pending_routing_decision.get('primary')

        return AuditResult(executed=executed, recommended=recommended, correction_step=correction, conflicts=conflicts)
