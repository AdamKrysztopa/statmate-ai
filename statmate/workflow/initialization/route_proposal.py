"""Phase 3: route proposal based on deterministic rules and agent hints."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from statmate.core.config import NodeName
from statmate.workflow.blueprint import DataBlueprint
from statmate.workflow.edges import decision_engine
from statmate.workflow.initialization.column_role_agent import ColumnRoleResult
from statmate.workflow.initialization.structural_check import StructuralCheckResult


@dataclass
class RouteProposal:
    primary: str | None
    alternatives: list[str]
    ordered_nodes: list[str]
    metadata: dict[str, Any]


def propose_route(
    *,
    structural: StructuralCheckResult,
    roles: ColumnRoleResult,
    blueprint: DataBlueprint | None,
) -> RouteProposal:
    """Determine route candidates with priority ordering."""
    candidates = [str(node.value) if hasattr(node, 'value') else str(node) for node in roles.route_to_test]
    primary = candidates[0] if candidates else None
    alternatives = candidates[1:] if len(candidates) > 1 else []

    # Use decision engine to produce a deterministic suggestion if agent provided none.
    if not primary:
        stub_state = type('Stub', (), {})()
        setattr(stub_state, 'data_blueprint', blueprint)
        setattr(stub_state, 'statistical_design', structural.design)
        setattr(stub_state, 'paired', structural.design.is_paired if structural.design else None)
        setattr(stub_state, 'design_verification', None)
        setattr(stub_state, 'pending_routing_decision', None)
        try:
            primary = decision_engine.evaluate_routing(stub_state)
        except Exception:
            primary = NodeName.ASSESS_STUDY_DESIGN

    ordered = [node for node in [primary, *alternatives] if node]

    return RouteProposal(
        primary=primary,
        alternatives=alternatives,
        ordered_nodes=ordered,
        metadata={
            'agent_candidates': candidates,
            'data_type': roles.data_type,
            'data_design': roles.data_design,
        },
    )
