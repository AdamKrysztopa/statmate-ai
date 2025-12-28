# Epic: Routing Recovery & Design Reconciliation

Goal: Eliminate "Design Mismatch" halts by allowing the system to re-verify structural assumptions when agents and validators disagree.

## Reconciliation Logic
- [x] Implement `DesignReconciliationAgent`:
  - If `DesignVerificationNode` detects a mismatch (e.g., Validator=Independent vs Agent=Paired), route to this agent instead of halting.
  - The agent must specifically check for "Wide-Format Pairing" (pre/post columns) vs "Long-Format Grouping."
- [x] Add `ForcePairingTransform`:
  - If reconciliation confirms pairing in wide format, force-update the `DataBlueprint.is_paired = True` and re-partition the data.

## Partitioning Guardrails
- [x] Grouping Sanity Check:
  - In `initial_insights_agent.py`, block the use of high-cardinality columns (like 'id') as grouping variables.
  - If a suggested grouping results in `n < 2` for all groups while the total dataset is large, trigger a `RoutingError`.

## Decision Engine Elasticity
- [x] Update `evaluate_routing` in `edges.py`:
  - Instead of `raise RoutingError` on mismatch, return `NodeName.DESIGN_RECONCILIATION`.
