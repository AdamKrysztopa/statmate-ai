# Routing & Decision Engine (Refactor)

This refactor replaces ad-hoc `if/else` routing with deterministic, data-aware components that make adding new tests predictable.

## Data Blueprint
- Built via `build_data_blueprint` in `statmate/workflow/blueprint.py`.
- Captures:
  - `variable_roles` (Independent, Dependent, Covariate, Group)
  - `distribution_metrics` (skewness, kurtosis, normality_p_value)
  - `sample_balance` (group sizes, ratio, balanced flag)
  - `survival_data` heuristic
- Blueprint is attached to `WorkflowState.data_blueprint` during initialization and treated as immutable context for all nodes.

## Decision Engine & Registry
- New `MethodRegistry` + `DecisionEngine` live in `statmate/workflow/edges.py`.
- Registry maps `MethodProfile(scale, normal, groups, paired)` to weighted `MethodSuggestion(primary, alternatives)`.
- `DecisionEngine.evaluate_routing` reads the blueprint + hard constraints:
  - survival → `COX_REGRESSION`
  - categorical + paired → `MCNEMAR`
  - continuous + non-normal + 2 groups → `NONPARAMETRIC`
  - categorical sample size threshold still chooses `CHI2` vs `FISHER`
- Registry suggestions are cached on state for user choices and the methodology auditor.

## User-in-the-Loop Nodes
- `Intent Agent` node records an intent summary + confidence, flagging ambiguity when < 0.8.
- `Choice Node` surfaces primary vs alternative routes; selections are logged in `choice_log`.

## Auditor & Rerouting
- `MethodologyAuditor` compares executed tests with registry suggestions (considering assumption failures).  
  - If variance/normality fails after an independent t-test, it proposes Welch as a `correction_step`.
  - Conflicting assumption diagnostics are surfaced in `test_hierarchy['auditor']`.

## Guardrails
- `@requires_assumptions(normality=..., variance=...)` decorator (in `statmate/core/validation.py`) raises `StatisticalAssumptionError` when blueprint or logged diagnostics show violations.
- `validate_assumptions` now returns a `status` field for deterministic rerouting.

## Graph Updates
- New nodes: `INTENT`, `CHOICE`, `MCNEMAR`, `COX_REGRESSION`, `METHODOLOGY_AUDITOR`, plus dedicated Welch/Mann-Whitney nodes.
- `graph_builder` wires these into the LangGraph pipeline so future methods can be registered without modifying edge logic.

## Tests
- Added `tests/test_decision_engine.py` covering blueprint construction, hard constraints, auditor corrections, choice node routing, and decorator enforcement.
- Existing workflow logic tests continue to pass under the new routing setup (`uv run pytest ...`).
