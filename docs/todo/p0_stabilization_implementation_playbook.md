# P0 Implementation Playbook: Stabilize Flow and Quality Gates

Date: 2026-02-07  
Priority: P0 (immediate, risk-reducing)  
Goal: Remove high-risk regressions, lock contracts, and ensure safe iteration speed.

## 0. Scope

This playbook implements the P0 items from `docs/todo/master_ai_project_audit_and_roadmap.md`:
- fix failing tests and contract drift
- split overloaded initialization flow
- wire real user-in-the-loop choice
- enforce/remove dead assumption guardrail abstraction
- add CI quality gates (`pytest` + targeted lint)

## 1. Pre-flight checklist

- [ ] Branch from latest main.
- [ ] Confirm local commands work:
  - `uv run python -m pytest -q`
  - `uv run ruff check statmate tests frontend/src --output-format concise`
- [ ] Create tracking issues for each workstream (`P0-1` to `P0-5`).

---

## P0-1. Fix failing tests and lock design-verification contract

### Why

Two tests currently fail because behavior and expectations diverged:
- rationale text assertion mismatch in wide-format design detection
- `design_verification_node` no longer raises and now routes to reconciliation

Unclear contract here creates future routing regressions.

### Implementation steps

1. Decide and document one contract for mismatch handling:
- Option A: mismatch should raise and stop
- Option B: mismatch should set `design_verification.mismatch=True` and continue to `DESIGN_RECONCILIATION` (current behavior)

2. Update tests to match chosen contract.
- File: `tests/test_validation_design.py`
- File: `tests/test_workflow_logic.py`

3. Ensure contract is explicit in code docstrings.
- File: `statmate/workflow/nodes.py` (`design_verification_node`)
- File: `statmate/workflow/edges.py` (`decide_outcome` / `DecisionEngine.evaluate_routing`)

4. Ensure rationale assertion is robust to wording changes.
- Prefer semantic assertion (contains `wide-format` and paired indicator), not exact phrase.

### Acceptance criteria

- [ ] `uv run python -m pytest -q` is green.
- [ ] Design mismatch behavior is documented in both implementation and tests.
- [ ] No test relies on brittle exact message phrasing for design rationale.

### How to test

Automated:
- `uv run python -m pytest -q tests/test_validation_design.py tests/test_workflow_logic.py`
- `uv run python -m pytest -q`

Manual:
- Run analysis on intentionally mismatched input and confirm route reaches `Design Reconciliation` in `decision_steps`.

---

## P0-2. Decompose initialization agent responsibilities

### Why

`statmate/agents/initial_insights_agent.py` currently mixes schema understanding, route inference, transformation decisioning, and metadata generation. This is the main flow bottleneck.

### Implementation steps

1. Introduce explicit sub-phases in initialization path:
- structural/schema pre-check (deterministic)
- column/role proposal (agent-assisted)
- route proposal (deterministic engine + agent hints)

2. Create dedicated helper functions and keep each pure where possible.
- File: `statmate/workflow/nodes.py` (`call_initialization_agent` helpers)
- File: `statmate/agents/initial_insights_agent.py`

3. Reduce prompt scope and remove conflicting instructions.
- Keep route logic references but avoid giant "do everything" prompt.
- Add strict output contract tests for missing/invalid `tool_arguments`.

4. Add unit tests for each sub-phase.
- New tests file: `tests/test_initialization_pipeline.py` (or equivalent)

### Acceptance criteria

- [ ] `call_initialization_agent` orchestration is small and delegates to named helpers.
- [ ] Prompt size and complexity reduced without behavior regression.
- [ ] New tests validate sub-phase outputs independently.

### How to test

Automated:
- `uv run python -m pytest -q tests/test_initialization_pipeline.py`
- `uv run python -m pytest -q tests/test_decision_engine.py tests/test_workflow_logic.py`

Manual:
- Run 3 datasets (paired wide, independent long, categorical) and verify `data_blueprint` + route are correct.

---

## P0-3. Wire real user-in-the-loop route choice

### Why

`Choice Node` exists, but no API/frontend contract sets `user_selected_option`, so user choice is effectively not implemented.

### Implementation steps

1. Add override field in analysis request model.
- File: `statmate/api/models/analysis.py`
- Add `route_override: str | None` (or enum-like validated node id)

2. Persist override into workflow state initialization/config.
- File: `statmate/api/services/analysis_service.py`
- File: `statmate/workflow/state.py`
- Set `user_selected_option` before run when override provided.

3. Optionally add dedicated endpoint for in-flight choice if suspend/resume is needed.
- File: `statmate/api/routes/analysis.py`
- Endpoint example: `POST /analysis/{id}/choice`

4. Frontend control to submit choice.
- File: `frontend/src/api/client.ts`
- File: `frontend/src/App.tsx`
- Show options from `pending_routing_decision` and submit chosen value.

5. Validate override against allowed candidates.
- Reject invalid node names with 400.

### Acceptance criteria

- [ ] User can submit route override from UI.
- [ ] Override appears in state (`user_selected_option`) and influences `resolve_choice`.
- [ ] Invalid override is rejected with explicit API error.

### How to test

Automated:
- Add API tests for override validation.
- Add workflow tests ensuring override picks alternative node.

Manual:
- Trigger a run that surfaces choice.
- Choose non-primary option and verify selected path in workflow graph.

---

## P0-4. Enforce or remove `requires_assumptions` dead abstraction

### Why

`requires_assumptions` exists but is not used on production test entrypoints, so guardrail intent is not realized.

### Implementation steps

1. Decide strategy:
- Strategy A: apply decorator to statistical core functions where appropriate.
- Strategy B: remove decorator and rely only on workflow-level rerouting.

2. If Strategy A:
- Apply to relevant functions in:
  - `statmate/statistical_core/comparison.py`
  - `statmate/statistical_core/anova.py`
- Ensure state/assumption context is available where decorator expects it.

3. If Strategy B:
- Remove unused decorator path from `statmate/core/validation.py`
- Update tests/docs accordingly.

### Acceptance criteria

- [ ] No "dead but critical-looking" guardrail API remains.
- [ ] Guardrail behavior is either enforced in code paths or intentionally removed.
- [ ] Tests cover assumption failure behavior.

### How to test

Automated:
- `uv run python -m pytest -q tests/test_decision_engine.py tests/test_workflow_logic.py`
- New unit tests specifically for enforced/removed behavior.

Manual:
- Feed clearly non-normal/heteroscedastic datasets and verify route/test outcomes match policy.

---

## P0-5. Add CI checks and pragmatic lint gate

### Why

No CI currently exists, increasing merge risk.

### Implementation steps

1. Add GitHub Actions workflow.
- New file: `.github/workflows/ci.yml`
- Steps:
  - checkout
  - setup Python
  - install with `uv`
  - run `uv run python -m pytest -q`
  - run `uv run ruff check ...`

2. Scope lint gate pragmatically for initial adoption:
- changed-files lint, or
- allow baseline suppressions with fail-on-new violations.

3. Document CI expectations.
- File: `README.md` or `docs/QUICK_START.md`

### Acceptance criteria

- [ ] CI runs on PR and main.
- [ ] Test failures block merge.
- [ ] Lint policy is explicit and sustainable.

### How to test

Automated:
- Open draft PR and verify CI executes.
- Intentionally break a test and confirm merge gate fails.

Manual:
- Review CI logs for clear failure messages and actionable output.

---

## 2. Definition of done for P0

- [ ] Full test suite green.
- [ ] Explicit mismatch contract documented and covered.
- [ ] Real route override path end-to-end working.
- [ ] Guardrail policy clarified in code (enforced or removed).
- [ ] CI gate active.

