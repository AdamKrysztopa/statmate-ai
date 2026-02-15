# StatmateAI — Junior Developer Task List

## How to work
- Pick the first unchecked item.
- Read the linked playbook section before coding.
- Keep tasks small and verify with tests.

---

## P0 — Stabilize (do these first)

- [x] **Fix 2 failing tests** _(Fixed: missing `InitialInsightsAgentResults` import in `nodes.py` — 38/38 pass)_
  - Read: `docs/todo/p0_stabilization_implementation_playbook.md` → P0-1
  - Files: `tests/test_validation_design.py`, `tests/test_workflow_logic.py`
  - Goal: Align tests with current design reconciliation behavior.

- [x] **Add CI pipeline** _(Created `.github/workflows/ci.yml` with pytest + ruff E-level gate)_
  - Read: `docs/todo/p0_stabilization_implementation_playbook.md` → P0-5
  - File: `.github/workflows/ci.yml`
  - Goal: `pytest` + targeted `ruff` gate.

- [x] **Split initialization agent into 3 phases**
  - Read: `docs/todo/p0_stabilization_implementation_playbook.md` → P0-2
  - Files: `statmate/agents/initial_insights_agent.py`, `statmate/workflow/nodes.py`
  - Goal: smaller prompt, cleaner helpers, add tests.

- [x] **Wire user-in-the-loop choice**
  - Read: `docs/todo/p0_stabilization_implementation_playbook.md` → P0-3
  - Files: `statmate/api/routes/analysis.py`, `statmate/api/models/analysis.py`, `frontend/src/App.tsx`
  - Goal: user override flows end-to-end.

- [x] **Apply or remove assumption guardrails**
  - Read: `docs/todo/p0_stabilization_implementation_playbook.md` → P0-4
  - Files: `statmate/core/validation.py`, `statmate/statistical_core/*.py`
  - Goal: no unused guardrail abstraction remains.

---

## P1 — Statistical correctness

- [ ] **Standardize effect sizes + CIs**
  - Read: `docs/todo/p1_statistical_correctness_implementation_playbook.md` → P1-1
  - Files: `statmate/statistical_core/*`

- [ ] **Add power interpretation**
  - Read: `docs/todo/p1_statistical_correctness_implementation_playbook.md` → P1-3

- [ ] **Add regression module**
  - Read: `docs/todo/p1_statistical_correctness_implementation_playbook.md` → P1-4

- [ ] **Add outlier diagnostics**
  - Read: `docs/todo/p1_statistical_correctness_implementation_playbook.md` → P1-5

- [ ] **Resolve survival placeholder**
  - Read: `docs/todo/p1_statistical_correctness_implementation_playbook.md` → P1-6

---

## P2 — Medical reporting quality

- [ ] **Structured summaries (Finding/Evidence/Caveat)**
  - Read: `docs/todo/p2_medical_reporting_implementation_playbook.md` → P2-1

- [ ] **CI/effect-size mention enforcement**
  - Read: `docs/todo/p2_medical_reporting_implementation_playbook.md` → P2-2

- [ ] **Clinical significance block**
  - Read: `docs/todo/p2_medical_reporting_implementation_playbook.md` → P2-3

- [ ] **Reviewer multiplicity checks**
  - Read: `docs/todo/p2_medical_reporting_implementation_playbook.md` → P2-4

- [ ] **Chart narrative captions**
  - Read: `docs/todo/p2_medical_reporting_implementation_playbook.md` → P2-5

---

## P3 — Production scale + UX

- [ ] **Durable workers (Celery/Temporal)**
  - Read: `docs/todo/p3_production_scale_implementation_playbook.md` → P3-1

- [ ] **Real scheduled-task parsing**
  - Read: `docs/todo/p3_production_scale_implementation_playbook.md` → P3-2

- [ ] **API rate limiting**
  - Read: `docs/todo/p3_production_scale_implementation_playbook.md` → P3-3

- [ ] **PII modes + audit report**
  - Read: `docs/todo/p3_production_scale_implementation_playbook.md` → P3-4

- [ ] **Data retention/TTL**
  - Read: `docs/todo/p3_production_scale_implementation_playbook.md` → P3-5

- [ ] **Guided UX wizard + stoplights + column casting**
  - Read: `docs/todo/p3_production_scale_implementation_playbook.md` → P3-6

- [ ] **Decide primary UI (React vs Streamlit)**
  - Read: `docs/todo/p3_production_scale_implementation_playbook.md` → P3-7

---

## Frontend refactor + QA (after P0)

- [ ] **Split `App.tsx` into components**
- [ ] **Add Playwright e2e tests**
- [ ] **Reduce ruff warnings baseline**
- [ ] **Update docs/screenshots**
