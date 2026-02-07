# P3 Implementation Playbook: Production Scale, Security, and UX Maturity

Date: 2026-02-07  
Priority: P3 (scale/compliance/extensions)  
Goal: Move from strong prototype to resilient multi-user production platform.

## 0. Scope

This playbook implements P3 from `docs/todo/master_ai_project_audit_and_roadmap.md`:
- durable worker execution
- real scheduler semantics
- API abuse controls
- stronger PII/data lifecycle controls
- guided frontend workflow UX
- primary UI strategy decision

---

## P3-1. Migrate analysis execution to durable workers

### Why

Current `BackgroundTasks` model is process-local and not durable across restarts.

### Implementation steps

1. Choose execution platform:
- Celery + Redis (recommended for current stack), or
- Temporal (if workflow orchestration maturity is desired).

2. Introduce worker boundary:
- API route only enqueues job.
- Worker executes `AnalysisService.run_analysis`.

3. Persist job metadata and retries.
- Update analysis status transitions to handle queued/retried/failed cleanly.

4. Keep SSE status compatibility.
- API reads status from DB and streams progress from persisted `decision_steps`.

### Target files

- `statmate/api/routes/analysis.py`
- `statmate/api/services/analysis_service.py`
- `statmate/api/scheduler/*` (if scheduler dispatch changes)
- infra files (`docker-compose`, deployment docs)

### Acceptance criteria

- [ ] Analysis continues/retries after API process restart.
- [ ] Worker failures are visible with deterministic status updates.

### How to test

Automated:
- Integration tests with worker queue and retry simulation.

Manual:
- Start analysis, restart API process, confirm run status survives.

---

## P3-2. Implement real scheduled-task parsing and next-run semantics

### Why

`TaskService` currently uses placeholder `next_run = datetime.utcnow()`.

### Implementation steps

1. Define supported schedule formats:
- one-time ISO datetime
- recurring cron expression

2. Implement parse/validate utility.
- File: `statmate/api/services/task_service.py`
- Add clear validation errors.

3. Compute and persist `next_run` for both types.

4. Update scheduler integration to consume persisted schedule deterministically.

### Acceptance criteria

- [ ] Invalid schedules return clear 400 errors.
- [ ] `next_run` is accurate and updates after task execution.

### How to test

Automated:
- New tests: `tests/test_task_schedule_parsing.py`
- Edge cases: timezone handling, invalid cron, past dates.

Manual:
- Create one-time and recurring tasks via API and verify expected next-run values.

---

## P3-3. Add API rate-limiting and abuse protections

### Why

No active rate-limiting middleware leaves platform vulnerable to abuse and cost spikes.

### Implementation steps

1. Add rate-limiting middleware/library.
- Candidate: `slowapi` or gateway-level enforcement.

2. Define quotas:
- auth endpoints
- dataset upload endpoints
- analysis run endpoints
- stream endpoints

3. Add user-aware and IP-aware limits.

4. Return explicit rate-limit headers and friendly error payloads.

### Target files

- `statmate/api/main.py`
- route modules under `statmate/api/routes/`
- config knobs in `config/settings.py`

### Acceptance criteria

- [ ] Limits enforced with deterministic 429 behavior.
- [ ] Limits are configurable per environment.

### How to test

Automated:
- Load tests validating 429 threshold behavior.

Manual:
- Burst requests and confirm throttling + retry semantics.

---

## P3-4. Add strict PII modes and auditability

### Why

Current masking is heuristic and one-mode; sensitive deployments need explicit policy controls.

### Implementation steps

1. Add configurable PII modes:
- `mask`
- `drop`
- `off` (dev-only)

2. Expand PII detection/report payload:
- column-level actions
- reason/pattern
- counts

3. Persist PII action report per analysis run.
- Include in `assumption_log` or dedicated field.

### Target files

- `statmate/core/pii.py`
- `statmate/workflow/statmate_flow_refactored.py`
- `statmate/api/models/analysis.py` (if new field added)

### Acceptance criteria

- [ ] Mode is configurable and enforced before any LLM calls.
- [ ] Every run has a clear PII audit report.

### How to test

Automated:
- Unit tests with synthetic PII-heavy datasets.

Manual:
- Run same dataset in each mode and verify expected masking/dropping behavior.

---

## P3-5. Add data lifecycle controls (retention/TTL)

### Why

No explicit retention policy is risky for compliance-sensitive data.

### Implementation steps

1. Add retention config:
- dataset TTL
- analysis artifact TTL
- logs TTL

2. Add cleanup jobs.
- scheduler job to remove expired assets + DB tombstones where needed.

3. Add audit logs for deletion actions.

### Target files

- `config/settings.py`
- `statmate/api/scheduler/jobs.py`
- `statmate/api/services/storage_service.py`
- `statmate/api/services/dataset_service.py`

### Acceptance criteria

- [ ] Expired artifacts are cleaned automatically.
- [ ] Cleanup actions are auditable.

### How to test

Automated:
- Time-travel or short TTL integration tests.

Manual:
- Create short-lived artifacts and verify auto-deletion path.

---

## P3-6. Implement guided frontend workflow UX

### Why

Current UX is powerful but not yet guided enough for non-experts.

### Implementation steps

1. Research question wizard.
- map user intent to likely statistical paths

2. Assumption stoplights.
- green/yellow/red from `assumption_log`

3. Spreadsheet-first type casting.
- editable column type overrides before run

4. Explainability overlays.
- "why this test" panel tied to decision steps/workflow graph

### Target files

- `frontend/src/App.tsx` (refactor recommended into feature components)
- `frontend/src/components/*`
- `frontend/src/api/client.ts`

### Acceptance criteria

- [ ] User can reach valid analysis path through wizard without manual API knowledge.
- [ ] Assumption status is visually obvious.
- [ ] Column type overrides are persisted and used in run request.

### How to test

Automated:
- Frontend unit tests for wizard state and override payloads.

Manual:
- Mobile + desktop walkthrough across three representative datasets.

---

## P3-7. Decide and execute primary UI strategy

### Why

Dual React/Streamlit paths increase maintenance overhead and product inconsistency.

### Implementation steps

1. Make product decision:
- React-primary with Streamlit internal/demo only, or
- maintain parity intentionally with explicit ownership.

2. If React-primary:
- freeze Streamlit features
- document support matrix
- remove drift-prone duplicated workstreams

### Acceptance criteria

- [ ] UI strategy documented and agreed.
- [ ] Ownership and support expectations are explicit.

### How to test

Manual:
- Validate onboarding docs and runbooks match chosen UI strategy.

---

## 2. Definition of done for P3

- [ ] Worker durability is in place.
- [ ] Scheduler semantics are real, not placeholder.
- [ ] Rate limits + PII policy + retention controls are active.
- [ ] Guided UX is implemented and measurable.
- [ ] UI strategy is explicitly decided.

