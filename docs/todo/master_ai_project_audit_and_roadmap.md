# StatmateAI Comprehensive Audit and Unified Roadmap

Date: 2026-02-07
Owner: Core product + AI workflow team
Scope: Statistical core, agent/router flow, prompts, guardrails, backend, frontend, delivery readiness.

## 1. Executive verdict

StatmateAI has a strong base and meaningful progress, but it is not yet production-ready for high-stakes medical analysis.

Current quality snapshot:
- Strongest areas:
  - Deterministic routing improvements and design reconciliation flow are implemented.
  - Streaming decision trace and workflow graph UX are implemented.
  - Auth and credential security are materially stronger than typical prototypes.
- Weakest areas:
  - Agent flow remains the main bottleneck (overloaded initialization prompt + incomplete user-in-the-loop wiring).
  - Statistical outputs are still incomplete for publication-grade reporting (effect sizes/CIs/power/regression coverage).
  - Production execution model still relies on process-local background tasks (no durable worker queue).

## 2. Evidence baseline (as of this audit)

- Automated tests:
  - `uv run python -m pytest -q` => 36 passed, 2 failed.
  - Failing tests indicate drift between older expectations and current reconciliation behavior/text.
- Lint quality:
  - `uv run ruff check statmate tests frontend/src --output-format concise` => 591 findings (550 remaining).
  - High signal: complexity hotspots and architecture debt are real, not just style issues.
- Structural complexity hotspots:
  - `frontend/src/App.tsx` ~1940 LOC.
  - `statmate/workflow/nodes.py` ~1201 LOC.
  - `statmate/core/validation.py` ~894 LOC.
  - `statmate/api/services/analysis_service.py` ~723 LOC.
  - `statmate/agents/initial_insights_agent.py` ~598 LOC.

## 3. Current status by domain

### 3.1 Statistical model/core

Implemented well:
- Two-group and paired core tests exist (t-test family, Mann-Whitney, Wilcoxon).
- Multi-group non-parametric/parametric routing exists (ANOVA path, Kruskal-Wallis + Dunn, Friedman).
- Categorical coverage improved (Chi-square, Fisher, McNemar, Cochran-Armitage).
- Result model already supports `effect_size_type` and `confidence_interval` fields.

Major gaps:
- Cohen's d and 95% CIs are not consistently produced by core comparison functions.
- Partial eta squared and standardized CI payloads are not standardized across ANOVA/non-parametric outputs.
- No real regression module (`cox_regression` is still placeholder in workflow).
- No robust outlier/influence pipeline (Cook's distance, leverage, VIF, formal power interpretation).

### 3.2 Agent and routing flow (primary bottleneck)

Implemented:
- Data blueprint, sufficiency checks, and deterministic decision engine constraints are in place.
- Design verification + reconciliation are implemented and route mismatches instead of hard-failing.

Key bottlenecks:
- Initialization agent is overburdened (large monolithic prompt + routing + column logic + transforms).
- `Choice Node` exists but user override is not actually wired through API/frontend (`user_selected_option` has no route contract).
- Intent node is heuristic-only (confidence based on simple state checks), not robust intent modeling.
- Assumption guardrail decorator exists but is not applied to production statistical functions.

### 3.3 Prompt quality

Strengths:
- Summarizer and reviewer include anti-hallucination constraints.
- Initial prompt includes structure-aware instructions and tool argument enforcement.

Gaps:
- Initial insights prompt is too long/fragile and mixes many responsibilities.
- Prompt contracts are not aligned with medical reporting standards (CONSORT/STROBE).
- Summary/reviewer chain is not yet explicitly enforcing Finding/Evidence/Caveat + clinical meaningfulness.

### 3.4 Guardrails and safety

Implemented:
- PII masking is integrated before LLM calls.
- Assumption diagnostics are logged and surfaced.
- Credential encryption + quota checks are implemented.
- Auth defaults to required and ownership checks are broadly enforced.

Gaps:
- PII masking remains heuristic and can over-mask ID-like columns without policy modes.
- No formal data retention/residency policy enforcement (auto-delete/TTL per tenant).
- No API rate-limiting middleware for abuse control.
- Development CORS is intentionally broad; production hardening checklist still open.

### 3.5 Backend/platform

Implemented:
- FastAPI API surface is broad and usable.
- Streaming endpoint exists and feeds UI decision/assumption updates.
- Export stack is robust (HTML/PDF/DOCX/CSV/LaTeX + reproducibility bundle elements).

Gaps:
- Analysis execution still uses FastAPI background tasks, not durable queue workers.
- Scheduled task parsing is placeholder (`next_run` currently mocked to `datetime.utcnow()`).
- No CI pipeline in repo; quality checks are local/manual.

### 3.6 Frontend (React + Streamlit)

Implemented:
- React app has live streaming, workflow graph visualization, and rich result panels.
- Assumption logs and hierarchy are surfaced.

Gaps:
- React app is monolithic and hard to evolve safely.
- Guided analysis wizard, assumption stoplights, deep "why this test?" explainer UX are still missing.
- Spreadsheet-first schema editing/type-casting UX is still not implemented.
- Streamlit app remains a second UI path and is materially behind React UX.

## 4. Unified roadmap (single source of truth)

Priority legend:
- P0: immediate, risk-reducing.
- P1: core product correctness/completeness.
- P2: medical publication quality.
- P3: scale/compliance/extensions.

Detailed implementation playbooks:
- `docs/todo/p0_stabilization_implementation_playbook.md`
- `docs/todo/p1_statistical_correctness_implementation_playbook.md`
- `docs/todo/p2_medical_reporting_implementation_playbook.md`
- `docs/todo/p3_production_scale_implementation_playbook.md`

### P0 - Stabilize flow and quality gates (now)

- [ ] Fix the 2 failing tests and align tests with current design reconciliation behavior.
- [ ] Break `initial_insights_agent` responsibilities into smaller contracts:
  - deterministic schema/design pre-check
  - column-role selection
  - route proposal
- [ ] Wire real user-in-the-loop choice:
  - API field for route override
  - frontend control to submit selected route
  - workflow consumption of override in `Choice Node`.
- [ ] Decide one design-mismatch contract and enforce it:
  - either hard-fail, or always reconcile, with clear telemetry and tests.
- [ ] Apply `requires_assumptions` guardrails to production statistical entrypoints or remove dead abstraction.
- [ ] Add CI workflow:
  - `pytest`
  - targeted `ruff` policy (at least changed-files gate).

Acceptance criteria:
- Test suite green.
- User route override demonstrably changes downstream node selection.
- CI blocks regressions on PRs.

### P1 - Complete statistical correctness for analyst workflows

- [ ] Implement standardized effect sizes and confidence intervals across all core tests.
- [ ] Add power interpretation for non-significant findings where feasible.
- [ ] Add regression module:
  - linear regression
  - multiple regression with VIF checks
  - logistic regression for binary outcomes.
- [ ] Add outlier/influence diagnostics:
  - Cook's distance
  - leverage
  - robust outlier summary in blueprint/assumption log.
- [ ] Replace survival placeholder with implemented survival analysis path or explicitly disable route.

Acceptance criteria:
- Result payload schema is consistent for all major test families.
- Regression/outlier coverage has unit tests and appears in summaries/exports.

### P2 - Medical-grade summaries and reviewer logic

- [ ] Refactor summarizer output to required template:
  - Finding
  - Evidence
  - Caveat.
- [ ] Enforce CI + effect size mention policy in summary/reviewer.
- [ ] Add clinical significance judgment block (explicit threshold/rationale hooks).
- [ ] Add reviewer checks for p-hacking / multiple-comparison caveats.
- [ ] Add chart narrative captions for non-statisticians.

Acceptance criteria:
- Every completed analysis contains structured finding/evidence/caveat text.
- Reviewer rejects incomplete summary payloads deterministically.

### P3 - Production execution, security, and UX maturity

- [ ] Migrate analysis runs to durable workers (Celery/Temporal/queue-backed design).
- [ ] Implement proper scheduled-task parsing and next-run logic.
- [ ] Add API rate-limiting and abuse protections.
- [ ] Add strict PII modes:
  - mask only
  - drop sensitive columns
  - audit report for every run.
- [ ] Add data lifecycle controls:
  - retention/TTL
  - deletion policy by org/project.
- [ ] Implement frontend guided workflow:
  - research-question wizard
  - assumption stoplights
  - spreadsheet-like column casting.
- [ ] Decide primary UI (React vs Streamlit) and avoid dual long-term divergence.

Acceptance criteria:
- Workloads survive API process restarts.
- Security posture includes explicit rate-limit + retention controls.
- Guided UX reduces wrong-route manual corrections.

## 5. MCP and A2A recommendation

### MCP (Model Context Protocol): Yes, phased adoption is worth it

Why it is worth adding:
- Clean way to expose enterprise context/tools to agents without embedding bespoke connectors in core workflow code.
- Good fit for this product's need for external context (protocol docs, templates, ontologies, data dictionaries, policy resources).

How to adopt safely:
- Start with read-mostly MCP servers:
  - statistical guideline server
  - internal metadata dictionary server
  - reporting template server.
- Keep decision engine deterministic; use MCP for context augmentation, not route authority.
- Require explicit auth/audit on every MCP tool call.

Recommendation:
- Add MCP after P0 stabilization, before large expansion of prompt complexity.

### A2A (Agent-to-Agent protocol): Defer for now

Why to defer:
- Current architecture is still a single workflow runtime with unresolved internal contracts.
- A2A adds value when agents are independently deployed services with stable contracts and cross-system orchestration needs.
- Introducing A2A now would increase failure modes before core correctness is locked.

When to revisit:
- After P1 and P2 are stable.
- After worker-based execution and clear agent API contracts exist.

Recommendation:
- Defer A2A; revisit in a later scaling phase.

## 6. Success metrics for this roadmap

- Reliability:
  - 0 failing tests on main branch.
  - No analysis-loss on API restart (after worker migration).
- Statistical quality:
  - >95% of completed analyses include effect size + CI payload.
  - Regression and outlier diagnostics available where applicable.
- AI quality:
  - Reviewer-adjusted summaries drop over time.
  - Hallucination flags tracked and trended.
- UX quality:
  - Lower manual reroute/override rate.
  - Faster time-to-first-useful-result from upload.
- Security/compliance:
  - Rate limits active.
  - PII policy mode visible per analysis run.
  - Retention policy auditable.

## 7. Source-of-truth policy

This file replaces all previous per-epic todo files in `docs/todo`.
From now on:
- Keep this file as the roadmap source of truth.
- Keep the four P0-P3 playbooks as implementation source of truth.
- Track progress in roadmap + playbooks, then propagate to release notes/changelogs.
