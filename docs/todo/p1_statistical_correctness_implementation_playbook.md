# P1 Implementation Playbook: Statistical Correctness and Completeness

Date: 2026-02-07  
Priority: P1 (core product correctness/completeness)  
Goal: Make outputs statistically complete and schema-consistent for analyst workflows.

## 0. Scope

This playbook implements P1 from `docs/todo/master_ai_project_audit_and_roadmap.md`:
- effect size + CI standardization
- power interpretation for non-significant results
- regression module
- outlier/influence diagnostics
- survival path cleanup (implement or disable placeholder)

---

## P1-1. Standardize statistical result schema across tests

### Why

`StatTestResult` supports `effect_size_type` and `confidence_interval`, but core functions emit these inconsistently.

### Implementation steps

1. Define a canonical output contract for all inferential tests:
- required:
  - `test_name`, `statistics`, `p_value`, `null_hypothesis`, `statistical_test_results`
- optional but standardized:
  - `effect_size_type`
  - `confidence_interval` (as `[low, high]`)
  - `test_specifics.effect_size` object shape

2. Implement helper utilities for effect size/CI packaging.
- File: `statmate/statistical_core/base.py`
- New helper module: `statmate/statistical_core/effect_size.py` (recommended)

3. Update core modules to use shared helpers.
- `statmate/statistical_core/comparison.py`
- `statmate/statistical_core/anova.py`
- `statmate/statistical_core/categorical_comparison.py`
- `statmate/statistical_core/linear_correlation.py`

### Acceptance criteria

- [ ] All primary test functions emit consistent effect size/CI fields when computable.
- [ ] JSON response shape is stable across test families.

### How to test

Automated:
- Add schema consistency tests: `tests/test_result_schema_consistency.py`
- Run existing method coverage tests.

Manual:
- Run completed analyses and inspect API `results_detail` payload for consistent keys.

---

## P1-2. Add effect sizes and confidence intervals to comparison/ANOVA paths

### Why

Publication-quality interpretation requires magnitude and precision, not just p-values.

### Implementation steps

1. Two-group comparisons:
- Cohen's d for independent and paired variants.
- 95% CI for mean difference/effect estimate when feasible.
- File: `statmate/statistical_core/comparison.py`

2. Multi-group:
- Partial eta squared (parametric ANOVA).
- Non-parametric analog note and standardized reporting payload.
- File: `statmate/statistical_core/anova.py`

3. Ensure values flow through workflow and API response.
- `statmate/workflow/nodes.py`
- `statmate/api/services/analysis_service.py`

### Acceptance criteria

- [ ] t-test/Welch/paired paths return Cohen's d + CI when valid.
- [ ] ANOVA outputs include partial eta squared (or explicit not-computable reason).

### How to test

Automated:
- Extend `tests/test_statistical_methods_expansion.py`.
- Add deterministic numeric fixtures with known effect-size ranges.

Manual:
- Verify UI displays effect sizes and CIs in summary tables/exports.

---

## P1-3. Add power interpretation for non-significant outcomes

### Why

Users need to know whether non-significance is likely due to low power.

### Implementation steps

1. Add post-hoc power utility module (bounded scope).
- New file: `statmate/statistical_core/power.py`
- Implement minimal power estimation for:
  - two-group mean comparison
  - one-way ANOVA (if feasible for current stack)

2. Add power summary into `test_specifics` when p >= alpha.

3. Surface power caveat in summarization input.
- `statmate/workflow/nodes.py` (summary context assembly)

### Acceptance criteria

- [ ] Non-significant tests include power interpretation when computable.
- [ ] Summaries can reference underpowered outcomes without hallucination.

### How to test

Automated:
- New tests: `tests/test_power_interpretation.py`

Manual:
- Use tiny-N dataset where effect exists but p > 0.05; confirm power warning appears.

---

## P1-4. Implement regression module and workflow integration

### Why

Regression is a core missing capability and currently blocks broader research use cases.

### Implementation steps

1. Add regression core module.
- New file: `statmate/statistical_core/regression.py`
- Include:
  - simple linear regression
  - multiple linear regression (+ VIF diagnostics)
  - logistic regression (+ basic fit diagnostics)

2. Add agents for regression paths.
- New file(s): `statmate/agents/regression_agents.py`

3. Add routing hooks and node names.
- `statmate/core/config.py`
- `statmate/workflow/edges.py`
- `statmate/workflow/graph_builder.py`
- `statmate/workflow/nodes.py`

4. Add API/UI exposure for regression outputs.
- `statmate/api/models/analysis.py`
- `frontend/src/api/client.ts`
- `frontend/src/App.tsx`

### Acceptance criteria

- [ ] Regression analyses run end-to-end and appear in results.
- [ ] VIF/high-collinearity flags are surfaced for multiple regression.
- [ ] Logistic regression path handles binary target and reports fit diagnostics.

### How to test

Automated:
- New tests: `tests/test_regression_methods.py`
- Route tests in `tests/test_decision_engine.py` for regression-triggering profiles.

Manual:
- Upload known regression dataset and validate coefficients/directions sanity.

---

## P1-5. Add outlier/influence diagnostics

### Why

Model conclusions can be distorted by high-leverage points.

### Implementation steps

1. Add diagnostics:
- Cook's distance
- leverage
- simple outlier flags (z-score or robust alternatives)
- File: `statmate/statistical_core/regression.py` (or separate diagnostics module)

2. Include diagnostic summary in blueprint/assumption logs.
- `statmate/workflow/blueprint.py`
- `statmate/workflow/nodes.py`

3. Ensure summaries include caveat when influence is high.

### Acceptance criteria

- [ ] Influential point diagnostics are computed for regression-capable runs.
- [ ] Diagnostics appear in `assumption_log` or structured metadata.

### How to test

Automated:
- Add synthetic dataset with one extreme leverage point and verify detection.

Manual:
- Confirm exported report includes influence caveat.

---

## P1-6. Resolve survival placeholder path

### Why

Current route can point to `cox_regression` placeholder, which is misleading.

### Implementation steps

1. Choose one:
- implement minimally viable survival analysis node, or
- disable routing to placeholder and emit explicit "not yet supported".

2. Update route metadata and UI labels accordingly.
- `statmate/core/config.py`
- `statmate/workflow/edges.py`
- `statmate/workflow/nodes.py`
- `statmate/workflow/graph_metadata.py`

### Acceptance criteria

- [ ] No silent placeholder behavior in production route.
- [ ] API response clearly indicates implemented or unsupported state.

### How to test

Automated:
- Survival route unit tests in `tests/test_decision_engine.py`.

Manual:
- Submit survival-like dataset and verify deterministic behavior.

---

## 2. Definition of done for P1

- [ ] Statistical payloads are consistent and complete for key methods.
- [ ] Regression and influence diagnostics exist end-to-end.
- [ ] No placeholder survival route ambiguity.

