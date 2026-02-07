# P2 Implementation Playbook: Medical Reporting and Reviewer Quality

Date: 2026-02-07  
Priority: P2 (medical publication quality)  
Goal: Produce non-hallucinated, publication-ready summaries with explicit evidence and caveats.

## 0. Scope

This playbook implements P2 from `docs/todo/master_ai_project_audit_and_roadmap.md`:
- finding/evidence/caveat summary structure
- CI + effect-size reporting enforcement
- clinical significance judgment
- p-hacking and multiple-comparison reviewer checks
- non-statistician chart narrative captions

---

## P2-1. Refactor summarizer to structured medical narrative

### Why

Current summarizer is constrained against hallucination but not shaped for medical reporting standards.

### Implementation steps

1. Extend summarizer result schema.
- File: `statmate/agents/summarizer_agent.py`
- Add structured fields:
  - `finding`
  - `evidence`
  - `caveat`
  - keep `summary` (composed) for backward compatibility.

2. Update summarizer prompt contract:
- enforce inclusion of p-values, CI, effect size when available
- forbid claims beyond `performed_tests` and raw outputs

3. Compose display-ready summary in workflow node.
- File: `statmate/workflow/nodes.py` (`summariser_node`)

### Acceptance criteria

- [ ] Every generated summary includes finding/evidence/caveat fields.
- [ ] Existing clients still receive `summary` text.

### How to test

Automated:
- Add tests: `tests/test_summarizer_structure.py`
- Validate JSON schema and required fields.

Manual:
- Run sample analyses and verify UI displays all three sections.

---

## P2-2. Enforce CI/effect-size mention policy

### Why

Medical readers need estimate precision and effect magnitude, not just significance statements.

### Implementation steps

1. Add validator in summary pipeline:
- If CI/effect-size exists in inputs, summary must include it.
- File: `statmate/workflow/nodes.py` (post-summary check or reviewer check)

2. Add fallback text rule:
- If unavailable, include explicit reason (e.g., insufficient sample or unsupported metric).

### Acceptance criteria

- [ ] Summary includes CI/effect-size references whenever available.
- [ ] Missing metrics are explained explicitly, not silently omitted.

### How to test

Automated:
- New tests for both positive and missing-metric branches.

Manual:
- Inspect exported report for CI/effect size consistency.

---

## P2-3. Add clinical significance assessment block

### Why

Statistical significance and clinical relevance are not equivalent.

### Implementation steps

1. Introduce clinical threshold config support.
- File: `config/settings.py` or a dedicated clinical config module.
- Optionally include per-outcome thresholds in analysis configuration.

2. Add summarizer/reviewer fields:
- `clinically_meaningful: bool | null`
- `clinical_rationale: str`

3. In workflow summary context, pass threshold metadata when available.
- `statmate/api/models/analysis.py` (`configuration`)
- `statmate/workflow/nodes.py`

### Acceptance criteria

- [ ] Output includes explicit clinical meaningfulness judgment when threshold data exists.
- [ ] If threshold absent, output explicitly says "not evaluated".

### How to test

Automated:
- Tests with and without clinical thresholds.

Manual:
- Run two scenarios:
  - statistically significant but small effect below threshold
  - statistically non-significant but clinically relevant CI range.

---

## P2-4. Add reviewer checks for multiplicity and p-hacking signals

### Why

Reviewer should proactively flag suspicious inference patterns.

### Implementation steps

1. Expand reviewer schema.
- File: `statmate/agents/reviewer_agent.py`
- Add flags:
  - `multiple_comparison_risk`
  - `multiplicity_adjustment_missing`
  - `selective_reporting_risk`

2. Enrich reviewer input context:
- number of tests run
- whether correction was applied
- exploratory vs confirmatory mode if present
- File: `statmate/workflow/nodes.py` (`reviewer_node` deps payload)

3. Add deterministic pre-check for simple multiplicity heuristics before LLM reviewer.

### Acceptance criteria

- [ ] Reviewer output includes structured multiplicity checks.
- [ ] Hallucination and methodological risks are separated into clear flag categories.

### How to test

Automated:
- New tests: `tests/test_reviewer_multiplicity.py`

Manual:
- Simulate many-comparison run and verify warning appears.

---

## P2-5. Add narrative chart captions for non-statisticians

### Why

Plots currently have generic descriptions, not interpretive narrative suitable for broad audiences.

### Implementation steps

1. Add caption generation logic in visualization service.
- File: `statmate/api/services/visualization_service.py`
- For each plot type, generate:
  - plain-language trend sentence
  - caveat sentence where applicable

2. Ensure captions propagate through exports and API.
- File: `statmate/api/services/export_service.py`
- File: `statmate/api/models/analysis.py` (if schema extension needed)

### Acceptance criteria

- [ ] Each chart payload has an interpretive caption.
- [ ] Captions appear in PDF/DOCX/HTML outputs.

### How to test

Automated:
- Add tests asserting caption presence per plot type.

Manual:
- Generate report and confirm caption quality in UI and exports.

---

## 2. Definition of done for P2

- [ ] Summaries are structured and publication-oriented.
- [ ] Clinical significance and methodological caveats are explicit.
- [ ] Reviewer flags are richer and auditable.
- [ ] Plot narratives are understandable for non-experts.

