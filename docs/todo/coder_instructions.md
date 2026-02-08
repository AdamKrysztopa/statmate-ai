# Coder Instructions (Generic)

## Purpose
This file defines how to work on StatmateAI tasks safely and consistently. Follow these steps for every task.

---

## 1. Before you start

- Read the relevant playbook section in `docs/todo/`.
- Confirm the acceptance criteria for the task.
- Identify the exact files to change before editing.

---

## 2. Workflow checklist

1. Create a small, focused branch.
2. Make one logical change at a time.
3. Run the relevant tests after each change.
4. Update documentation if the behavior or API changes.
5. Keep commits small and descriptive.

---

## 3. Coding rules

- Match existing style and patterns.
- Avoid large refactors unless explicitly required.
- Prefer explicit, readable code over clever shortcuts.
- Add unit tests for new logic.
- Do not change public API responses without updating docs and tests.

---

## 4. Required checks

- `uv run python -m pytest -q`
- `uv run ruff check statmate tests frontend/src --output-format concise`

---

## 5. Definition of done

A task is done only when:
- Tests pass.
- Acceptance criteria are met.
- UI changes are visible (if applicable).
- Docs are updated for any user-facing change.

---

## 6. Where to look

- Roadmap: `docs/todo/master_ai_project_audit_and_roadmap.md`
- Finalization summary: `docs/todo/finalization_roadmap.md`
- Execution playbooks: `docs/todo/p0_stabilization_implementation_playbook.md` → `p3_production_scale_implementation_playbook.md`
- Junior task list: `TODO.md`
