# Consolidated refactoring & fix plan (MoSCoW) — StatmateAI

- Date: 2026-06-14
- Merges three audits into one prioritised plan:
  - `docs/agentic-review-2026-06-14.md` (agentic patterns) → items tagged **AG#**
  - `docs/architecture-review-2026-06-14.md` (architecture) → items tagged **AR#**
  - `docs/design-pattern-review-2026-06-14.md` (object-level patterns) → items tagged **DP#**
- Companion ADR: `docs/adr/0001-table-driven-construction-and-injected-model-factory.md`

## How the audits overlap (deduplicated)

Several findings are the same issue from different angles — merged into **workstreams**:

| Workstream | Source findings | Net change |
|------------|-----------------|------------|
| **W-LEAF** — leaf test nodes | AG1 (compute-then-narrate) · AG2 (collapse wrappers, cache models) · DP2 (node-wrapper boilerplate) | Compute first, narrate with a single no-tool LLM call, build nodes from a data table. |
| **W-MODEL** — LLM model layer | AR3 (consolidate port into `core/`) · DP1 (provider registry dict) · DP3 (inject factory, kill global) | One LLM port in `core/`, provider table, injected factory. |
| **W-GUARD** — runtime guardrails | AG7 (tracing) · AG3 (loop cap) · AG5 (dead checkpointer) | Observability + recursion cap + resolve false durability. |
| **W-TIDY** — code hygiene | AR1 (dead ABCs) · AR5 (settings leak) · AR2 (god method) · AR4 (big files) · DP4 (dup string) | Local cleanups, no behaviour change. |

Overall verdict from all three reviews agrees: **the architecture is sound and right-sized — the
fixes subtract and consolidate, they do not add structure.**

---

## MUST — correctness, safety, production-blocking

> Without these the system is unsafe or undiagnosable in production. Each is cheap-to-medium and
> reversible.

### M1. Add observability/tracing (AG7) — *additive, do first*
No `logfire`/`langsmith`/`otel` anywhere. For a clinical tool spanning 5 LLM providers, run failures
and cost regressions are undiagnosable. Pydantic-AI ships first-class Logfire:
`logfire.configure()` + `logfire.instrument_pydantic_ai()`, a span per graph node, gated behind a
`Settings` flag. Zero behaviour change — ship it first so later steps are measurable.
**Effort: S.**

### M2. Cap the loop & make reconciliation terminal (AG3)
`stream_config` (`statmate_flow_refactored.py:113`) sets no `recursion_limit`; the
`DESIGN_RECONCILIATION ⇄ decide_outcome` path (`edges.py:292-341`) can cycle to an opaque
`GraphRecursionError`. Add an explicit `recursion_limit`; add a `reconciliation_attempted` flag on
`WorkflowState` and fall through to `USER_INTERVENTION` on the second pass. **Effort: S.**

### M3. Inject the model factory; remove the mutable global (DP3, part of W-MODEL)
`workflow/model_factory.py` reassigns a process-global `default_factory` on **every run** via
`initialize_default_factory(...)`. Concurrent API analyses with different model configs race on it.
Build the factory once, pass it via DI; keep a build-once module default for back-compat imports;
drop the per-run `global` reassignment. **Effort: M.** *(Correctness — the one race hazard.)*

### M4. Compute-then-narrate: stop paying for a discarded agent loop (AG1, core of W-LEAF)
Every leaf runs a full Pydantic-AI agent with a `run_test` tool and `retries=3`, then
`_node_helpers.py:117-134` **recomputes the statistic and overwrites the LLM result**. The tool loop
(tokens, latency, 3× failure surface) is thrown away. Fix: in `call_test_agent`, compute the
`StatTestResult` *first*, pass it to a **no-tool** narrator that returns only `result`/`comments`,
delete the override block. Numbers are byte-identical (already overridden); only narration *input*
changes. **Effort: M.** *(Do after M1 so the token/latency drop is measurable.)*

---

## SHOULD — strong pain, false guarantees, high-value cleanup

> Important; omitting them leaves real debt, but none blocks a safe release.

### S1. Resolve the dead checkpointer (AG5)
`SqliteSaver` is wired but `WorkflowState`/DataFrames aren't msgpack-serializable, so the first
checkpoint write throws and the code silently reruns with no checkpointer — a false durability
guarantee plus a wasted first attempt on failures. **Decide:** (a) remove the checkpointer +
`thread_id` plumbing (conservative — runs are short/single-window), or (b) make it real by storing
the DataFrame via the existing parquet `storage_service` and checkpointing references only. **Effort:
S (remove) / M (make real).**

### S2. Consolidate the LLM port into `core/` with a provider registry (AR3 + DP1, rest of W-MODEL)
Provider logic is triplicated (`core/model_config.py`, `core/model_provider.py`,
`workflow/model_factory.py`) and `model_provider.create_model` is a 5-arm `if provider == …` chain
into five near-identical `_create_*_model` methods. Merge `ModelProviderSystem` + `ModelFactory`
into one adapter in `core/`; replace the branches with a `dict {provider: (ProviderCls, ModelCls)}`
+ one `_build_model`; leave a re-export shim at the old path. **Pairs with M3 — same files. Effort:
M.**

### S3. Decompose `AnalysisService.run_analysis` (AR2)
732-line file; `run_analysis` nests seven closures mixing orchestration, DB persistence, JSON
sanitization, retry, model-config assembly. Extract a `RunProgressRecorder` (the `_persist_*` /
`_record_*` closures) and a `ResultSerializer` (`_sanitize_for_json` / `_clean_probabilities` /
`_build_test_hierarchy`). The central web↔engine seam becomes testable. **Effort: M — do in 3–4 tiny
commits.**

### S4. Collapse leaf nodes to one parameterized narrator + data table (AG2 + DP2, rest of W-LEAF)
After M4, the 10+ identical `*_wrapper` closures in `graph_builder.py:203-346` collapse to one
`make_test_node(agent_builder, probability_key)` driven by a `_TEST_NODES` table; build
model/settings once per run instead of rebuilding in each wrapper. **Depends on M4. Effort: S.**

---

## COULD — low-risk tidy-ups, do when touching the file (W-TIDY)

### C1. Delete dead abstractions + fix the name collision (AR1)
`core/base_interfaces.py` ABCs `StatMateAgent`, `WorkflowNode`, `DataTransformer`, `DataValidator`
are never subclassed; `WorkflowNode` collides with a real dataclass in `graph_metadata.py`; the
import is flagged as triggering a cycle. Delete the dead ABCs + exports (removes the cycle). **S.**

### C2. Hoist the duplicated narration string (DP4)
`'Please suggest the best way to perform the test…'` is pasted into every `*_agent` factory. Hoist to
one `DEFAULT_TEST_SUGGESTION` constant. *(Mostly absorbed by M4/S4 if leaves are rebuilt.)* **S.**

### C3. Remove the `config.settings` leak from `graph_builder` (AR5)
`graph_builder.py:450` imports `config.settings` inside the inner ring. Pass the needed value in as a
parameter from API startup. **S.**

### C4. Split the oversized core files (AR4)
`core/validation.py` (897) → `design.py` + `validators.py`; `core/model_config.py` (558) → split the
static model registry into `model_registry.py`. Navigation only; do only when editing them. **S.**

---

## WON'T (this round) — confirmed sound, do not touch

All three audits independently flagged these as correct:

- **Deterministic `DecisionEngine` router** (`edges.py`) — keep routing deterministic; do **not** move
  it into an LLM. (AG4/AG keep)
- **Two-layer split** `statistical_core/` (numeric) ↔ `agents/` (narration) — the best decision here.
- **Integrator-style validation** of `column_role_agent` output — keep.
- **No HITL, no multi-agent** — correctly absent; do not add. (AG6)
- **`edges.py` `decide_*` guard chains** — encode a genuine multi-attribute decision tree; a dict
  dispatch would obscure them. Leave linear. (DP)
- **`get_db` context manager, `WorkflowState` Pydantic model, fluent graph builder** — idiomatic. (DP)
- **Modular-monolith topology, shared DB, two-config split, dual frontend** — right-sized; no
  microservices, no extra axes. (AR)

---

## Suggested execution order

Guardrails first (measurable, low risk), then the simplification, then tidy-ups:

1. **M1** (tracing) — additive; makes everything after it measurable.
2. **M2** (loop cap) — cheap crash-safety.
3. **M3 + S2** (W-MODEL together) — inject factory + provider registry + port consolidation; one
   pass over the model modules. Pairs with arch-review AR3 and ADR-0001.
4. **M4 → S4** (W-LEAF, in order) — compute-then-narrate, then collapse wrappers to a table; absorbs
   C2.
5. **S1** (checkpointer decision) — independent; remove or make real.
6. **S3** (decompose `run_analysis`) — independent; tiny commits.
7. **C1, C3, C4** (W-TIDY) — opportunistic, when next editing those files.

Every item is internal (no public-API/behaviour change beyond M4's narration input), independently
shippable, and covered by the existing `tests/` suite.
