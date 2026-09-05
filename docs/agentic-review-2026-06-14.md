# Agentic system review — StatmateAI

- Date: 2026-06-14
- Reviewer: agentic-patterns audit (arch-crew)
- Scope: `statmate/workflow/`, `statmate/agents/`, `statmate/statistical_core/`, the LangGraph engine and its LLM seams.

---

## Closest architecture

StatmateAI is **not an autonomous agent system** — and that is the right call. It is a
**deterministic Routing workflow** (`#ap-routing`) whose leaf nodes are **single-purpose LLM
narration calls** (`#ap-prompt-chaining` degenerate to one call) over deterministic SciPy/statsmodels
computations, closed by a short **deterministic-auditor + LLM-reviewer** tail
(a Reflection/Evaluator shape, `#ap-reflection` / `#ap-evaluator-optimizer`).

Mapped to the autonomy spectrum:

| Layer | What's actually there | Pattern |
|-------|----------------------|---------|
| Autonomy | A dataset is classified by data type / design / assumptions and dispatched to the matching test. Steps are developer-laid-out in `graph_builder.py`, not model-chosen. | **Workflow → Routing** (`#ap-routing`) |
| Router brain | `DecisionEngine` in `edges.py` — registry + hard-constraint filtering. 100% deterministic. | **Routing classifier** (deterministic — ideal) |
| Pre-flight | `column_role_agent` (LLM) wrapped in heavy deterministic validation + fallbacks. | **Integrator** (`#ap-integrator`) — done well |
| Leaf test nodes | `call_test_agent` runs a Pydantic-AI `Agent` with a `run_test` tool… then **recomputes the test deterministically and overwrites the LLM result**. LLM output kept = narration only. | **Tool Use** present but **vestigial** |
| Closing tail | `summariser_node` (LLM) → `MethodologyAuditor` (deterministic re-route check) → `reviewer_node` (LLM). | **Evaluator/Reflection** (`#ap-evaluator-optimizer`) |
| Memory | `SqliteSaver` checkpointer wired in… but it throws on `WorkflowState`/DataFrame serialization and **silently falls back to no checkpointer**. | **Disabled checkpointer** |
| Governance | No irreversible external actions (computes stats, writes results to its own DB). `CHOICE` is a soft user-in-the-loop that auto-resolves to default in batch. | HITL not required |
| Observability | None found (no logfire / langsmith / otel / instrumentation). | **Missing** |

**Headline:** the architecture is sound and appropriately *low-autonomy* for the task. The problems
are not "needs more agents" — they are **agent machinery carried where a single call would do, two
features that are wired but dead, and no tracing.** The fixes subtract, they don't add.

---

## Current design (one line)

Autonomy: **Routing workflow** (deterministic engine) · Loop: per-test **single LLM narration**
(tool result discarded) · Agents: **~15 structurally-identical leaf narrators + 3 LLM aux agents**,
no true multi-agent collaboration · Memory: **checkpointer present but broken/disabled** · Governance:
**no irreversible actions; soft CHOICE gate** · Observability: **none**.

---

## Findings (seven-defect checklist, translated to "this is a workflow")

### 1. Agent machinery where a single call would do — **HIGH** *(the headline)*
- **Where:** `statmate/agents/agent_builder.py:140-157` (the `run_test` agent tool) + `statmate/workflow/_node_helpers.py:106-134`.
- **What:** Every leaf node runs a full Pydantic-AI `Agent` with a `run_test` tool and `retries=3`,
  then `_node_helpers.py:117-134` **unconditionally recomputes the statistic with the raw function
  and overwrites `result.statistical_test_result`.** So the LLM's tool call (and the tokens spent
  letting it "decide" to call the tool) is thrown away. The LLM's *only* surviving contribution is
  the `result` + `comments` narration strings.
- **Impact:** You pay agentic tool-calling latency, token cost, and a 3-retry failure surface per
  test, for an output you discard. The numbers are already trustworthy *because* you recompute them —
  the agent loop adds risk (malformed tool calls, refusals) with no upside.
- **Fix:** Demote each leaf from an **agent-with-tools** to a **single structured LLM call** that
  receives the already-computed `StatTestResult` and returns only narration (`result`, `comments`).
  Drop the `run_test` tool and the post-hoc override entirely — compute first, narrate second.
- **First step:** In `call_test_agent`, compute the `StatTestResult` *before* the LLM call; pass it
  into a no-tool narration agent; delete the override block. No graph changes.

### 2. Multi-agent where one agent would do — **LOW** (already effectively one)
- **Where:** `statmate/agents/comparison_agents.py`, `anova_agents.py`, `categorical_comparison_agent.py`, all via `build_stat_test_agent`.
- **What:** ~15 "agents" exist, but they are not collaborating peers — they are leaf handlers of a
  router, and all share one builder. This is the *correct* shape (a router with specialized handlers),
  **not** a multi-agent topology. The only smell is that each `*_wrapper` in `graph_builder.py`
  re-instantiates the model + settings + agent on every node call (`graph_builder.py:203-346`).
- **Impact:** Minor — redundant object construction per run; conceptual noise ("agents" implies
  autonomy that isn't there).
- **Fix:** After finding #1, this collapses to **one parameterized narrator** invoked per test. Cache
  model/settings per run instead of rebuilding in each wrapper.

### 3. Loops without a step budget / early-exit — **MEDIUM**
- **Where:** `statmate/workflow/statmate_flow_refactored.py:113-120` (`stream_config` sets only
  `thread_id`, never `recursion_limit`) and the `DESIGN_RECONCILIATION ⇄ decide_outcome` path
  (`edges.py:292-341`).
- **What:** The graph is mostly a DAG, but `evaluate_routing` can return `DESIGN_RECONCILIATION`
  (`edges.py:334-341`, `paired_data_requires_paired_node`) and reconciliation routes back through
  `decide_outcome` (`graph_builder.py:110-131`). If reconciliation fails to flip the condition, this
  can cycle until LangGraph's *default* `recursion_limit` (25) throws `GraphRecursionError`.
- **Impact:** Rare, but the failure is an opaque recursion crash rather than a clean fallback.
- **Fix:** Set an explicit `recursion_limit` in `stream_config`, and have the reconciliation→routing
  branch fall through to `USER_INTERVENTION`/`DESCRIPTIVE_SUMMARY` after one reconciliation attempt
  (track a `reconciliation_attempted` flag on state) instead of being able to re-enter.

### 4. Unvalidated tool / observation outputs — **SOUND**
- The deterministic core returns typed `StatTestResult`; `column_role_agent.py:112-156` validates and
  defaults every LLM field (Integrator pattern, `#ap-integrator`); routing reads a typed
  `DataBlueprint` snapshot, not raw model text. This is a strength — keep it.

### 5. Durable memory when runs outlive the window — **MEDIUM (dead feature)**
- **Where:** `graph_builder.py:445-469` (SqliteSaver checkpointer) + `statmate_flow_refactored.py:134-147`.
- **What:** The checkpointer is constructed and passed in, but `WorkflowState`/DataFrames aren't
  msgpack-serializable, so the first checkpoint write raises and the code **rebuilds the graph with
  no checkpointer and reruns.** Result: resume-after-crash *appears* supported but never works, and
  every failing run silently pays for one aborted attempt.
- **Impact:** False durability guarantee; wasted first attempt on the failure path; `thread_id` plumbing implies a resume capability that doesn't exist.
- **Fix (pick one):** Either (a) **remove** the checkpointer and `thread_id` plumbing to stop
  advertising a capability you don't have, or (b) make it real — store the DataFrame out-of-band
  (you already persist datasets as parquet) and checkpoint only lightweight references + scalar state
  with a custom serializer. Given runs are short and single-window, **(a) is the conservative choice**
  unless long-running/resumable analyses are on the roadmap.

### 6. Irreversible actions gated by HITL — **N/A (correctly)**
- The workflow takes no outward irreversible action; it writes results to its own DB. `CHOICE`
  (`nodes_routing.py`) is a soft user-in-the-loop that auto-resolves to the registry default in batch,
  which is appropriate. No HITL gate needed. Don't add one.

### 7. Tracing / observability — **HIGH**
- **Where:** entire workflow; grep for `logfire|langsmith|opentelemetry|instrument` returns nothing.
- **What:** No per-run trace of LLM calls, tokens, cost, tool latency, or routing decisions. The
  `execution_trace`/`add_step` log is a *product* artifact for the UI, not an engineering trace — it
  won't tell you why a model refused, how many tokens a run burned, or which provider was slow.
- **Impact:** For a clinical/observational-research tool spanning 5 LLM providers, run failures and
  cost regressions are effectively undiagnosable in production.
- **Fix:** Pydantic-AI has first-class **Logfire** integration — `logfire.configure()` +
  `logfire.instrument_pydantic_ai()` traces every agent call with tokens/latency for near-zero code.
  Add a span around each graph node. Gate it behind a settings flag.

---

## Simplify (subtract)

1. **Collapse the per-test agents into one narration call** (finding #1) — biggest win: removes the
   discarded tool loop, the 3× retry surface, and ~15 near-identical agent definitions.
2. **Resolve the dead checkpointer** (finding #5) — remove it, or make it real; don't ship a
   capability that silently no-ops.
3. **Stop rebuilding model/agent objects in every `*_wrapper`** (finding #2) — build once per run.

## Sound as-is (keep)

- The **deterministic `DecisionEngine`** as the router brain — correct, testable, the right amount of
  autonomy. Do not move routing into an LLM.
- The **two-layer split** (`statistical_core/` deterministic ↔ `agents/` narration).
- The **Integrator-style validation** of `column_role_agent` output.
- **No HITL / no multi-agent** — both correctly absent.

---

## Step-by-step refactor plan (tiny, independently shippable)

Ordered for safety: each step is reversible and leaves the system green.

1. **Add tracing (additive, zero behavior change).** Wire Logfire behind a `Settings` flag;
   `instrument_pydantic_ai()`; wrap each node in a span. Ship. *(Finding #7.)*
2. **Cap the loop.** Add explicit `recursion_limit` to `stream_config` in
   `statmate_flow_refactored.py`. Add a `reconciliation_attempted` flag on `WorkflowState` and make
   the reconciliation→routing branch fall through to `USER_INTERVENTION` on the second pass.
   *(Finding #3.)*
3. **Decide the checkpointer.** If resumable runs aren't on the roadmap: delete the SqliteSaver path
   and `thread_id` plumbing. If they are: store the DataFrame via the existing parquet
   `storage_service` and checkpoint references only. *(Finding #5.)*
4. **Compute-then-narrate (the core change).** In `call_test_agent`: compute the `StatTestResult`
   first; pass it to the narrator; delete the post-hoc override (`_node_helpers.py:117-134`). Behavior
   for numbers is identical (you already override); only the narration's *input* changes. Verify with
   existing tests. *(Finding #1.)*
5. **Demote the agent to a single call.** Replace `build_stat_test_agent`'s `run_test` tool with a
   no-tool structured-output narrator that takes the precomputed result. Drop `retries=3` down to 1
   (idempotent narration). *(Finding #1.)*
6. **Collapse to one parameterized narrator + cache models.** Replace the ~15 `*_wrapper` closures in
   `graph_builder.py` with a single factory bound to the test name; build model/settings once per run
   and reuse. *(Finding #2.)*

Steps 1–3 are guardrails (do first, low risk). Steps 4–6 are the simplification (do after tracing is
in place so you can measure the token/latency drop).

---

*If the compute-then-narrate direction (steps 4–6) is one you commit to, it's worth capturing as an
ADR under `docs/adr/` so the rationale for demoting the per-test agents is recorded.*
