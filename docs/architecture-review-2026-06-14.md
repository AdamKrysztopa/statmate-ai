# Architecture review — StatmateAI

- Date: 2026-06-14
- Reviewer: arch-crew / decide-architecture (Mode B: refactor review)
- Branch: feat/productionalization-hardening

## Current shape

| Axis | What's actually there |
|------|-----------------------|
| **Structure** | **Hexagonal / Clean core** (de facto). `statistical_core/` is a pure numeric ring (imports only scipy/statsmodels/sklearn + its own `core`), `agents/` are adapter wrappers around it, `workflow/` orchestrates, `api/` is the outer ring. Dependencies point inward — clean. |
| **Topology** | **Modular Monolith** — one deployable (FastAPI :8000 + Streamlit :8501), one shared SQLite/Postgres DB. Right-sized. |
| **Data / ML** | **Pipe-and-Filter pipeline + Blackboard** — the LangGraph state machine (`graph_builder.py`) is a staged transform chain; `WorkflowState` is the shared blackboard that nodes/agents converge on. This is the heart of the system and is well-modelled. |
| **Presentation** | **MVC-ish, single team** — Streamlit primary UI + a thin React/Vite client. n/a for splitting. |
| **Background work** | APScheduler in-process (`api/scheduler/`) for recurring runs. Fine at this scale. |
| **Volatile seam** | Multi-provider LLM (OpenAI/Anthropic/Google/Groq/Ollama) — isolated, but across **three** modules (see Finding 3). |

**Verdict up front: the architecture is sound and right-sized.** The composition (hexagonal
core inside a modular monolith with a pipeline/blackboard engine) genuinely fits a rule-heavy
statistical domain with a swappable LLM dependency. **No re-architecture is warranted.** Every
finding below is targeted cleanup of *aesthetic/coupling debt*, not structural pain — ordered by
value, each safely incremental.

---

## Findings

### 1. Dead/speculative abstraction layer in `core/base_interfaces.py` — and a name collision
`core/base_interfaces.py:1` declares ABCs `StatMateAgent`, `WorkflowNode`, `DataTransformer`,
`DataValidator` and exports them from `core/__init__.py`. **None are subclassed anywhere.** The
real abstractions in use are `StatTestDeps`/`AgentResult` (`agents/agent_builder.py:80`), plain
LangGraph node functions, and a *different* `WorkflowNode` **dataclass** in
`workflow/graph_metadata.py:21`. So `WorkflowNode` is two unrelated classes with one name.
`core/__init__.py:13` even carries a comment: *"Now import base_interfaces, which triggers the cycle"* — a known import cycle.

- **Impact:** Speculative architecture (catalog: "least architecture" violation). Misleads readers
  into thinking there's a plugin/ABC contract that doesn't exist; the name collision actively
  confuses; the import cycle is a latent fragility.
- **Move:** Delete the unused ABCs, or collapse `StatisticalTest` (the one `Protocol` that *does*
  describe a real contract) down to where it's used. Keep `base_interfaces.py` only for live contracts.
- **First step:** `grep` confirms zero subclasses → delete `StatMateAgent`, `WorkflowNode`,
  `DataTransformer`, `DataValidator` from `base_interfaces.py` and their `__init__` exports. Run tests.
- **Cost:** ~1 hour. Removes the import cycle as a side effect.

### 2. `AnalysisService.run_analysis` is a god method (the seam is overloaded)
`api/services/analysis_service.py` (732 lines). `run_analysis` (`:314`) nests **seven** closures —
`_persist_decision_steps`, `_persist_assumption_log`, `_persist_intermediate_log`,
`_on_state_update`, `_record_system_step`, `_run_workflow`, `_normalize_steps`. The one method
mixes orchestration, DB persistence, JSON sanitization, retry/backoff, and model-config assembly.

- **Impact:** Genuine maintainability pain — the central web↔engine seam is the hardest file to
  change or test in isolation. This is the one finding touching *real* fragility, not aesthetics.
- **Move:** Extract collaborators (layered/vertical-slice hygiene, not a new axis):
  a `RunProgressRecorder` (the persist/record closures → a class taking `db` + `analysis_id`),
  and a `ResultSerializer` (the module-level `_sanitize_for_json` / `_clean_probabilities` /
  `_build_test_hierarchy`). `run_analysis` then reads as: build config → run workflow → record → persist.
- **First step:** Pull the four `_persist_*`/`_record_*` closures into a `RunProgressRecorder`
  class in a new `api/services/run_progress.py`; inject it into `run_analysis`. Tests stay green.
- **Cost:** ~half a day; do it in 3–4 tiny commits (one collaborator at a time).

### 3. LLM provider logic is triplicated across three modules
Three overlapping layers describe the same volatile seam:
`core/model_config.py` (`MultiModelConfig`, `ModelInfo`, `ModelProvider`),
`core/model_provider.py` (`ModelProviderSystem` — `create_model`, `list_available_models`,
`get_model_info`, backoff), and `workflow/model_factory.py` (`ModelFactory` — *also*
`create_model`, `list_available_models`, `get_model_info`, plus a global singleton). Two of the
three expose the same method surface, and the lowest adapter (`model_factory`) lives under
`workflow/` rather than with the rest of the seam in `core/`.

- **Impact:** Confusion about which entry point to call (catalog: leaked/spread port). Not painful
  yet, but it's the seam most likely to grow as providers are added.
- **Move:** Make this **one explicit Hexagonal port** (catalog `#p-rings`, Q10). Keep
  `model_config` as config DTOs; merge `ModelProviderSystem` + `ModelFactory` into a single
  adapter (`core/model_provider.py`), and have `workflow/model_factory.py` become a thin
  re-export shim for backward-compat imports.
- **First step:** Move `ModelFactory` into `core/` next to `ModelProviderSystem`; collapse the
  duplicated `create_model`/`list_available_models`/`get_model_info` into one. Leave a re-export
  at the old path so `api/` and `agents/` imports don't break.
- **Cost:** ~half a day. Pure consolidation; no behavior change.

### 4. Two oversized core files (`validation.py` 897, `model_config.py` 558)
`core/validation.py:1` mixes the `StatisticalDesign` domain model with many validators;
`core/model_config.py` mixes enums, `ModelInfo`, the big static model registry, and config
classes.

- **Impact:** Navigation friction only — no coupling violation. Lowest priority.
- **Move:** Split `validation.py` into `design.py` (the `StatisticalDesign` model) + `validators.py`;
  split the static model catalog out of `model_config.py` into a `model_registry.py` data module.
- **First step:** Extract `StatisticalDesign` to its own module; re-export from `validation.py`.
- **Cost:** ~2 hours. Do only if these files are actively edited.

### 5. One inward-pointing config leak (minor)
`workflow/graph_builder.py:450` does `from config.settings import settings` — an inner ring
reaching out to app/deployment config. The only such leak found (statistical_core and agents are
clean).

- **Impact:** Tiny; breaks the otherwise-clean dependency direction in one spot.
- **Move:** Pass the needed setting into `build_workflow_graph(...)` as a parameter from the API
  startup, rather than importing `settings` inside the builder.
- **First step:** Add a typed parameter to the builder; have `api/main.py` supply it.
- **Cost:** ~1 hour.

---

## Recommended order (step-by-step)

1. **Finding 1** — delete dead ABCs + fix the name collision/import cycle. *Cheapest, removes confusion.*
2. **Finding 3** — consolidate the LLM port into `core/`. *Highest structural clarity gain.*
3. **Finding 2** — decompose `run_analysis` via `RunProgressRecorder` + `ResultSerializer`. *Biggest real-maintainability win; do in tiny commits.*
4. **Finding 5** — remove the `config.settings` leak from `graph_builder`. *Quick, restores clean rings.*
5. **Finding 4** — split the two large core files. *Only if/when you're editing them anyway.*

Each step is independently shippable and test-covered by the existing `tests/` suite — no big-bang
refactor, no Strangler Fig needed.

## Leave alone (explicitly)

- **The two-layer `statistical_core` ↔ `agents` pattern.** This is the codebase's best architectural
  decision — pure numeric core, LLM adapter on top. Don't touch it.
- **The LangGraph pipeline/blackboard engine** (`workflow/`). Already decomposed into per-phase
  `nodes_*.py` modules with `edges.py` routing; right-sized for the domain.
- **Modular-monolith topology.** Do **not** split into services — there's no org-size or scaling
  forcing function. A shared DB + one deployable is correct here.
- **The two-config split** (`config/settings.py` vs `core/config.py`). Deliberate and documented;
  app config vs statistical thresholds are genuinely different concerns.
- **The dual frontend** (Streamlit + React). Fine as a V1/transition choice.

---

### The least-architecture check
Nothing here adds an axis. Two findings (1, 3) *remove* speculative/spread structure; the rest are
file-hygiene. The system already sits at the right amount of architecture for what it is — these
moves just tighten what's there.
