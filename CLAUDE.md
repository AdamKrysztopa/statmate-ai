# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

StatmateAI automates statistical-test selection and execution for clinical/observational
research. The core is a **LangGraph state machine** that routes a dataset through
assumption checks to the appropriate statistical test, where **Pydantic AI agents** wrap
each SciPy/statsmodels test and produce structured, LLM-narrated results. A FastAPI backend
and a Streamlit UI sit on top.

## Commands

Tooling is `uv`. Most workflows go through the `Makefile`; run `make help` for the full list.

```bash
make install-dev        # uv sync --all-extras (dev deps: ruff, pyright, pytest, locust)
make dev                # API + UI in development mode (reads .env)
make api / make ui      # run just FastAPI (:8000) or Streamlit (:8501)
make kill               # stop API + UI processes (frees :8000 even on hung runs)
make db-init db-seed    # initialize + seed the SQLite database
make db-reset           # destructive: deletes database/statmate.db then re-inits

make test               # uv run pytest tests/ -v
make lint               # uv run ruff check .
make format             # uv run ruff format .
```

Run a single test:

```bash
uv run pytest tests/test_blueprint.py -v
uv run pytest tests/test_workflow_logic.py::test_name -v
```

### Things that bite

- **`PYTHONPATH` must include the repo root.** Scripts and the API import top-level
  `config`, `database`, and `statmate` packages. The Makefile exports it for you; when
  running Python directly, prefix with `PYTHONPATH=$(pwd)` (or `uv run` from repo root).
- **Two ruff versions.** The lockfile pins an old ruff (`0.0.289`); `make lint` uses it,
  but CI lints with `uvx ruff@0.9.6 check statmate tests --select E`. Match CI when in
  doubt — only error-level (`E`) lint is gated.
- **Type checker mismatch.** `pyproject.toml` configures **pyright** (`[tool.pyright]`),
  but `make type-check` calls `mypy` (not a declared dep). Prefer `uv run pyright`.
- **Ruff is strict**: line-length 120, single quotes, Google docstrings, and the `ANN`,
  `ARG`, `D`, `S`, `B` rule families are enabled. New code needs type annotations and
  docstrings to pass.

## Architecture

### Request → analysis flow

```
Streamlit UI / React frontend / REST client
        │  HTTP
FastAPI (statmate/api/main.py) ── routes/ ── services/ ── scheduler/ (APScheduler)
        │
AnalysisService (api/services/analysis_service.py)
        │  wraps
StatMateWorkflow (statmate/workflow/statmate_flow_refactored.py)
        │  compiles + runs
LangGraph graph (statmate/workflow/graph_builder.py)
```

`AnalysisService` is the seam between the web app and the statistical engine — it loads the
dataset (parquet), constructs a `Config`, instantiates `StatMateWorkflow`, streams progress
back to the API, and persists results. Background/recurring runs go through
`api/scheduler/`.

### The workflow engine (the heart of the system)

`statmate/workflow/` is where almost all real logic lives. `graph_builder.py`
(`build_workflow_graph`) assembles the graph as a fluent builder chain; read its `build()`
sequence to see the full pipeline:

```
Initialization → Intent → Design Verification → (guardrails) → Choice
  → Assess Study Design → paired / independent / ANOVA / categorical / survival test paths
  → Summary → Methodology Auditor → Reviewer
```

Key modules:
- **`state.py`** — `WorkflowState` (Pydantic, not TypedDict). The single object threaded
  through every node; holds the dataframe, `probabilities`, `assumption_log`, results, and
  the `DataBlueprint`.
- **`blueprint.py`** — `DataBlueprint`, an immutable typed snapshot of dataset diagnostics
  (distribution metrics, sample balance, influence/outlier diagnostics) computed once so
  routing nodes don't re-inspect raw data.
- **`nodes*.py`** — node implementations split by phase: `nodes_init`, `nodes_comparison`,
  `nodes_anova`, `nodes_routing`, `nodes_summary` (re-exported via `nodes.py`).
- **`edges.py`** — conditional routing functions (`decide_outcome`, `assess_study_design`,
  `parametric_assumptions`, `decide_anova_path`, …) that pick the next node.
- **`initialization/`** — pre-flight agents: `column_role_agent`, `route_proposal`,
  `structural_check`.
- **`graph_metadata.py` / `workflow_graph_service`** — emit the machine-readable
  nodes/edges + live state powering `GET /analysis/workflow-graph` and the rendered
  SVG/PNG path highlighting in exports.
- **`model_factory.py`** — global LLM model factory; `initialize_default_factory(config)`
  must be called before running (the API does this at startup, `StatMateWorkflow` does it
  in `__init__`).

Node names are constants in `NodeName` (`statmate/core/config.py`) — use these, not string
literals, when referencing graph nodes.

### Agents and statistical core (two-layer pattern)

Every statistical test exists in two layers:
- **`statmate/statistical_core/`** — pure numeric implementation returning a
  `StatTestResult` (`statistical_core/base.py`): test name, statistic, p-value, hypotheses,
  effect size, confidence interval. No LLM involvement.
- **`statmate/agents/`** — Pydantic AI `Agent` wrappers built via `agent_builder.py` that
  call the core function as a tool, add diagnostics/interpretation, and return narrated
  structured output. Agents are imported by name into `graph_builder.py` (e.g.
  `ttest_ind_agent`, `mannwhitneyu_agent`, `chi2_agent`).

When adding a test: implement the numeric function in `statistical_core/`, wrap it as an
agent in `agents/`, register the agent + a node + routing edges in the workflow.

### Two distinct config objects (don't confuse them)

- **`config/settings.py`** — `Settings` (Pydantic Settings). App/deployment config from env
  / `.env`: ports, `DATABASE_URL`, data dirs, CORS, and the **required security secrets**
  (`SECRET_KEY`, `PASSWORD_PEPPER`, `EMAIL_HASH_SECRET`, `EMAIL_ENCRYPTION_KEY`). It
  **fails fast** if these are missing or left as sample placeholders. `create_multi_model_config()`
  bridges env API keys into the model factory.
- **`statmate/core/config.py`** — `Config` dataclass (+ `NodeName`, `DataType`,
  `StatisticalTestConfig`). Statistical thresholds (alpha, normality, variance) and workflow
  behavior. Passed into `StatMateWorkflow`.

### Storage

- **Datasets** are stored as **parquet** files under `data/` (see `storage_service.py`),
  not in the database. The DB holds metadata, analyses, results, tasks, and users.
- **Database** is SQLAlchemy 2.0 with Alembic migrations (`database/`). SQLite by default
  (`database/statmate.db`); swap to PostgreSQL via `DATABASE_URL`.
- **Auth/PII**: passwords are Argon2id-hashed with a server-side pepper; emails are
  encrypted at rest with a keyed lookup hash (see README "Accounts & Data Storage" and
  `statmate/core/pii.py`). Datasets are PII-sanitized (`sanitize_dataframe`) before entering
  the workflow.

### Multi-provider LLM support

Agents run against any Pydantic-AI-compatible provider: OpenAI (default), Anthropic, Google
Gemini, Groq, or **Ollama (local, free)**. Provider/model are selectable per-run (passed
through `StatMateWorkflow.run`) and in the UI. `scripts/list_available_models.py` /
`make list-models` lists what's reachable with current keys.

## Frontend

There are two frontends. **Streamlit** (`statmate/ui/app.py`, `make ui`) is the primary V1
UI. A minimal **React + Vite + TypeScript** client lives in `frontend/` (`make frontend-dev`
on :3000, target set via `frontend/.env` `VITE_API_BASE`).
