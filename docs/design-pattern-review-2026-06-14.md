# Design-pattern review — StatmateAI

- Date: 2026-06-14
- Reviewer: arch-crew / design-patterns (Mode B: object-level pattern review)
- Companion ADR: `docs/adr/0001-table-driven-construction-and-injected-model-factory.md`
- Companion (architecture-level): `docs/architecture-review-2026-06-14.md`

Scope: object-level patterns (GoF + Pythonic idioms). The architecture-level findings live in the
companion review. **Headline: the code is broadly idiomatic** — Pydantic models instead of
getter/setter ceremony, closures instead of Command classes, a fluent builder for the graph, a
generator-based `get_db` context manager. The findings below are duplication and one global-state
hazard, each fixed by a *language feature*, not added ceremony.

## Findings

### 1. Provider construction: 5-arm `if` chain + five near-identical methods
`core/model_provider.py` — `create_model` branches `if provider == ModelProvider.OPENAI / ANTHROPIC
/ GOOGLE / GROQ / OLLAMA`, each calling a `_create_*_model` method. The five methods differ only in
which `Provider`/`Model` class they import; the body (assemble `provider_params` from `api_key` +
`api_base`, instantiate, wrap) is copy-pasted.

- **Fix: Registry / Strategy → in Python, a dict + one builder.**
  ```python
  _PROVIDERS = {
      ModelProvider.OPENAI:    (OpenAIProvider,    OpenAIModel),
      ModelProvider.ANTHROPIC: (AnthropicProvider, AnthropicModel),
      ModelProvider.GOOGLE:    (GeminiProvider,    GeminiModel),
      ModelProvider.GROQ:      (GroqProvider,      OpenAIModel),   # OpenAI-compatible
      ModelProvider.OLLAMA:    (OpenAIProvider,    OpenAIModel),   # adjust to actual
  }
  def _build_model(self, provider, model_name, api_key, cfg, **kw):
      provider_cls, model_cls = _PROVIDERS[provider]
      params = {"api_key": api_key} if api_key else {}
      if cfg.api_base and provider_cls is not GeminiProvider:
          params["base_url"] = cfg.api_base
      return model_cls(model_name, provider=provider_cls(**params), **kw)
  ```
- **Why:** collapses ~5 branches + ~5 methods into a table + one function; a new provider is one row.
- **Effort:** small. (Keep lazy `from pydantic_ai...import` inside the builder if import cost matters.)

### 2. Node-wrapper boilerplate: 10+ identical closures in `graph_builder.py`
`workflow/graph_builder.py` — `normality_wrapper`, `paired_t_wrapper`, `wilcoxon_wrapper`,
`indep_t_wrapper`, `welch_wrapper`, `mann_wrapper`, `chi2_wrapper`, `fisher_wrapper`,
`mcnemar_wrapper`, `trend_wrapper` are the same 4 lines, varying only by agent-builder fn and
`probability_key`.

- **Fix: Factory Method → in Python, one closure factory + a data table.**
  ```python
  def make_test_node(agent_builder, probability_key):
      def node(state: WorkflowState) -> WorkflowState:
          model = create_model(model_name=state.model_name, provider=state.provider)
          settings = create_model_settings(model_name=state.model_name)
          agent = agent_builder(model=model, model_settings=settings)
          return call_test_agent(agent, state, probability_key=probability_key)
      return node

  _TEST_NODES = [
      (NodeName.PAIRED_T, ttest_rel_agent,  "paired_t_test"),
      (NodeName.WILCOXON, wilcoxon_agent,   "wilcoxon_signed_rank"),
      # ...
  ]
  for name, builder, key in _TEST_NODES:
      self.graph.add_node(name, make_test_node(builder, key))
  ```
- **Why:** removes 10× copy-paste and the drift risk (a wrapper forgetting `state.provider`).
  The conditional-edge wiring stays explicit where it differs.
- **Effort:** small.

### 3. Mutable global model factory (hidden coupling + concurrency hazard)
`workflow/model_factory.py` — module-global `default_factory` is **reassigned per run** via
`initialize_default_factory(...)` (from `StatMateWorkflow.__init__`) and lazily built in
`get_default_factory()` with `global`. Node wrappers reach it implicitly through `create_model`.

- **Fix: Dependency Injection (NOT Singleton).** Build the `ModelFactory` once at startup and pass it
  where needed (or read it from the run's `Config`/state), instead of mutating a process-global on
  each run. Keep a build-once module default for backward-compatible imports; drop the per-run
  reassignment.
- **Why:** the API can run concurrent analyses with different model configs; a reassigned global is a
  race. This is the one finding with a **correctness** (not just tidiness) payoff.
- **Effort:** medium — touches the wrapper call sites from Finding 2, so do them together.

### 4. Repeated `potential_suggestions` literal across every `*_agent` factory
`agents/comparison_agents.py`, `anova_agents.py`, `categorical_comparison_agent.py` — the same
`'Please suggest the best way to perform the test. If results are not clear, propose different
tests.'` string is pasted into every factory.

- **Fix: no pattern — plain duplication.** Hoist to one `DEFAULT_TEST_SUGGESTION` constant (in
  `agent_builder.py`) and default the `build_stat_test_agent` parameter to it.
- **Why:** one edit point for prompt wording; the named `*_agent` factories otherwise stay (they read
  well as Factory Methods — see Leave-as-is).
- **Effort:** small.

## Refactoring plan (step by step)

1. **Finding 4** — hoist `DEFAULT_TEST_SUGGESTION`. *Trivial warm-up; isolated; run tests.*
2. **Finding 1** — provider registry dict + `_build_model`. *Pair with arch-review Finding 3 (move the
   factory into `core/`) since both edit the model modules.* Run `test_*` model paths.
3. **Finding 3** — inject the `ModelFactory`; remove per-run global reassignment, keep a build-once
   default. *Correctness fix; do before/with step 4 because they share call sites.*
4. **Finding 2** — `make_test_node` + `_TEST_NODES` table in `graph_builder.py`. *Largest line
   reduction; the conditional edges stay explicit.* Run `test_workflow_logic`, `test_decision_engine`.

Each step is internal-only (no behaviour change), independently shippable, and covered by the
existing suite. Suggested as 4 small commits.

## Leave as-is (already idiomatic)

- **`agent_builder.build_stat_test_agent`** — a clean Factory Method using closures + pydantic-ai
  tools; the per-test `*_agent` functions are readable named factories. Keep them (only dedupe the
  shared string, Finding 4).
- **`graph_builder` fluent builder chain** — appropriate Builder; the `.add_*_path().build()` reads well.
- **`database/session.py` `get_db`** — textbook generator context manager with `try/finally` close.
  No change.
- **`edges.py` `decide_*` routing functions** — sequential `if scale == ... / if assumption_status
  ...` guards. These encode a genuine multi-attribute decision tree; a dict/Strategy dispatch would
  *obscure* the guard combinations, not clarify them. Leave linear.
- **`WorkflowState` (Pydantic)** — attributes/`Field`, no getter/setter ceremony. Idiomatic.

## Consider simplifying away

- The mutable-global factory (Finding 3) — replace the hand-rolled singleton with DI.
- The unused ABCs in `core/base_interfaces.py` (`StatMateAgent`, `WorkflowNode`, `DataTransformer`,
  `DataValidator`) — never subclassed; tracked in the architecture review (Finding 1). They are dead
  speculative structure; delete.
