# 0001. Replace branch/closure boilerplate with table-driven construction and an injected model factory

- Status: Accepted
- Date: 2026-06-14
- Deciders: AdamKrysztopa

## Context

A design-pattern (object-level) audit of StatmateAI found three repeated construction smells, all
of the same family — *behaviour that varies by a key is expressed as duplicated code rather than as
data*:

1. **Provider construction** (`statmate/core/model_provider.py`). `ModelProviderSystem.create_model`
   dispatches with a five-arm `if provider == ModelProvider.X` chain into five `_create_*_model`
   methods that are near-identical: build `provider_params` from `api_key` + `api_base`, instantiate
   the pydantic-ai `Provider`, wrap it in a `Model`. Adding a provider means a new branch **and** a
   new copy-pasted method.

2. **Test-node construction** (`statmate/workflow/graph_builder.py`). Ten-plus `*_wrapper(state)`
   closures (`paired_t_wrapper`, `welch_wrapper`, `mann_wrapper`, `chi2_wrapper`, …) are byte-for-byte
   identical except for the agent-builder function and a `probability_key` string:
   ```python
   def paired_t_wrapper(state: WorkflowState) -> WorkflowState:
       model = create_model(model_name=state.model_name, provider=state.provider)
       settings = create_model_settings(model_name=state.model_name)
       agent = ttest_rel_agent(model=model, model_settings=settings)
       return call_test_agent(agent, state, probability_key="paired_t_test")
   ```

3. **Mutable global model factory** (`statmate/workflow/model_factory.py`). A module-global
   `default_factory` is reassigned on every workflow run via `initialize_default_factory(...)` (called
   from `StatMateWorkflow.__init__`), and lazily created in `get_default_factory()` with the `global`
   keyword. This is a hand-rolled singleton holding **mutable, per-run** state (the model config),
   reachable implicitly from deep inside node wrappers — a hidden-coupling and concurrency hazard for
   the API, which can run multiple analyses with different model configs.

A secondary, smaller duplication: every `*_agent` factory in `statmate/agents/*_agents.py` repeats
the same `potential_suggestions` literal string verbatim.

The cardinal Pythonic rule applies: none of these need a GoF class hierarchy — the language already
solves them with a dict, a closure factory, and plain dependency injection.

## Decision

Adopt **data/closure-driven construction** for the two duplication sites, and **dependency
injection** for the factory — in their Pythonic forms:

| Smell | Pattern (concept) | Pythonic form chosen |
|-------|-------------------|----------------------|
| Provider `if`-chain + 5 twin methods | **Registry / Strategy** | A module-level `dict` `{ModelProvider: (ProviderCls, ModelCls)}` plus **one** generic `_build_model(...)` that reads `api_key`/`api_base` once. Groq/OpenAI share `OpenAIModel`. |
| 10+ identical node closures | **Factory Method** | One `make_test_node(agent_builder, probability_key)` returning the closure (or `functools.partial`); call sites become a single data table of `(NodeName, agent_builder, probability_key)`. |
| Mutable global `default_factory` | **Dependency Injection** (not Singleton) | Construct one `ModelFactory` at app/workflow startup and **pass it in** (constructor/param). Keep a module-level *default* object that is built once and never reassigned per-run; remove the `global`-reassignment path. |
| Repeated `potential_suggestions` string | *(no pattern — plain duplication)* | Hoist to one module constant `DEFAULT_TEST_SUGGESTION`. |

Explicitly **not** chosen: a metaclass/`__new__` singleton, an Abstract Factory class tree, or a
per-provider subclass hierarchy — all heavier than the dict/closure forms above and discouraged by
the Pythonic rule.

## Consequences

- **Less code, one place to change.** Adding a provider = one dict row; adding a test node = one
  table row. The copy-paste drift risk (a wrapper that forgets `state.provider`, a `_create_*` that
  omits `base_url`) is removed structurally.
- **Safer concurrency.** Injecting the factory removes the per-run mutation of a process-global, so
  concurrent API analyses with different model configs no longer race on shared state. This is the
  main correctness payoff and should be sequenced first if concurrency bugs are suspected.
- **Migration is incremental and low-risk.** Each change is internal (no public API or behaviour
  change) and covered by the existing `tests/` suite (`test_workflow_logic`, `test_decision_engine`,
  `test_statistical_methods_expansion`). The factory DI step keeps a backward-compatible module-level
  default so existing imports (`from ...model_factory import create_model`) keep working.
- **Cost accepted.** A table/closure factory is a hair less greppable than ten named functions — a
  reader jumps to the table instead of to a uniquely-named wrapper. Judged worthwhile given the 10×
  duplication. The named `*_agent` factory functions are **kept** (they read well as Factory Methods);
  only their shared string literal is hoisted.
- **Related decision:** this ADR overlaps with the architecture review's Finding 3 (consolidate the
  LLM port into `core/`). Do the provider-registry change (here) and the port consolidation (arch
  review) together — same files.

The refactoring is sequenced in `docs/design-pattern-review-2026-06-14.md`.
