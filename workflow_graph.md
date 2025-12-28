```mermaid
---
config:
  flowchart:
    curve: linear
---
graph TD;
    start([Start]):::first --> initialization_agent(Initialization Agent)
    initialization_agent --> assess_study_design(Assess Study Design)
    initialization_agent --> chi_square_test(Chi-square test)
    initialization_agent --> fisher_exact_test(Fisher exact test)

    assess_study_design --> parametric_assumptions_hold(Parametric assumptions hold?)
    assess_study_design --> two_independent_groups(Two Independent Groups?)
    assess_study_design --> end_node

    parametric_assumptions_hold --> paired_t_test(Paired t-test)
    parametric_assumptions_hold --> wilcoxon_signed_rank_test(Wilcoxon Signed-Rank test)
    parametric_assumptions_hold --> end_node

    two_independent_groups --> independent_t_test(Independent t-test)
    two_independent_groups --> nonparametric_tests(Nonparametric Tests)
    two_independent_groups --> end_node

    chi_square_test --> summary(Summary)
    fisher_exact_test --> summary
    paired_t_test --> summary
    wilcoxon_signed_rank_test --> summary
    independent_t_test --> summary
    nonparametric_tests --> summary
    summary --> reviewer_agent(Reviewer Agent)
    reviewer_agent --> end_node([End]):::last

    classDef default fill:#f2f0ff,line-height:1.2
    classDef first fill-opacity:0,stroke-dasharray:4 2
    classDef last fill:#bfb6fc
```

## Canonical metadata (machine-readable)

The API exposes the graph at `GET /analysis/workflow-graph`, and embeds the same payload (with `visited_nodes`, `active_node`, and `selected_path`) in `/analysis/{id}` responses and SSE `step` events. Node IDs are normalized, lowercase identifiers:

- `start` → `initialization_agent` → `assess_study_design`
- Branches: `parametric_assumptions_hold` → `paired_t_test` / `wilcoxon_signed_rank_test`; `two_independent_groups` → `independent_t_test` / `nonparametric_tests`; categorical: `chi_square_test` / `fisher_exact_test`
- Sink: `summary` → `reviewer_agent` → `end` (rendered as `end_node` in the diagram to avoid Mermaid reserved keywords)

Each node includes `id`, `label`, `kind`, and `transitions`; edges are provided as source/target pairs for clients/exports.
