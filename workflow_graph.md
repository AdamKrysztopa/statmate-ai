```mermaid
---
config:
  flowchart:
    curve: linear
---
graph TD;
    start([Start]):::first --> initialization_agent(Initialization Agent)
    initialization_agent --> design_verification(Design Verification)
    design_verification --> assess_study_design(Assess Study Design)
    design_verification --> chi_square_test(Chi-square test)
    design_verification --> fisher_exact_test(Fisher exact test)
    design_verification --> cochran_armitage_trend_test(Cochran-Armitage trend test)
    design_verification --> mcnemar_test(McNemar test)
    design_verification --> nonparametric_tests(Nonparametric Tests)
    design_verification --> anova_assumptions(ANOVA assumptions)
    design_verification --> one_way_anova(One-way ANOVA)
    design_verification --> kruskal_wallis_h_test(Kruskal-Wallis H-test)
    design_verification --> anova_repeated_measures(ANOVA repeated measures)
    design_verification --> friedman_test(Friedman test)
    design_verification --> choice_node(Choice Node)
    design_verification --> cox_regression_placeholder["Cox regression (placeholder)"]
    design_verification --> descriptive_summary(Descriptive Summary)
    design_verification --> user_intervention_needed(User Intervention Needed)
    design_verification --> design_reconciliation(Design Reconciliation)

    design_reconciliation --> assess_study_design
    design_reconciliation --> chi_square_test
    design_reconciliation --> fisher_exact_test
    design_reconciliation --> cochran_armitage_trend_test
    design_reconciliation --> mcnemar_test
    design_reconciliation --> nonparametric_tests
    design_reconciliation --> anova_assumptions
    design_reconciliation --> one_way_anova
    design_reconciliation --> kruskal_wallis_h_test
    design_reconciliation --> anova_repeated_measures
    design_reconciliation --> friedman_test
    design_reconciliation --> choice_node
    design_reconciliation --> cox_regression_placeholder
    design_reconciliation --> descriptive_summary
    design_reconciliation --> user_intervention_needed
    design_reconciliation --> design_reconciliation

    assess_study_design --> parametric_assumptions_hold(Parametric assumptions hold?)
    assess_study_design --> two_independent_groups(Two Independent Groups?)
    assess_study_design --> end_node

    parametric_assumptions_hold --> paired_t_test(Paired t-test)
    parametric_assumptions_hold --> wilcoxon_signed_rank_test(Wilcoxon Signed-Rank test)
    parametric_assumptions_hold --> end_node

    two_independent_groups --> independent_t_test(Independent t-test)
    two_independent_groups --> nonparametric_tests
    two_independent_groups --> anova_assumptions
    two_independent_groups --> end_node

    anova_assumptions --> one_way_anova
    anova_assumptions --> kruskal_wallis_h_test
    anova_assumptions --> end_node

    paired_t_test --> summary(Summary)
    wilcoxon_signed_rank_test --> summary
    independent_t_test --> summary
    nonparametric_tests --> summary
    welch_s_t_test(Welch's t-test) --> summary
    mann_whitney_u(Mann-Whitney U) --> summary
    one_way_anova --> summary
    kruskal_wallis_h_test --> summary
    anova_repeated_measures --> summary
    friedman_test --> summary
    chi_square_test --> summary
    fisher_exact_test --> summary
    mcnemar_test --> summary
    cochran_armitage_trend_test --> summary

    cox_regression_placeholder --> end_node([End]):::last
    descriptive_summary --> end_node
    user_intervention_needed --> end_node
    summary --> reviewer_agent(Reviewer Agent)
    reviewer_agent --> end_node

    classDef default fill:#f2f0ff,line-height:1.2
    classDef first fill-opacity:0,stroke-dasharray:4 2
    classDef last fill:#bfb6fc
```

## Canonical metadata (machine-readable)

The API exposes the graph at `GET /analysis/workflow-graph`, and embeds the same payload (with `visited_nodes`, `active_node`, and `selected_path`) in `/analysis/{id}` responses and SSE `step` events. Node IDs are normalized, lowercase identifiers:

- `start` → `initialization_agent` → `design_verification` (or `design_reconciliation` on mismatches) → downstream test nodes.
- Decisions: `assess_study_design` → `parametric_assumptions_hold` → `paired_t_test` / `wilcoxon_signed_rank_test`; `two_independent_groups` → `independent_t_test` / `nonparametric_tests` / `anova_assumptions`; `anova_assumptions` → `one_way_anova` / `kruskal_wallis_h_test`.
- Categorical/paired: `chi_square_test`, `fisher_exact_test`, `mcnemar_test`, `cochran_armitage_trend_test`.
- Multi-group/repeated: `anova_repeated_measures`, `friedman_test`.
- Guardrails: `descriptive_summary`, `user_intervention_needed`, `cox_regression_placeholder` (survival placeholder), `choice_node`.
- Sink: `summary` → `reviewer_agent` → `end` (rendered as `end_node` in the diagram to avoid Mermaid reserved keywords).

Each node includes `id`, `label`, `kind`, and `transitions`; edges are provided as source/target pairs for clients/exports.
