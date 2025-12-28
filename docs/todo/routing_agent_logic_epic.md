# Epic: Improving Method Selection & Agent Routing

Goal: Solve the "method selection" issue by making the router more deterministic, data-aware, and elastic for future expansion.

## Data-Aware Registry & Intent-Driven Routing (v2)
Make the registry the enforcer of structural constraints so the router cannot "forget" pairing rules or sample sufficiency mid-run.
- [x] NodeMetadata hard constraints:
  - Add `is_paired: bool`, `min_sample_size: int`, and `group_count_range: tuple[int, int]` to each registered node; block nodes that violate the blueprint.
- [x] Data topology detection:
  - Update `initial_insights_agent.py` to lock `is_paired=True` when IDs overlap across groups; emit a `partition_report` with group names, Ns, and overlap percent.
  - Ensure blueprint records `index_column` (pairing id) and `target_column`.
- [x] Pre-flight sufficiency:
  - Add a `SufficiencyValidator` that runs before routing; if any group has `n < 2`, route directly to `DescriptiveSummaryNode`.
- [x] Path locking in router:
  - In `edges.py`/MatchEngine, if `blueprint.is_paired` is true, filter out all nodes where `is_paired` is false; if nothing matches + sample size fails, route to `UserInterventionNode`.
  - If Normality node returns `p < 0.05`, auto-reroute from `PairedTTest` to `WilcoxonSignedRank`; if variance fails, reroute `IndependentTTest` to `WelchTTest`.
- [x] Auditor structural integrity:
  - Add `StructureAuditor` to verify a paired test was used for paired data; if not, trigger a rerun with the correct node before the user sees the report.
- [ ] Implementation guide:
  - [x] Step 1: Extend `DataBlueprint` with `is_paired` and `group_samples` dict.
  - [x] Step 2: Update `registry.py` metadata with the new constraints.
  - [ ] Step 3: Split `ComparisonAgent` into `PairedComparisonHandler` and `IndependentComparisonHandler` nodes.
  - [x] Step 4: Make `edges.py` raise `RoutingError` when an independent-only node is suggested for paired data.

## Router: Deterministic & Elastic Decision Engine
Instead of hardcoding `if/else` blocks in `edges.py`, move to a registry-based `DecisionEngine` that can scale as new tests are added.
- [x] Registry Pattern for Statistical Methods:
  - Create a `MethodRegistry` that maps data profiles (e.g., Scale=Ratio, Normal=False, Groups=2) to suggested statistical tests.
  - Implement a weighting system: if multiple tests apply, the engine selects the preferred one but keeps alternatives as backups for the Auditor Agent.
- [x] Refactor `edges.py` to `DecisionEngine`:
  - Implement `evaluate_routing(state: StatMateState) -> str` which reads the `DataBlueprint`.
  - Handle hard constraints:
    - continuous + non-normal + 2-groups → `non_parametric_node`
    - categorical + paired → `mcnemar_node`
    - survival_data → `cox_regression_node` (placeholder for future)
- [ ] Assumption-Triggered Rerouting:
  - If a test node runs and finds a violation (e.g., Levene's test fails for ANOVA), it should return a `fail` status that the `DecisionEngine` uses to reroute to a Robust or Non-parametric alternative.

## Agents: Data Blueprint & Deep Metadata
The `InitialInsightsAgent` must be the brain of the operation, providing a machine-readable roadmap.
- [x] Enhanced `DataBlueprint` Schema:
  - Update `initial_insights_agent.py` to produce a strictly typed JSON:
    - `variable_roles`: List identifying Independent, Dependent, Covariate, Group.
    - `distribution_metrics`: Skewness, Kurtosis, and p-values for Normality tests.
    - `sample_balance`: Whether groups have similar N (affects Homoscedasticity importance).
- [x] State Injection:
  - Ensure every node in the graph has immutable access to this blueprint to prevent hallucinated method selection.

## Agents: Methodology Auditor & Adaptive Critique
The Auditor should not just flag errors but suggest the Next Best Action to keep the flow elastic.
- [x] Implement `MethodologyAuditorAgent`:
  - Logic: Compare the executed test against the registry-recommended test based on the actual results of the assumptions.
- [x] Adaptive Critique:
  - If a t-test was run but variance was unequal, the auditor should inject a `correction_step` to run Welch's t-test instead.
- [ ] Assumption Conflict Resolution:
  - Handle cases where Shapiro-Wilk says "Normal" but Q-Q plot (analyzed via Vision) says "Non-Normal".

## Workflow: User-in-the-Loop & Intent Clarification
Make the graph suspendable for human-in-the-loop decisions.
- [x] Intent Discovery Node:
  - Add `IntentAgent` to check if the user prompt specifies a goal (e.g., "Prove my drug works" vs "Explore differences").
  - Trigger an Ambiguity Modal in the UI if the agent is < 80% confident in the research question.
- [x] Elastic Choice Points:
  - Implement a `ChoiceNode` where the AI presents options like "Standard Analysis (ANOVA)" vs "Conservative Analysis (Kruskal-Wallis)" and lets the user click to decide.

## Validation: Constraint Guardrails
- [x] ConstraintValidator Middleware:
  - Create a decorator `@requires_assumptions(normality=True, variance=True)` for statistical core functions.
  - If called without these being met in the state, raise a `StatisticalAssumptionError` that the graph catches to trigger rerouting.
