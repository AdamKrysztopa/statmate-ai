# Epic: Improving Method Selection & Agent Routing

Goal: Solve the "method selection" issue by making the router more deterministic, data-aware, and elastic for future expansion.

## Router: Deterministic & Elastic Decision Engine
Instead of hardcoding `if/else` blocks in `edges.py`, move to a registry-based `DecisionEngine` that can scale as new tests are added.
- [ ] Registry Pattern for Statistical Methods:
  - Create a `MethodRegistry` that maps data profiles (e.g., Scale=Ratio, Normal=False, Groups=2) to suggested statistical tests.
  - Implement a weighting system: if multiple tests apply, the engine selects the preferred one but keeps alternatives as backups for the Auditor Agent.
- [ ] Refactor `edges.py` to `DecisionEngine`:
  - Implement `evaluate_routing(state: StatMateState) -> str` which reads the `DataBlueprint`.
  - Handle hard constraints:
    - continuous + non-normal + 2-groups → `non_parametric_node`
    - categorical + paired → `mcnemar_node`
    - survival_data → `cox_regression_node` (placeholder for future)
- [ ] Assumption-Triggered Rerouting:
  - If a test node runs and finds a violation (e.g., Levene's test fails for ANOVA), it should return a `fail` status that the `DecisionEngine` uses to reroute to a Robust or Non-parametric alternative.

## Agents: Data Blueprint & Deep Metadata
The `InitialInsightsAgent` must be the brain of the operation, providing a machine-readable roadmap.
- [ ] Enhanced `DataBlueprint` Schema:
  - Update `initial_insights_agent.py` to produce a strictly typed JSON:
    - `variable_roles`: List identifying Independent, Dependent, Covariate, Group.
    - `distribution_metrics`: Skewness, Kurtosis, and p-values for Normality tests.
    - `sample_balance`: Whether groups have similar N (affects Homoscedasticity importance).
- [ ] State Injection:
  - Ensure every node in the graph has immutable access to this blueprint to prevent hallucinated method selection.

## Agents: Methodology Auditor & Adaptive Critique
The Auditor should not just flag errors but suggest the Next Best Action to keep the flow elastic.
- [ ] Implement `MethodologyAuditorAgent`:
  - Logic: Compare the executed test against the registry-recommended test based on the actual results of the assumptions.
- [ ] Adaptive Critique:
  - If a t-test was run but variance was unequal, the auditor should inject a `correction_step` to run Welch's t-test instead.
- [ ] Assumption Conflict Resolution:
  - Handle cases where Shapiro-Wilk says "Normal" but Q-Q plot (analyzed via Vision) says "Non-Normal".

## Workflow: User-in-the-Loop & Intent Clarification
Make the graph suspendable for human-in-the-loop decisions.
- [ ] Intent Discovery Node:
  - Add `IntentAgent` to check if the user prompt specifies a goal (e.g., "Prove my drug works" vs "Explore differences").
  - Trigger an Ambiguity Modal in the UI if the agent is < 80% confident in the research question.
- [ ] Elastic Choice Points:
  - Implement a `ChoiceNode` where the AI presents options like "Standard Analysis (ANOVA)" vs "Conservative Analysis (Kruskal-Wallis)" and lets the user click to decide.

## Validation: Constraint Guardrails
- [ ] ConstraintValidator Middleware:
  - Create a decorator `@requires_assumptions(normality=True, variance=True)` for statistical core functions.
  - If called without these being met in the state, raise a `StatisticalAssumptionError` that the graph catches to trigger rerouting.
