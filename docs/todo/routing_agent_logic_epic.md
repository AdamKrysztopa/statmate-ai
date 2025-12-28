# Improving Method Selection & Agent Routing

Goal: Solve the method selection issue by making the router more deterministic and data-aware.

## Router: Deterministic Decision Tree (Data-Driven Routing)
The LLM currently has too much freedom in selecting methods; shift to a checklist-first routing logic.
- [ ] Refactor `edges.py` to use a `DecisionEngine` that evaluates hard constraints:
  - If type == continuous AND normal == false → route to non-parametric.
  - If groups > 2 → route to ANOVA/Kruskal-Wallis.
  - If paired == true → route to paired test.
- [ ] Create a `ConstraintValidator` that prevents the agent from executing a test if assumptions (normality, homoscedasticity) are violated.

## Agents: Enhanced Initial Insights Agent
`initial_insights_agent.py` needs to provide a metadata blueprint that dictates the path.
- [ ] Update `InitialInsightsAgent` to output a structured `DataBlueprint` JSON:
  - measurement_scale: Nominal | Ordinal | Interval | Ratio
  - sample_size_per_group: integer
  - missing_data_percentage: float
- [ ] Pass this `DataBlueprint` to all subsequent nodes in the state.

## Agents: Methodology Auditor Agent
- [ ] Implement a `ReviewerAgent` variant specifically for method selection validation.
- [ ] Challenge the `ComparisonAgent`: "You chose a t-test, but the Shapiro-Wilk test failed. Explain why or suggest a transition to Mann-Whitney."

## Workflow: Intent Clarification Loop
Sometimes the data does not reveal the research question (e.g., "Is this a superiority or equivalence trial?").
- [ ] Add an `IntentAgent` node that triggers if the user prompt is ambiguous.
- [ ] Implement an interrupt/resume flow in LangGraph to ask the user: "I see paired data; should I treat this as a pre-post comparison?"
