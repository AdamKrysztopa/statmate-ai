# Non-Statistician Friendly Summary & Medical Accuracy

Goal: Transform raw statistical output into ready-for-publication medical summaries.

## Summarizer: Template-Based Narrative Generation
- [ ] Refactor `summarizer_agent.py` to use medical reporting standards (CONSORT/STROBE) with required fields for population, intervention, comparator, and outcomes.
- [ ] Implement the "Big 3" structure for every summary:
  - The Finding: Plain English (e.g., "Treatment A was significantly better than B").
  - The Evidence: Key numbers (e.g., "Mean diff 5.2, 95% CI [2.1, 8.3], p=0.001").
  - The Caveat: Assumptions (e.g., "Data was slightly skewed; non-parametric tests were used for robustness").
- [ ] Ensure every summary includes 95% confidence intervals alongside p-values and effect sizes.

## Summarizer: Clinical vs. Statistical Significance
- [ ] Add a prompt instruction to SummarizerAgent to distinguish between statistically significant and clinically meaningful (include an explicit "Clinically Meaningful? Yes/No + rationale").
- [ ] Flag cases where the effect size is small but the p-value is significant due to large N ("Small effect size despite significance").
- [ ] Highlight when CIs cross clinically meaningful thresholds even if p < 0.05.

## Reviewer: Medical Peer Reviewer Persona
- [ ] Update `reviewer_agent.py` to simulate a picky medical journal editor with explicit CONSORT/STROBE checklists.
- [ ] Add a check for p-hacking indicators (e.g., "The analysis suggests multiple comparisons were made without Bonferroni correction").
- [ ] Ensure all 95% confidence intervals are presented alongside p-values (medical journal standard).
- [ ] Require a "Finding / Evidence / Caveat" sanity check against raw statistical outputs before sign-off.

## Visuals: Interpretive Captioning
- [ ] Create a service (`visualization_service.py`) that generates alt-text for every chart that explains the trend to a non-expert (e.g., "This boxplot shows that the median recovery time for the Drug group is 2 days shorter than the Placebo group").
