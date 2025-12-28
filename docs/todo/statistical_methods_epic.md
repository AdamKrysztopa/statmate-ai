# Statistical Method Completeness

Goal: Expand the core statistical capabilities to cover advanced research requirements and edge cases.

## Core: Non-Parametric Expansion
The current implementation favors parametric tests (ANOVA, t-test). Add robust fallbacks for non-normal or small-sample data.
- [ ] Implement Wilcoxon Signed-Rank Test as an alternative to paired t-tests in `comparison.py`.
- [ ] Implement Kruskal-Wallis H-test as a non-parametric alternative to one-way ANOVA in `anova.py`.
- [ ] Implement Friedman Test for repeated measures with non-normal distributions.
- [ ] Implement Dunn's post-hoc test for Kruskal-Wallis results.

## Core: Categorical & Contingency Analysis
- [ ] Add Fisher’s Exact Test for 2x2 tables with small expected frequencies (< 5).
- [ ] Implement McNemar’s Test for paired categorical data (e.g., pre/post treatment presence of symptoms).
- [ ] Add Cochran-Armitage Trend Test for ordered categorical variables.

## Core: Effect Size & Power
Current results provide p-values but lack magnitude context.
- [ ] Calculate Cohen's d for t-tests and partial eta squared for ANOVA.
- [ ] Calculate Cramer's V or Phi coefficient for Chi-Square tests.
- [ ] Add post-hoc power analysis to interpret non-significant results (determine if the study was underpowered).

## Core: Multi-Level & Regression
- [ ] Implement simple linear regression (beyond just correlation) to provide coefficients and intercepts.
- [ ] Add multiple linear regression support with collinearity checks (VIF).
- [ ] Implement basic logistic regression for binary outcome prediction.

## Data Quality: Outlier & Influence Detection
- [ ] Implement Cook’s distance and leverage calculation for regression tasks.
- [ ] Add Tukey’s fences or z-score-based outlier detection in the initial insights phase.
