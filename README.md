# StatMate

**StatMate** is an AI-driven statistical analysis tool tailored for clinical and observational research, especially in small-sample studies. It automates test selection, statistical computation, and result summarization using LLM agents and Python statistical libraries.

---

## ✨ Features

- 🧠 Automatically selects the appropriate statistical test based on study design.
- 📊 Runs standard tests (t-tests, ANOVA, correlations, etc.) via SciPy and Statsmodels.
- 📋 Returns structured results in `Pydantic` objects for easy downstream use.
- 📈 Generates natural-language summaries and plots for publication-ready output.
- 🔍 Performs assumption checks and flags methodological issues.
- 🧪 Human-in-the-loop optional for verification and edits.

---

## 🧰 Tech Stack

- **LLM Agents**: Pydantic AI, LangGraph
- **Backend**: FastAPI
- **Frontend**: Streamlit
- **Stats**: SciPy, Statsmodels
- **Data Handling**: Pandas, NumPy
- **Visualization**: Seaborn, Matplotlib

---

## 🧪 Implemented Statistical Tests

- [x] Shapiro–Wilk Test (normality) and other less powerfool
- [x] Paired & Independent t-tests
- [x] Wilcoxon Signed-Rank, Mann–Whitney U
- [x] One-way & Repeated Measures ANOVA
- [ ] Pearson & Spearman correlation
- [x] Chi-square test, Fisher’s exact test

## 🤖 Implemented Statistical Agents

- [x] Normality test agents
- [x] Normality multi-tests agent
- [x] Shapiro-Wilk for sample diffetece
- [x] Wilcoxon Signed-Rank, Mann–Whitney U
- [ ] Pearson & Spearman correlation
- [x] Chi-square test, Fisher’s exact test


---

## 🚧 Roadmap

- [ ] Add effect size metrics
- [ ] Extend to regression models (OLS, GLM)
- [ ] Multi-comparison adjustment tools
- [ ] User-configurable reporting styles
- [ ] Streamlit + FastAPI deployment

---

## Data Flowchart

```mermaid
flowchart TD
    A[Start: What is your analysis objective?] --> B{Outcome Type?}

    B -- Continuous Outcome --> C[Assess Study Design]
    B -- Categorical Outcome --> D[Choose Chi-square test<br/>or Fisher exact test if expected counts are low]

    C --> E{Paired Measurements on Each Subject?}
    E -- Yes --> F{Parametric assumptions hold?}
    F -- Yes --> G[Paired t-test]
    F -- No --> H[Wilcoxon Signed‑Rank test]

    E -- No --> I{Two Independent Groups?}
    I -- Yes --> J{Are assumptions met?<br/>normality & equal variances}
    J -- Yes --> K[Independent‑samples t-test<br/>Student’s t-test]
    J -- No --> L[Consider Both Options:]
    L --> M[Option A: Welch’s t-test<br/>parametric adjustment]
    L --> N[Option B: Mann‑Whitney U test<br/>non‑parametric]

    %%I -- No --> subgraph Unused["More than Two Groups *Not used right now*"]
    %%    O{More than Two Groups?}
    %%    O -- Yes --> P{Repeated measures?}
    %%    P -- Yes --> Q[Repeated‑measures ANOVA]
    %%    P -- No --> R[One‑way ANOVA]
    %% end
    %% This branch is not used right now -->

    C --> S{Assess relationship/association?}
    S -- Yes --> T{Data characteristics?}
    T -- Parametric continuous --> U[Pearson correlation]
    T -- Ordinal/non‑normal --> V[Spearman correlation]
    S -- No --> W[Consider regression:<br/>Linear Regression Analysis]