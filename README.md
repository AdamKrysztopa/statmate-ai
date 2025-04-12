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

- Shapiro–Wilk Test (normality)
- Paired & Independent t-tests
- Wilcoxon Signed-Rank, Mann–Whitney U
- One-way & Repeated Measures ANOVA
- Pearson & Spearman correlation
- Chi-square test, Fisher’s exact test

---

## 🚧 Roadmap

- [ ] Add effect size metrics
- [ ] Extend to regression models (OLS, GLM)
- [ ] Multi-comparison adjustment tools
- [ ] User-configurable reporting styles
- [ ] Streamlit + FastAPI deployment

---
