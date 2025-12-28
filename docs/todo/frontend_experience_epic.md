# Frontend User Experience (UX) Analysis & Improvements

Goal: Make the complex statistical workflow feel intuitive and transparent.

## UI: Interactive Research Question Wizard
Users often do not know which test to ask for.
- [ ] Build a "Guided Analysis" wizard on the dashboard:
  - "What are you looking for?" (Difference between groups / Relationship between variables / Prediction).
  - "How many groups do you have?"
- [ ] Dynamically update `WorkflowGraph.tsx` as the user answers, showing the likely path.

## Visuals: Statistical Assumption Stoplights
- [ ] Implement a visual dashboard of assumptions (normality, variance, independence).
- [ ] Use a red/yellow/green stoplight system in the UI:
  - Green: assumptions met.
  - Yellow: minor violations, using robust methods.
  - Red: significant violations; interpretation may be biased.

## Interaction: Deep-Dive Explanation Overlays
- [ ] Add "What is this?" tooltips for every statistical term (p-value, R-squared, degrees of freedom).
- [ ] Implement a "Show Logic" button on the final report that opens the relevant part of the LangGraph decision stream explaining why a specific test was chosen.

## Data: Spreadsheet-First Interaction
- [ ] Replace simple file upload with a data preview grid (e.g., ag-grid or react-table).
- [ ] Allow users to manually cast columns (e.g., "Treat this ID column as categorical") before starting analysis.
