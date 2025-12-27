- Current gap: LangGraph supports per-node streaming (`graph.stream`/`stream_events`, see docs.langchain.com/oss/python/langgraph/use-graph-api), but `StatMateWorkflow.run` collapses the stream to a final state and the API only returns the completed payload. Frontend polling therefore shows “waiting for first agent decision” until completion. **(Done)** 

## P0 – Streaming & Decision Visibility
- [x] Backend | Streaming: Switch analysis workflow to LangGraph streaming (`statmate/api/services/analysis_service.py`, `StatMateWorkflow`) using `.stream_events`/`.stream()` so each node emits an event; stop collapsing to a final state only.
- [x] Backend | Persistence: Extend Analysis persistence to store intermediate steps (e.g., `decision_steps` JSON/`intermediate_log` text); write entries as they stream; add migration and model fields.
- [x] Backend | Background jobs: After each node, flush DB/log so clients see progress; keep final summary/results intact; ensure model fallback (OpenAI→fallback) does not restart/lose the stream.
- [x] Backend | API: Add a streaming endpoint (SSE or chunked `StreamingResponse`) like `GET /analysis/{id}/stream` that relays LangGraph events; keep current polling endpoints working with backoff/rate-limit handling.
- [x] Frontend (React) | Live timeline/tree: Consume the stream via `Response.body.getReader()` or `EventSource`, append steps to a timeline, and highlight the active branch on the decision tree (Mermaid/graph data from `decision_steps`).
- [x] Frontend (Streamlit) | Live view: Poll `intermediate_log`/`decision_steps` every 2–3s, show the first decision as soon as it arrives, and provide clear loading/error states instead of a blank “waiting” view.

## P1 – Security & Reliability for V1.0.0
- [ ] Authn/Authz | Backend/UI: Gate all routes/pages with auth; enforce dataset/analysis ownership checks; implement registration + email verification; store tokens securely; add logout; align Streamlit/React with protected APIs.
- [ ] Error Handling | Backend: Normalize LLM/workflow errors and timeouts into user-friendly responses; validate uploads (type/size) with clear feedback; add retry/backoff around stream consumers to avoid rate-limit storms.
- [ ] Observability | Platform: Add structured logs/metrics/tracing per LangGraph node and per stream delivery; capture timings and delivery failures for frontend consumption.
- [ ] Security Hardening | Platform: Enforce HTTPS in prod; add API rate limiting; run dependency scans (`pip-audit`/Dependabot); sanitize uploads (size limits, optional AV) to reduce risk.
- [ ] Testing | QA: Push unit/integration coverage (>80%) for streaming, decision persistence, auth guards, upload validation; add e2e flows covering major decision branches and live updates.

## P2 – Product & UX Extras
- [ ] Data Viz | Results: Auto-generate plots (distribution/box/scatter) for selected columns; ship in results payload and render in UI next to p-values/effect sizes.
- [ ] Export/Docs | Product: Add exports (PDF/Word via WeasyPrint, CSV) and docs for the step-by-step view with screenshots; include seeded example datasets plus in-app tooltips/guided tour.
- [ ] Release Prep | Ops: Run load/perf tests on realistic datasets; define SLAs/alerts; set up backups/retention; document rollback/runbook for V1.0.0; clarify pricing/usage limits if applicable.
