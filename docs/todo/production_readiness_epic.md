# Production Readiness & Scalability

Goal: Transition from a local development tool to a secure, multi-user SaaS platform.

## Security: PII Sanitization & Data Privacy
HIPAA/GDPR compliance is critical for medical papers.
- [ ] Harden `pii.py`: improve regex patterns for names/emails/addresses and add an automatic "PII column drop" mode for strict datasets.
- [ ] Implement data residency logic: delete uploaded files immediately after analysis or store them in encrypted S3 buckets with auto-expiry.
- [ ] Add a PII audit log showing which columns were masked/dropped before analysis.

## Infra: Async Task Management & Scalability
LLM calls and complex bootstrap simulations take time.
- [ ] Integrate Celery or Temporal for long-running analysis tasks.
- [ ] Move `statmate_flow`/`AnalysisService` execution to a worker process to prevent blocking the FastAPI main thread while preserving streaming updates.
- [ ] Implement WebSockets or Server-Sent Events to stream the LangGraph "Decision Stream" to the frontend in real time.
- [ ] Add retry/backoff + circuit breaker for worker failures with user-facing status updates.

## Auth: Multi-Tenancy & User Workspace
- [ ] Implement proper JWT-based auth (currently placeholders in some areas).
- [ ] Add organization or project-level scoping for datasets and results.

## Monitoring: Statistical Observability
- [ ] Implement LLM evaluation (using RAGAS or LangSmith) to track if the agents are choosing the correct statistical test over time.
- [ ] Log statistical errors (e.g., when a user corrects the AI's method choice) to a feedback dataset for fine-tuning.

## Export: Professional Reporting
- [ ] Build a LaTeX-to-PDF export service for professional-grade reports.
- [ ] Add DOCX export support (using `python-docx`) since many medical researchers work in Microsoft Word.
- [ ] Create a reproducibility bundle (data sample + decision/assumption logs + code snippets) packaged for download.
- [ ] Create `statmate/api/services/export_service.py` to centralize LaTeX-to-PDF/DOCX exports and bundle assembly.
