# Release prep checklist for v1.0.0

## Load/perf
- Use `python scripts/perf_baseline.py --dataset frontend/public/samples/clinical_trial_sample.csv` to validate plotting/export runtime without touching the LLM stack.
- For API-level load, run `locust -f tests/load/locustfile.py --host=https://<api-host>` (locustfile stub to add scenarios) targeting `/analysis/run`, `/analysis/{id}/stream`, and `/analysis/{id}/results` with 50–100 concurrent users and seeded datasets.
- Target p95 under 3s for `/analysis/{id}` polling and under 8s for streaming step delivery on 1k-row datasets.

## SLAs & alerts
- Uptime: 99.5% monthly for API; 99.9% for auth endpoints.
- Latency: p95 < 800ms for GET endpoints; <3s for POST `/analysis/run` acknowledgement.
- Alerting: Grafana/Prometheus on error_rate >1%, latency budget burn, SSE disconnects, and worker queue depth > 25.

## Backups/retention
- Run `scripts/backup_data.sh` nightly to archive `data/` (uploads, logs, results); retain 30 days, mirror to cloud bucket.
- Database: daily logical dump + PITR if using managed Postgres/SQLite snapshots on the host.
- Redaction: purge raw uploads older than 30 days unless explicitly pinned for audit.

## Rollback/runbook
- Blue/green deploy with health checks on `/health` + `/analysis/{id}` probe using a canned analysis ID.
- If streaming fails: switch traffic back to previous revision, invalidate cache for `/frontend` and requeue in-progress analyses.
- Keep migration rollback SQL alongside forward migrations; on failure, pause writers and restore last backup tarball (`tmp_devcontainer/backups`).

## Pricing / usage limits
- Default free tier: 10 analyses/day, datasets capped at 25 MB; enforce via rate limit middleware + dataset size validation.
- Paid tier guidance: lift to 200 analyses/day, 250 MB datasets; include overage alerts based on Prometheus counters.
- UI copy should surface current quota + remaining runs next to the Run button (to be wired once billing provider is picked).
