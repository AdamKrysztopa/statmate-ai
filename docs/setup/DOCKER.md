# Docker Setup – StatmateAI

## Quick Start

```bash
# 1. Create your .env file
cp .env.example .env
# Edit .env with your API keys

# 2. Start the backend
docker compose up --build

# 3. (Optional) Start with frontend dev server
docker compose --profile dev up --build

# 4. (Optional) Start production stack (nginx + backend)
docker compose --profile prod up --build -d
```

## Architecture

| Service          | Port  | Description                          |
| ---------------- | ----- | ------------------------------------ |
| `backend`        | 8000  | FastAPI API server (always started)  |
| `frontend`       | 3000  | Nginx serving React build (prod)     |
| `frontend-dev`   | 3000  | Vite dev server with HMR (dev)       |

## Make Targets

```bash
make docker-build   # Build all images
make docker-up      # Start backend only
make docker-dev     # Start backend + frontend dev server
make docker-prod    # Start production stack (detached)
make docker-down    # Stop all services
make docker-logs    # Tail logs
```

## Profiles

- **Default** (no profile): Only the `backend` service starts.
- **`dev`**: Adds `frontend-dev` (Vite dev server with hot-reload, mounts local `frontend/` directory).
- **`prod`**: Adds `frontend` (nginx serving the built React app, proxies `/api/` to backend).

## Environment Variables

All environment variables are read from `.env` at the project root. See [`.env.example`](../.env.example) for the full list.

Key Docker-specific variables:

| Variable         | Default | Description                  |
| ---------------- | ------- | ---------------------------- |
| `API_PORT`       | 8000    | Host port for the backend    |
| `FRONTEND_PORT`  | 3000    | Host port for the frontend   |
| `ENVIRONMENT`    | development | `development` or `production` |

## Volumes

| Volume                  | Purpose                                  |
| ----------------------- | ---------------------------------------- |
| `statmate-data`         | Persistent uploads, results, and logs    |
| `statmate-db`           | SQLite database                          |
| `frontend-node-modules` | Cached node_modules for dev server       |

## Building Individual Stages

The multi-stage [`Dockerfile`](../Dockerfile) supports building specific targets:

```bash
# Backend only
docker build --target backend -t statmate-backend .

# Frontend only (nginx)
docker build --target frontend -t statmate-frontend .
```

## Local Development (without Docker)

If you prefer running natively:

```bash
make install        # Install Python deps via uv
make setup-env      # Copy .env.example → .env
make db-init        # Initialize SQLite database
make dev            # Start FastAPI backend
make frontend-dev   # Start React dev server (separate terminal)
```
