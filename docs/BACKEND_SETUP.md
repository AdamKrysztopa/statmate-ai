## StatmateAI Backend Setup Guide

### Quick Start

```bash
# 1. Install dependencies
pip install uv  # if not already installed
uv sync

# 2. Set up environment variables
cp config/.env.example .env
# Edit .env and add your OPENAI_API_KEY

# 3. Initialize database
python scripts/init_db.py --seed

# 4. Start the API server
python statmate/api/main.py
```

The API will be available at:
- **API**: http://localhost:8000
- **Docs**: http://localhost:8000/docs
- **Health**: http://localhost:8000/health

---

## Architecture Overview

```
statmate-ai/
├── statmate/api/           # FastAPI Backend
│   ├── main.py            # Application entry point
│   ├── models/            # Pydantic request/response models
│   ├── routes/            # API endpoints
│   ├── services/          # Business logic layer
│   └── scheduler/         # APScheduler configuration
├── database/              # SQLAlchemy models & session
├── config/                # Application settings
├── data/                  # Data storage (uploads, results, logs)
└── scripts/               # Utility scripts
```

---

## API Endpoints

### Datasets
- `POST /api/v1/datasets/upload` - Upload CSV/Excel file
- `GET /api/v1/datasets/` - List all datasets
- `GET /api/v1/datasets/{id}` - Get dataset details
- `GET /api/v1/datasets/{id}/preview` - Preview dataset contents
- `DELETE /api/v1/datasets/{id}` - Delete dataset

### Analysis
- `POST /api/v1/analysis/run` - Run immediate analysis
- `GET /api/v1/analysis/{id}` - Get analysis status
- `GET /api/v1/analysis/{id}/results` - Get analysis results
- `GET /api/v1/analysis/{id}/log` - Get execution log
- `GET /api/v1/analysis/` - List all analyses

### Scheduled Tasks
- `POST /api/v1/tasks/schedule` - Schedule a task
- `GET /api/v1/tasks/` - List scheduled tasks
- `GET /api/v1/tasks/{id}` - Get task details
- `PUT /api/v1/tasks/{id}/pause` - Pause task
- `PUT /api/v1/tasks/{id}/resume` - Resume task
- `DELETE /api/v1/tasks/{id}` - Delete task

### Results
- `GET /api/v1/results/` - List all results
- `GET /api/v1/results/{id}` - Get detailed result
- `GET /api/v1/results/{id}/log` - Get result log

---

## Testing the API

### Using cURL

```bash
# Upload a dataset
curl -X POST "http://localhost:8000/api/v1/datasets/upload" \
  -F "file=@data.csv" \
  -F "description=Test dataset"

# Run analysis
curl -X POST "http://localhost:8000/api/v1/analysis/run" \
  -H "Content-Type: application/json" \
  -d '{"dataset_id": "your-dataset-id", "selected_columns": ["col1", "col2"]}'

# Check status
curl "http://localhost:8000/api/v1/analysis/{analysis_id}"

# Get results
curl "http://localhost:8000/api/v1/analysis/{analysis_id}/results"
```

### Using Python

```python
import httpx

# Upload dataset
with open("data.csv", "rb") as f:
    response = httpx.post(
        "http://localhost:8000/api/v1/datasets/upload",
        files={"file": f}
    )
dataset_id = response.json()["dataset_id"]

# Run analysis
response = httpx.post(
    "http://localhost:8000/api/v1/analysis/run",
    json={"dataset_id": dataset_id}
)
analysis_id = response.json()["id"]

# Check results
import time
while True:
    response = httpx.get(f"http://localhost:8000/api/v1/analysis/{analysis_id}")
    status = response.json()["status"]
    if status in ["completed", "failed"]:
        break
    time.sleep(2)

# Get detailed results
results = httpx.get(f"http://localhost:8000/api/v1/analysis/{analysis_id}/results")
print(results.json())
```

---

## Configuration

### Environment Variables

See `config/.env.example` for all available settings:

- `OPENAI_API_KEY`: Required for LLM agents
- `DATABASE_URL`: SQLite (default) or PostgreSQL
- `DATA_DIR`: Storage directory
- `API_HOST`, `API_PORT`: Server configuration
- `CORS_ORIGINS`: Allowed origins for CORS

### Storage

Data is stored in:
- `data/uploads/`: Uploaded datasets (parquet format)
- `data/results/`: Analysis results (JSON)
- `data/logs/`: Execution logs (text)

---

## Development

### Running Tests

```bash
pytest tests/
```

### Code Quality

```bash
# Lint
ruff check statmate/

# Format
ruff format statmate/

# Type check
pyright statmate/
```

### Database Migrations

```bash
# Create migration
alembic revision --autogenerate -m "description"

# Apply migrations
alembic upgrade head

# Rollback
alembic downgrade -1
```

---

## Troubleshooting

### Port already in use
```bash
# Kill process on port 8000
lsof -ti:8000 | xargs kill -9
```

### Database locked
```bash
# Remove database and reinitialize
rm database/statmate.db
python scripts/init_db.py --seed
```

### Module not found
```bash
# Reinstall dependencies
uv sync --reinstall
```

---

## Production Deployment

See `DEVOPS_PLAN.md` for comprehensive deployment guide.

### Quick Docker Setup

```bash
# Build
docker build -t statmate-api -f docker/Dockerfile.api .

# Run
docker run -p 8000:8000 \
  -e OPENAI_API_KEY=your-key \
  -v $(pwd)/data:/app/data \
  statmate-api
```

---

## Next Steps

1. ✅ Backend complete - test the API
2. 📝 Build Streamlit UI (see `ARCHITECTURE_PROPOSAL.md`)
3. 🚀 Deploy to production (see `DEVOPS_PLAN.md`)
4. 📱 Optional: Build React/mobile frontend (V2)

---

For questions or issues, see documentation or create an issue on GitHub.

