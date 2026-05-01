# StatmateAI Quick Start Guide

## ✅ Setup Status

Your backend is **ready to run**! Here's what's been configured:

- ✅ Dependencies installed (219 packages)
- ✅ Database created and seeded with sample data
- ✅ Environment file created (`.env`)
- ✅ Data directories ready (`data/uploads`, `data/results`, `data/logs`)

---

## 🚀 Start the API Server

### Option 1: Direct Python
```bash
python statmate/api/main.py
```

### Option 2: Uvicorn with auto-reload
```bash
uvicorn statmate.api.main:app --reload --port 8000
```

**Access the API at:**
- 🌐 API Root: http://localhost:8000
- 📚 Interactive Docs: http://localhost:8000/docs
- 🏥 Health Check: http://localhost:8000/health

---

## 🔑 Add Your OpenAI API Key (Important!)

The API will work for most endpoints, but **statistical analysis requires an OpenAI API key**.

Edit the `.env` file:
```bash
# If nano is available:
nano .env

# Otherwise, edit directly in your IDE
# Change this line:
OPENAI_API_KEY=sk-placeholder-add-your-real-key-here

# To your actual key:
OPENAI_API_KEY=sk-proj-xxxxx...
```

---

## 🧪 Test the API

### 1. Check Health
```bash
curl http://localhost:8000/health
```

### 2. List Sample Datasets
```bash
curl http://localhost:8000/api/v1/datasets/
```

You should see 2 datasets:
- `patient_blood_pressure.csv` (80 rows, paired data)
- `smoking_exercise_study.csv` (200 rows, categorical data)

### 3. View Sample Data
```bash
# Get the dataset ID from step 2, then:
curl "http://localhost:8000/api/v1/datasets/{dataset-id}/preview"
```

### 4. Run Analysis
```bash
curl -X POST "http://localhost:8000/api/v1/analysis/run" \
  -H "Content-Type: application/json" \
  -d '{
    "dataset_id": "your-dataset-id",
    "selected_columns": null
  }'
```

**Note**: Analysis requires OpenAI API key to be set!

### 5. Check Results
```bash
# Using the analysis ID from step 4:
curl "http://localhost:8000/api/v1/analysis/{analysis-id}/results"
```

---

## 📊 Sample Data Available

The database was seeded with:

### Datasets
1. **patient_blood_pressure.csv**
   - 80 rows
   - Columns: `patient_id`, `before_treatment`, `after_treatment`
   - Use case: Paired t-test

2. **smoking_exercise_study.csv**
   - 200 rows
   - Columns: `user_id`, `smoker`, `exercise_level`
   - Use case: Chi-square test

### Pre-run Analyses
- 1 completed analysis
- 1 pending analysis

### Scheduled Tasks
- "Nightly Regression Analysis" (recurring)
- "Weekly User Cohort Report" (recurring)

---

## 🎯 Interactive API Documentation

The easiest way to test the API is through the **interactive docs**:

1. Start the server (see above)
2. Open http://localhost:8000/docs in your browser
3. Click on any endpoint to expand it
4. Click "Try it out" to test
5. Fill in parameters and click "Execute"

All endpoints are documented with:
- Request/response schemas
- Example values
- Error codes
- Try-it-out functionality

---

## 📁 Project Structure

```
statmate-ai/
├── .env                    # ✅ Your environment config
├── database/
│   └── statmate.db         # ✅ SQLite database (seeded)
├── data/
│   ├── uploads/            # Uploaded datasets go here
│   ├── results/            # Analysis results stored here
│   └── logs/               # Execution logs
├── statmate/
│   ├── api/                # ✅ FastAPI backend
│   │   ├── main.py         # Start here!
│   │   ├── routes/         # API endpoints
│   │   ├── services/       # Business logic
│   │   └── models/         # Request/response models
│   ├── workflow/           # Your existing LangGraph workflow
│   ├── agents/             # Your existing LLM agents
│   └── statistical_core/   # Your existing statistical tests
└── scripts/
    ├── init_db.py          # Database initialization
    └── run_dev.sh          # Development startup
```

---

## 🔍 Common Tasks

### Upload a New Dataset
```bash
curl -X POST "http://localhost:8000/api/v1/datasets/upload" \
  -F "file=@path/to/your/data.csv" \
  -F "description=My analysis dataset"
```

### Schedule a Recurring Task
```bash
curl -X POST "http://localhost:8000/api/v1/tasks/schedule" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Daily Analysis",
    "task_type": "recurring",
    "dataset_id": "your-dataset-id",
    "schedule": "0 2 * * *"
  }'
```

### View All Results
```bash
curl "http://localhost:8000/api/v1/results/"
```

---

## 🐛 Troubleshooting

### Port 8000 already in use
```bash
# Kill the process
lsof -ti:8000 | xargs kill -9

# Or use a different port
uvicorn statmate.api.main:app --port 8001
```

### Database errors
```bash
# Reset database
rm database/statmate.db
python scripts/init_db.py --seed
```

### Module not found
```bash
# Reinstall dependencies
uv sync
```

### OpenAI API errors
- Make sure `OPENAI_API_KEY` is set in `.env`
- Verify the key is valid
- Check you have credits available

---

## 📚 Full Documentation

- **Setup Guide**: `BACKEND_SETUP.md`
- **Architecture**: `ARCHITECTURE_PROPOSAL.md`
- **DevOps**: `DEVOPS_PLAN.md`
- **Status**: `IMPLEMENTATION_STATUS.md`

---

## ⏭️ Next Steps

1. **Test the API** - Start the server and try the endpoints
2. **Add your OpenAI key** - Required for running analyses
3. **Upload your own data** - Try with real datasets
4. **Build the UI** - Streamlit frontend is next!

---

## 🎉 You're Ready!

Everything is configured and working. Just run:

```bash
python statmate/api/main.py
```

Then visit: **http://localhost:8000/docs** 🚀

