# StatmateAI Architecture Proposal
## Backend (FastAPI) + Frontend (Streamlit) + Scheduling

---

## 📁 Proposed Directory Structure

```
statmate-ai/
├── statmate/
│   ├── core/                    # [Existing] Core functionality
│   ├── agents/                  # [Existing] Statistical agents
│   ├── statistical_core/        # [Existing] Statistical tests
│   ├── workflow/                # [Existing] LangGraph workflow
│   │
│   ├── api/                     # [NEW] FastAPI Backend
│   │   ├── __init__.py
│   │   ├── main.py             # FastAPI app initialization
│   │   ├── dependencies.py     # Shared dependencies (DB, services)
│   │   ├── models/             # Pydantic models for API
│   │   │   ├── __init__.py
│   │   │   ├── dataset.py      # Dataset upload/storage models
│   │   │   ├── analysis.py     # Analysis request/response models
│   │   │   ├── task.py         # Scheduled task models
│   │   │   └── result.py       # Result/output models
│   │   ├── routes/             # API endpoints
│   │   │   ├── __init__.py
│   │   │   ├── datasets.py     # Dataset upload/management
│   │   │   ├── analysis.py     # Run analysis (immediate)
│   │   │   ├── tasks.py        # Schedule/manage tasks
│   │   │   └── results.py      # Retrieve results
│   │   ├── services/           # Business logic
│   │   │   ├── __init__.py
│   │   │   ├── dataset_service.py    # Dataset handling
│   │   │   ├── analysis_service.py   # Wraps LangGraph workflow
│   │   │   ├── task_service.py       # Task management
│   │   │   └── storage_service.py    # File storage
│   │   └── scheduler/          # Task scheduling
│   │       ├── __init__.py
│   │       ├── scheduler.py    # APScheduler configuration
│   │       └── jobs.py         # Scheduled job definitions
│   │
│   └── ui/                      # [NEW] Streamlit Frontend
│       ├── __init__.py
│       ├── app.py              # Main Streamlit app
│       ├── components/         # Reusable UI components
│       │   ├── __init__.py
│       │   ├── file_upload.py        # File upload component
│       │   ├── data_preview.py       # Data preview & selection
│       │   ├── analysis_config.py    # Analysis configuration
│       │   ├── results_display.py    # Results visualization
│       │   └── task_manager.py       # Scheduled tasks UI
│       ├── utils/              # UI utilities
│       │   ├── __init__.py
│       │   ├── api_client.py   # API communication
│       │   ├── session.py      # Session state management
│       │   └── formatters.py   # Data formatting
│       └── pages/              # Multi-page app (optional)
│           ├── 1_upload.py
│           ├── 2_analysis.py
│           └── 3_results.py
│
├── data/                        # [NEW] Data storage
│   ├── uploads/                # Uploaded datasets
│   ├── results/                # Analysis results
│   └── logs/                   # Analysis logs
│
├── database/                    # [NEW] Database (SQLite for simplicity)
│   ├── statmate.db
│   └── migrations/
│
├── config/                      # [NEW] Configuration
│   ├── settings.py             # Application settings
│   └── .env.example            # Environment variables template
│
├── tests/
│   ├── api/                    # API tests
│   └── ui/                     # UI tests
│
├── docker/                      # [NEW] Docker configurations
│   ├── Dockerfile.api
│   ├── Dockerfile.ui
│   └── docker-compose.yml
│
├── scripts/                     # Utility scripts
│   ├── init_db.py              # Database initialization
│   └── run_dev.sh              # Development startup script
│
├── pyproject.toml              # [UPDATE] Add new dependencies
├── README.md
└── .env                        # Environment variables (gitignored)
```

---

## 🏗️ Backend Architecture (FastAPI)

### 1. Core Components

#### **API Layer** (`statmate/api/routes/`)
- **`datasets.py`**: Upload, list, delete datasets
- **`analysis.py`**: Run immediate analysis
- **`tasks.py`**: Create, list, cancel scheduled tasks
- **`results.py`**: Retrieve analysis results and logs

#### **Service Layer** (`statmate/api/services/`)
- **`analysis_service.py`**: Wraps your existing LangGraph workflow
  - Converts uploaded data to workflow state
  - Executes `compiled.stream()` from `statmate_flow.py`
  - Captures results and logs
  
- **`dataset_service.py`**: 
  - Validates uploaded files (CSV/Excel)
  - Stores files with metadata
  - Provides data preview and column info
  
- **`task_service.py`**:
  - CRUD operations for scheduled tasks
  - Interfaces with APScheduler
  
- **`storage_service.py`**:
  - File I/O operations
  - Result serialization/deserialization

#### **Scheduler** (`statmate/api/scheduler/`)
- **APScheduler** for background task management
- Supports:
  - One-time scheduled tasks
  - Recurring tasks (nightly, weekly)
  - Cron expressions

#### **Database Models** (SQLite + SQLAlchemy)
```python
# Dataset
- id (UUID)
- filename
- original_filename
- upload_timestamp
- file_size
- column_names (JSON)
- data_types (JSON)

# Analysis
- id (UUID)
- dataset_id (FK)
- status (pending, running, completed, failed)
- selected_columns (JSON)
- start_time
- end_time
- result_path
- log_path

# ScheduledTask
- id (UUID)
- task_type (one_time, recurring)
- schedule (cron or datetime)
- dataset_id (FK)
- selected_columns (JSON)
- status (active, paused, completed)
- next_run
- last_run
- created_at
```

### 2. API Endpoints

```
POST   /api/v1/datasets/upload           # Upload CSV/Excel
GET    /api/v1/datasets/                 # List datasets
GET    /api/v1/datasets/{id}             # Get dataset details
GET    /api/v1/datasets/{id}/preview     # Preview data
DELETE /api/v1/datasets/{id}             # Delete dataset

POST   /api/v1/analysis/run              # Run immediate analysis
GET    /api/v1/analysis/{id}             # Get analysis status
GET    /api/v1/analysis/{id}/results     # Get results
GET    /api/v1/analysis/{id}/log         # Get execution log

POST   /api/v1/tasks/schedule            # Schedule a task
GET    /api/v1/tasks/                    # List scheduled tasks
GET    /api/v1/tasks/{id}                # Get task details
PUT    /api/v1/tasks/{id}/pause          # Pause task
PUT    /api/v1/tasks/{id}/resume         # Resume task
DELETE /api/v1/tasks/{id}                # Delete task

GET    /api/v1/results/                  # List all results
GET    /api/v1/results/{id}              # Get specific result
```

### 3. Key Technologies

- **FastAPI**: REST API framework
- **SQLAlchemy**: ORM for database
- **APScheduler**: Background task scheduling
- **Pandas**: Data handling (already in your stack)
- **Pydantic**: Request/response validation
- **Uvicorn**: ASGI server

---

## 🎨 Frontend Architecture (Streamlit)

### 1. Main App Flow (`statmate/ui/app.py`)

```python
# Single-page app with step-by-step workflow:
1. File Upload Section
2. Data Preview & Column Selection
3. Analysis Configuration
4. Results Display
5. Scheduled Tasks Panel (sidebar)
```

### 2. Components (`statmate/ui/components/`)

#### **`file_upload.py`**
```python
def render_file_upload():
    """File upload widget, calls API to upload."""
    uploaded_file = st.file_uploader("Upload CSV or Excel", type=['csv', 'xlsx'])
    if uploaded_file:
        # Call API: POST /api/v1/datasets/upload
        # Store dataset_id in session state
```

#### **`data_preview.py`**
```python
def render_data_preview(dataset_id):
    """Display data table with column selection."""
    # Call API: GET /api/v1/datasets/{id}/preview
    # Show dataframe with st.dataframe()
    # Render column checkboxes for selection
```

#### **`analysis_config.py`**
```python
def render_analysis_controls(dataset_id, selected_columns):
    """Run or schedule analysis."""
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Run Stat Test"):
            # Call API: POST /api/v1/analysis/run
    with col2:
        if st.button("Schedule Task"):
            # Show scheduling dialog
            # Call API: POST /api/v1/tasks/schedule
```

#### **`results_display.py`**
```python
def render_results(analysis_id):
    """Display statistical tree, logs, and summaries."""
    # Call API: GET /api/v1/analysis/{id}/results
    # Render:
    # - Statistical summary text
    # - Decision tree visualization (graphviz/mermaid)
    # - Log file in expandable section
```

#### **`task_manager.py`**
```python
def render_scheduled_tasks():
    """Sidebar panel for scheduled tasks."""
    # Call API: GET /api/v1/tasks/
    # Display list with status badges
    # Controls: pause, resume, delete
```

### 3. API Client (`statmate/ui/utils/api_client.py`)

```python
class StatmateAPIClient:
    """Wrapper for all API calls."""
    
    def __init__(self, base_url: str):
        self.base_url = base_url
    
    def upload_dataset(self, file) -> dict:
        ...
    
    def run_analysis(self, dataset_id: str, columns: list) -> dict:
        ...
    
    def get_results(self, analysis_id: str) -> dict:
        ...
    
    def schedule_task(self, dataset_id: str, schedule: str, ...) -> dict:
        ...
```

### 4. Session State Management

```python
# Store in st.session_state:
{
    'uploaded_dataset_id': str,
    'selected_columns': list[str],
    'current_analysis_id': str,
    'results': dict,
    'scheduled_tasks': list[dict]
}
```

---

## 🔄 Integration Flow

### Immediate Analysis Flow
```
User → Streamlit UI → FastAPI → LangGraph Workflow → FastAPI → Streamlit UI
                          ↓
                      Store Results
```

### Scheduled Task Flow
```
User → Streamlit → FastAPI → APScheduler (store task)
                                   ↓
                        [At scheduled time]
                                   ↓
                      APScheduler → FastAPI → LangGraph → Store Results
                                                                ↓
User → Streamlit → FastAPI → Retrieve Results
```

---

## 🗄️ Data Storage Strategy

### File Storage
```
data/
├── uploads/
│   └── {dataset_id}.parquet        # Efficient storage
├── results/
│   └── {analysis_id}/
│       ├── summary.json            # Structured results
│       ├── tree.png                # Decision tree viz
│       └── log.txt                 # Execution log
```

### Database (SQLite)
- Metadata only (datasets, analyses, tasks)
- Lightweight, no separate server needed
- Easy to upgrade to PostgreSQL later

---

## 🚀 Deployment Strategy

### Development
```bash
# Terminal 1: Start FastAPI backend
uvicorn statmate.api.main:app --reload --port 8000

# Terminal 2: Start Streamlit frontend
streamlit run statmate/ui/app.py --server.port 8501
```

### Production (Docker Compose)
```yaml
services:
  api:
    build: ./docker/Dockerfile.api
    ports: ["8000:8000"]
    volumes: ["./data:/app/data"]
    
  ui:
    build: ./docker/Dockerfile.ui
    ports: ["8501:8501"]
    environment:
      API_URL: http://api:8000
    depends_on: [api]
```

---

## 📦 New Dependencies to Add

```toml
[project.dependencies]
# Existing: fastapi, streamlit, pydantic, pandas, etc.

# New:
"sqlalchemy>=2.0.0",           # ORM
"alembic>=1.12.0",             # DB migrations
"apscheduler>=3.10.0",         # Task scheduling
"aiosqlite>=0.19.0",           # Async SQLite
"python-multipart>=0.0.6",     # File uploads
"httpx>=0.25.0",               # HTTP client for UI→API
"python-dotenv>=1.0.0",        # Environment variables
"openpyxl>=3.1.0",             # Excel support
"pyarrow>=14.0.0",             # Parquet storage
```

---

## 🔐 Configuration Management

### `.env` file
```bash
# API
API_HOST=0.0.0.0
API_PORT=8000
DATABASE_URL=sqlite:///./database/statmate.db
DATA_DIR=./data

# OpenAI (for LLM agents)
OPENAI_API_KEY=your_key_here

# Streamlit UI
STREAMLIT_SERVER_PORT=8501
API_BASE_URL=http://localhost:8000
```

---

## ✅ Implementation Phases

### Phase 1: Backend Foundation
1. ✅ Set up FastAPI project structure
2. ✅ Create database models & migrations
3. ✅ Implement dataset upload/storage
4. ✅ Wrap LangGraph workflow in service layer
5. ✅ Create immediate analysis endpoint

### Phase 2: Backend Scheduling
6. ✅ Set up APScheduler
7. ✅ Implement task scheduling endpoints
8. ✅ Create background job execution
9. ✅ Add task status tracking

### Phase 3: Frontend Core
10. ✅ Set up Streamlit app structure
11. ✅ Build file upload component
12. ✅ Create data preview & selection
13. ✅ Implement "Run Test" functionality
14. ✅ Build results display

### Phase 4: Frontend Scheduling
15. ✅ Create scheduled tasks UI
16. ✅ Add task management controls
17. ✅ Implement real-time status updates

### Phase 5: Polish & Deploy
18. ✅ Add error handling & validation
19. ✅ Create Docker setup
20. ✅ Write documentation
21. ✅ Add tests

---

## 🎯 Next Steps

1. **Review this proposal** - Any changes needed?
2. **Start with Phase 1** - Backend foundation
3. **Iterate** - Test each phase before moving forward

---

## 💡 Design Decisions Explained

### Why SQLite?
- Simple, serverless, perfect for MVP
- Easy migration to PostgreSQL if needed
- No additional infrastructure

### Why APScheduler?
- Lightweight, Python-native
- Supports various triggers (cron, interval, date)
- Integrates well with FastAPI

### Why Single-Page Streamlit?
- Simpler state management
- Matches your UI mockup flow
- Can easily split into multi-page later

### Why Parquet for storage?
- Efficient columnar format
- Preserves data types
- Fast read/write with pandas

---

## 🚀 Frontend V2: Modern Stack + Mobile Apps

Once the FastAPI backend is stable, we can build a more sophisticated frontend using modern frameworks.

### Option A: React + TypeScript (Web)

#### Technology Stack
```
Frontend Framework: React 18+ with TypeScript
State Management: Zustand or Redux Toolkit
API Client: TanStack Query (React Query)
UI Library: shadcn/ui + Tailwind CSS
Data Visualization: Recharts, D3.js, or Plotly.js
Forms: React Hook Form + Zod validation
Build Tool: Vite
Testing: Vitest + React Testing Library
```

#### Project Structure
```
statmate-web/
├── src/
│   ├── components/
│   │   ├── ui/                  # shadcn/ui components
│   │   ├── features/
│   │   │   ├── FileUpload/
│   │   │   │   ├── FileUpload.tsx
│   │   │   │   ├── FileUploadModal.tsx
│   │   │   │   └── useFileUpload.ts
│   │   │   ├── DataPreview/
│   │   │   │   ├── DataTable.tsx
│   │   │   │   ├── ColumnSelector.tsx
│   │   │   │   └── useDataPreview.ts
│   │   │   ├── Analysis/
│   │   │   │   ├── AnalysisPanel.tsx
│   │   │   │   ├── RunAnalysisButton.tsx
│   │   │   │   └── useAnalysis.ts
│   │   │   ├── Results/
│   │   │   │   ├── ResultsView.tsx
│   │   │   │   ├── StatisticalTree.tsx
│   │   │   │   ├── LogViewer.tsx
│   │   │   │   └── useResults.ts
│   │   │   └── Tasks/
│   │   │       ├── TaskScheduler.tsx
│   │   │       ├── TaskList.tsx
│   │   │       └── useTasks.ts
│   ├── lib/
│   │   ├── api/                 # API client
│   │   │   ├── client.ts        # Axios/Fetch wrapper
│   │   │   ├── datasets.ts      # Dataset endpoints
│   │   │   ├── analysis.ts      # Analysis endpoints
│   │   │   ├── tasks.ts         # Task endpoints
│   │   │   └── types.ts         # TypeScript types
│   │   └── utils/
│   ├── hooks/
│   │   ├── useWebSocket.ts      # Real-time updates
│   │   └── usePolling.ts        # Status polling
│   ├── stores/
│   │   ├── datasetStore.ts
│   │   ├── analysisStore.ts
│   │   └── taskStore.ts
│   ├── pages/
│   │   ├── Dashboard.tsx
│   │   ├── Upload.tsx
│   │   ├── Analysis.tsx
│   │   ├── Results.tsx
│   │   └── Tasks.tsx
│   └── App.tsx
├── package.json
├── vite.config.ts
└── tsconfig.json
```

#### Key Features
- **Real-time updates** via WebSocket or Server-Sent Events
- **Responsive design** - works on desktop, tablet, mobile
- **Progressive Web App (PWA)** - installable, offline-capable
- **Advanced data tables** - sorting, filtering, pagination
- **Interactive visualizations** - zoom, pan, export
- **Dark mode** - theme switching
- **Accessibility** - WCAG compliant

### Option B: Next.js (Full-Stack Web)

#### Technology Stack
```
Framework: Next.js 14+ (App Router)
Language: TypeScript
State: Zustand + React Query
UI: shadcn/ui + Tailwind CSS
Auth: NextAuth.js (if user accounts needed)
Deployment: Vercel or self-hosted
```

#### Advantages
- Server-side rendering (SSR) for better SEO
- API routes (could proxy to FastAPI or replace some endpoints)
- Built-in routing and optimization
- Excellent developer experience

### Option C: Vue 3 + Nuxt (Alternative)

#### Technology Stack
```
Framework: Vue 3 + Nuxt 3
Language: TypeScript
State: Pinia
UI: Nuxt UI or PrimeVue
API: useFetch composable
```

#### Advantages
- Simpler learning curve than React
- Excellent documentation
- Great performance
- Growing ecosystem

---

## 📱 Mobile Apps

### Option 1: React Native (iOS + Android)

#### Technology Stack
```
Framework: React Native + Expo
Language: TypeScript
Navigation: React Navigation
State: Zustand + React Query
UI: React Native Paper or NativeBase
Charts: Victory Native or Recharts Native
```

#### Project Structure
```
statmate-mobile/
├── src/
│   ├── screens/
│   │   ├── UploadScreen.tsx
│   │   ├── DataPreviewScreen.tsx
│   │   ├── AnalysisScreen.tsx
│   │   ├── ResultsScreen.tsx
│   │   └── TasksScreen.tsx
│   ├── components/
│   │   ├── FileUploader.tsx
│   │   ├── DataTable.tsx
│   │   ├── ResultsCard.tsx
│   │   └── TaskCard.tsx
│   ├── services/
│   │   └── api.ts              # Shared with web
│   ├── navigation/
│   │   └── AppNavigator.tsx
│   └── App.tsx
├── app.json
└── package.json
```

#### Key Features
- **Native feel** - smooth animations, gestures
- **Camera integration** - scan data tables (OCR)
- **Push notifications** - task completion alerts
- **Offline mode** - queue tasks for later
- **Biometric auth** - Face ID, Touch ID
- **Share results** - export to PDF, share via apps

### Option 2: Flutter (iOS + Android + Web)

#### Technology Stack
```
Framework: Flutter
Language: Dart
State: Riverpod or Bloc
UI: Material Design 3
API: Dio + Retrofit
Charts: fl_chart
```

#### Advantages
- Single codebase for iOS, Android, Web
- Excellent performance
- Beautiful native-looking UI
- Strong typing with Dart

### Option 3: Progressive Web App (PWA)

Convert the React/Next.js web app into a PWA:

#### Features
- **Installable** - add to home screen
- **Offline-first** - service workers
- **App-like** - full screen, splash screen
- **Push notifications** - web push API
- **Native capabilities** - camera, file system access

#### Advantages
- No app store submission needed
- Single codebase for web + mobile
- Automatic updates
- Lower development cost

---

## 🔄 Backend Enhancements for Frontend V2

### Add WebSocket Support
```python
# statmate/api/websockets.py
from fastapi import WebSocket

@app.websocket("/ws/analysis/{analysis_id}")
async def analysis_updates(websocket: WebSocket, analysis_id: str):
    """Stream real-time analysis progress."""
    await websocket.accept()
    # Stream progress updates
```

### Add Server-Sent Events (SSE)
```python
from fastapi.responses import StreamingResponse

@app.get("/api/v1/analysis/{id}/stream")
async def stream_analysis(id: str):
    """Stream analysis progress via SSE."""
    async def event_generator():
        # Yield progress updates
        yield f"data: {json.dumps(progress)}\n\n"
    return StreamingResponse(event_generator(), media_type="text/event-stream")
```

### Enhanced API Features
- **Pagination** - for large datasets
- **Filtering & Sorting** - query parameters
- **Batch operations** - multiple analyses
- **Export formats** - PDF, Excel, JSON
- **User authentication** - JWT tokens (if multi-user)
- **Rate limiting** - prevent abuse
- **API versioning** - /v1, /v2
- **GraphQL** (optional) - alternative to REST

---

## 📊 Comparison: Streamlit vs Modern Frontend

| Feature               | Streamlit (V1) | React/Next.js (V2) | React Native (Mobile) |
| --------------------- | -------------- | ------------------ | --------------------- |
| **Development Speed** | ⚡⚡⚡ Very Fast  | ⚡⚡ Fast            | ⚡⚡ Fast               |
| **Customization**     | ⚡⚡ Limited     | ⚡⚡⚡ Full Control   | ⚡⚡⚡ Full Control      |
| **Performance**       | ⚡⚡ Good        | ⚡⚡⚡ Excellent      | ⚡⚡⚡ Native            |
| **Real-time Updates** | ⚡ Polling      | ⚡⚡⚡ WebSocket      | ⚡⚡⚡ WebSocket         |
| **Mobile Experience** | ⚡ Basic        | ⚡⚡ Responsive      | ⚡⚡⚡ Native            |
| **State Management**  | ⚡⚡ Session     | ⚡⚡⚡ Advanced       | ⚡⚡⚡ Advanced          |
| **UI/UX Quality**     | ⚡⚡ Standard    | ⚡⚡⚡ Custom         | ⚡⚡⚡ Native            |
| **Deployment**        | ⚡⚡⚡ Simple     | ⚡⚡ Moderate        | ⚡ App Stores          |
| **Learning Curve**    | ⚡⚡⚡ Easy       | ⚡⚡ Moderate        | ⚡⚡ Moderate           |
| **Team Skills**       | 🐍 Python       | ⚛️ React/TS         | ⚛️ React/TS            |

---

## 🎯 Recommended Roadmap

### Phase 1: MVP (Now)
- ✅ FastAPI Backend
- ✅ Streamlit Frontend
- 🎯 **Goal**: Working prototype, user testing

### Phase 2: Production Web (3-6 months)
- ✅ Enhanced FastAPI (WebSocket, auth)
- ✅ React + TypeScript Frontend
- ✅ PWA support
- 🎯 **Goal**: Production-ready web app

### Phase 3: Mobile Apps (6-12 months)
- ✅ React Native iOS/Android
- ✅ Push notifications
- ✅ Offline support
- 🎯 **Goal**: App Store + Google Play

### Phase 4: Advanced Features (12+ months)
- ✅ Multi-user collaboration
- ✅ Team workspaces
- ✅ Advanced visualizations
- ✅ API for third-party integrations
- 🎯 **Goal**: Enterprise-ready platform

---

## 🛠️ Technology Decision Matrix

### When to Use What

**Streamlit (V1)** ✅ Use when:
- Building MVP/prototype quickly
- Team is primarily Python developers
- Internal tool or academic use
- Budget/time is limited

**React/Next.js (V2)** ✅ Use when:
- Need professional, custom UI
- Building SaaS product
- Have frontend developers
- Need advanced interactivity

**React Native** ✅ Use when:
- Mobile-first users
- Need native features (camera, notifications)
- Want code sharing with web
- Target both iOS and Android

**Flutter** ✅ Use when:
- Want single codebase for all platforms
- Need highest performance
- Team knows Dart or willing to learn
- Want beautiful, consistent UI

**PWA** ✅ Use when:
- Want mobile without app stores
- Need quick deployment
- Users prefer web-based tools
- Limited mobile development resources

---

## 📦 Additional Dependencies for Frontend V2

### Backend Enhancements
```toml
[project.dependencies]
# Existing + new for V2:
"fastapi-websocket>=0.1.0",     # WebSocket support
"sse-starlette>=1.6.0",         # Server-Sent Events
"python-jose[cryptography]",    # JWT auth
"passlib[bcrypt]",              # Password hashing
"slowapi>=0.1.9",               # Rate limiting
"fastapi-pagination>=0.12.0",   # Pagination
"redis>=5.0.0",                 # Caching (optional)
```

### Frontend (React)
```json
{
  "dependencies": {
    "react": "^18.2.0",
    "react-dom": "^18.2.0",
    "typescript": "^5.0.0",
    "@tanstack/react-query": "^5.0.0",
    "zustand": "^4.4.0",
    "axios": "^1.6.0",
    "tailwindcss": "^3.3.0",
    "shadcn-ui": "latest",
    "recharts": "^2.10.0",
    "react-hook-form": "^7.48.0",
    "zod": "^3.22.0"
  }
}
```

---

## 🔐 Security Considerations for Production

- **Authentication**: JWT tokens, OAuth2
- **Authorization**: Role-based access control
- **Data encryption**: At rest and in transit
- **API rate limiting**: Prevent abuse
- **Input validation**: Sanitize all inputs
- **CORS**: Proper configuration
- **HTTPS**: SSL certificates
- **File upload**: Size limits, type checking
- **SQL injection**: Parameterized queries (SQLAlchemy handles this)

---

**Ready to start building?** 🚀

### Immediate Next Steps:
1. **Now**: Build FastAPI backend + Streamlit V1 (MVP)
2. **Later**: Evaluate React vs Next.js for V2 based on user feedback
3. **Future**: Mobile apps when user base justifies investment

