# Service Layer Documentation

**Purpose**: The service layer contains the business logic of StatmateAI, separating concerns between API routes (HTTP handling) and domain logic (statistical operations, file management, etc.).

**Location**: `statmate/api/services/`

---

## Architecture Pattern

```
┌─────────────────────┐
│   API Routes        │  HTTP Request/Response handling
│   (routes/*.py)     │  Parameter validation
└──────────┬──────────┘  Error formatting
           │
           ▼ Calls service methods
┌─────────────────────┐
│  Service Layer      │  Business logic
│  (services/*.py)    │  Data transformation
└──────────┬──────────┘  Workflow orchestration
           │
           ▼ Interacts with
┌─────────────────────┐
│  Data Layer         │  Database operations
│  (database/models)  │  File I/O
└─────────────────────┘  External APIs
```

**Benefits**:
- **Testability**: Services can be tested without HTTP layer
- **Reusability**: Same service methods used by API, CLI, or other interfaces
- **Maintainability**: Changes to business logic don't affect routes
- **Separation of Concerns**: Each layer has a single responsibility

---

## Services Overview

| Service           | Purpose                   | Dependencies                              |
| ----------------- | ------------------------- | ----------------------------------------- |
| `StorageService`  | File I/O operations       | Settings, Pandas                          |
| `DatasetService`  | Dataset lifecycle         | StorageService, Database                  |
| `AnalysisService` | Analysis execution        | DatasetService, StorageService, LangGraph |
| `TaskService`     | Scheduled task management | Database, APScheduler                     |

---

## StorageService

**File**: `statmate/api/services/storage_service.py`

### Purpose

Handles all file system operations for:
- Dataset uploads (CSV, Excel → Parquet)
- Dataset reading (Parquet → DataFrame)
- Results storage (JSON)
- Log file management (Text)

### Key Features

- **Format Detection**: Automatically detects CSV, Excel, Parquet
- **Efficient Storage**: Converts all datasets to Parquet for performance
- **Type Preservation**: Maintains data types across save/load cycles
- **Timestamp Naming**: Unique filenames with timestamps to prevent collisions
- **Error Handling**: Graceful failure with detailed error messages

### Methods

#### `save_upload(file: UploadFile) -> tuple[str, int]`

Save uploaded file to temporary location.

**Parameters**:
- `file`: FastAPI UploadFile object

**Returns**:
- `(filepath, file_size)`: Path to saved file and size in bytes

**Usage**:
```python
from statmate.api.services.storage_service import StorageService

filepath, size = StorageService.save_upload(uploaded_file)
```

**Notes**:
- Creates `data/uploads/` directory if not exists
- Generates unique filename with timestamp
- Preserves original file extension

---

#### `read_dataset(filepath: str) -> pd.DataFrame`

Read dataset from any supported format.

**Parameters**:
- `filepath`: Path to dataset file

**Returns**:
- `pd.DataFrame`: Loaded dataset

**Supported Formats**:
- `.csv` - Comma-separated values
- `.xlsx`, `.xls` - Excel workbooks (reads first sheet)
- `.parquet` - Apache Parquet (preferred)

**Usage**:
```python
df = StorageService.read_dataset("data/uploads/myfile.csv")
```

**Errors**:
- `ValueError`: Unsupported file format
- `FileNotFoundError`: File does not exist

---

#### `save_dataset(df: pd.DataFrame, filename: str) -> str`

Save DataFrame as Parquet file.

**Parameters**:
- `df`: DataFrame to save
- `filename`: Original filename (will be converted to .parquet)

**Returns**:
- `str`: Path to saved Parquet file

**Usage**:
```python
parquet_path = StorageService.save_dataset(df, "mydata.csv")
# Returns: "data/uploads/20251016_080000_mydata.parquet"
```

**Features**:
- Timestamp prefix for uniqueness
- Automatic .parquet extension
- Efficient columnar storage
- Preserves dtypes, nulls, index

---

#### `save_results(analysis_id: str, results: dict) -> str`

Save analysis results as JSON.

**Parameters**:
- `analysis_id`: UUID of analysis
- `results`: Dictionary of results to save

**Returns**:
- `str`: Path to saved JSON file

**Usage**:
```python
result_path = StorageService.save_results(
    analysis_id="550e8400-e29b-41d4-a716-446655440000",
    results={
        "test": "t-test",
        "p_value": 0.0023,
        "statistic": 3.45
    }
)
```

**Storage**: `data/results/{analysis_id}.json`

---

#### `save_log(analysis_id: str, log_content: str) -> str`

Save execution log as text file.

**Parameters**:
- `analysis_id`: UUID of analysis
- `log_content`: Log text to save

**Returns**:
- `str`: Path to saved log file

**Usage**:
```python
log_path = StorageService.save_log(
    analysis_id="550e8400-e29b-41d4-a716-446655440000",
    log_content="Analysis started...\nCompleted successfully"
)
```

**Storage**: `data/logs/{analysis_id}.log`

---

#### `read_results(result_path: str) -> dict`

Load results from JSON file.

**Parameters**:
- `result_path`: Path to JSON file

**Returns**:
- `dict`: Loaded results

**Usage**:
```python
results = StorageService.read_results("data/results/550e8400.json")
```

---

#### `read_log(log_path: str) -> str`

Load log file content.

**Parameters**:
- `log_path`: Path to log file

**Returns**:
- `str`: Log content

**Usage**:
```python
log_content = StorageService.read_log("data/logs/550e8400.log")
```

---

#### `delete_file(filepath: str) -> bool`

Delete a file from the filesystem.

**Parameters**:
- `filepath`: Path to file to delete

**Returns**:
- `bool`: True if deleted, False if file didn't exist

**Usage**:
```python
success = StorageService.delete_file("data/uploads/old_file.parquet")
```

---

## DatasetService

**File**: `statmate/api/services/dataset_service.py`

### Purpose

Manages the complete lifecycle of datasets:
- Upload and conversion
- Metadata extraction
- Preview generation
- Retrieval
- Deletion

### Methods

#### `create_dataset(db: Session, file: UploadFile, description: str | None) -> Dataset`

Upload and create new dataset.

**Workflow**:
1. Save uploaded file via `StorageService`
2. Read into DataFrame
3. Extract metadata (columns, types, row count)
4. Convert to Parquet
5. Create database record
6. Delete original upload

**Parameters**:
- `db`: Database session
- `file`: Uploaded file
- `description`: Optional description

**Returns**:
- `Dataset`: Created database model

**Usage**:
```python
from database import get_db

def upload_endpoint(file: UploadFile, db: Session = Depends(get_db)):
    dataset = DatasetService.create_dataset(db, file, "My dataset")
    return dataset
```

**Error Handling**:
- Cleans up files on failure
- Rolls back database transaction
- Raises descriptive errors

---

#### `get_dataset(db: Session, dataset_id: str) -> Dataset | None`

Retrieve dataset by ID.

**Parameters**:
- `db`: Database session
- `dataset_id`: UUID string

**Returns**:
- `Dataset` if found, `None` otherwise

**Usage**:
```python
dataset = DatasetService.get_dataset(db, "550e8400...")
if not dataset:
    raise HTTPException(status_code=404)
```

---

#### `list_datasets(db: Session, skip: int, limit: int) -> list[Dataset]`

List datasets with pagination.

**Parameters**:
- `db`: Database session
- `skip`: Records to skip (offset)
- `limit`: Max records to return

**Returns**:
- `list[Dataset]`: List of dataset records

**Usage**:
```python
datasets = DatasetService.list_datasets(db, skip=0, limit=10)
```

---

#### `delete_dataset(db: Session, dataset_id: str) -> bool`

Delete dataset and associated files.

**Workflow**:
1. Get dataset record
2. Delete Parquet file
3. Delete all associated analyses
4. Delete results files for analyses
5. Delete log files for analyses
6. Delete dataset record from database

**Parameters**:
- `db`: Database session
- `dataset_id`: UUID string

**Returns**:
- `bool`: True if deleted, False if not found

**Usage**:
```python
success = DatasetService.delete_dataset(db, "550e8400...")
```

**Cascade Deletes**:
- All analyses for this dataset
- All result files
- All log files
- Dataset file itself

---

#### `get_dataset_preview(db: Session, dataset_id: str, num_rows: int) -> dict`

Generate preview of dataset rows.

**Parameters**:
- `db`: Database session
- `dataset_id`: UUID string
- `num_rows`: Number of rows to preview

**Returns**:
- `dict`: Preview data with metadata

**Response Structure**:
```python
{
    "dataset_id": "550e8400...",
    "original_filename": "mydata.csv",
    "row_count": 100,
    "column_names": ["col1", "col2"],
    "data_types": {"col1": "int64", "col2": "float64"},
    "preview_data": [
        {"col1": 1, "col2": 1.5},
        {"col1": 2, "col2": 2.5}
    ],
    "preview_rows": 2
}
```

**Usage**:
```python
preview = DatasetService.get_dataset_preview(db, "550e8400...", num_rows=5)
```

---

#### `load_dataset_dataframe(db: Session, dataset_id: str) -> pd.DataFrame`

Load full dataset as DataFrame for analysis.

**Parameters**:
- `db`: Database session
- `dataset_id`: UUID string

**Returns**:
- `pd.DataFrame`: Full dataset

**Usage**:
```python
df = DatasetService.load_dataset_dataframe(db, "550e8400...")
# Now run analysis on df
```

**Note**: Loads entire dataset into memory. For large datasets (>1GB), consider chunking.

---

## AnalysisService

**File**: `statmate/api/services/analysis_service.py`

### Purpose

**The Core Service** - Orchestrates statistical analysis execution:
- Creates analysis records
- Loads datasets
- Executes LangGraph workflow
- Captures results and logs
- Updates status

### Integration with Existing Code

This service wraps your existing `statmate.workflow.statmate_flow.compiled` workflow without modifying it.

```python
from statmate.workflow.statmate_flow import compiled  # Your existing workflow

def run_analysis(db: Session, analysis_id: str):
    # Load data
    df = DatasetService.load_dataset_dataframe(db, dataset_id)
    
    # Prepare state (YOUR existing state structure)
    initial_state = {
        'df': df,
        'secondary_df': None,
        'target_columns': [],
        'paired': False,
        'data_type': None,
        'do_association': False,
        'number_of_samples': 0,
        'results': [],
        'probabilities': {},
    }
    
    # Run YOUR existing workflow
    for msg, meta in compiled.stream(initial_state, stream_mode='messages'):
        messages.append(str(msg.content))
    
    # Save results
    StorageService.save_results(analysis_id, results_data)
```

### Methods

#### `create_analysis(db: Session, dataset_id: str, selected_columns: list[str] | None, configuration: dict | None) -> Analysis`

Create analysis record in database.

**Parameters**:
- `db`: Database session
- `dataset_id`: UUID of dataset to analyze
- `selected_columns`: Specific columns (None = all columns)
- `configuration`: Additional config parameters

**Returns**:
- `Analysis`: Created analysis record with status="pending"

**Usage**:
```python
analysis = AnalysisService.create_analysis(
    db,
    dataset_id="550e8400...",
    selected_columns=["age", "treatment"],
    configuration={"alpha": 0.05}
)
```

---

#### `run_analysis(db: Session, analysis_id: str) -> None`

Execute statistical analysis workflow.

**Workflow**:
1. Update status to "running"
2. Load dataset as DataFrame
3. Filter columns if specified
4. Prepare initial state for LangGraph
5. Execute `compiled.stream()` (YOUR workflow)
6. Capture all messages
7. Extract results and probabilities
8. Save results to JSON
9. Save log to text file
10. Update analysis record with paths and summary
11. Update status to "completed" or "failed"

**Parameters**:
- `db`: Database session
- `analysis_id`: UUID of analysis

**Returns**:
- `None` (updates database record)

**Usage**:
```python
# In background thread or task
AnalysisService.run_analysis(db, "660e8400...")
```

**Error Handling**:
- Catches all exceptions
- Updates status to "failed"
- Stores error message in database
- Logs full traceback

**State Preparation**:
```python
initial_state = {
    'df': df,  # Your DataFrame
    'secondary_df': None,  # For paired tests
    'target_columns': selected_columns or [],
    'paired': False,  # Will be determined by workflow
    'data_type': None,  # Will be inferred
    'do_association': False,
    'number_of_samples': len(df),
    'results': [],
    'probabilities': {},
}
```

**Result Extraction**:
```python
# After workflow completes
final_state = compiled.get_state()
results = final_state['results']
probabilities = final_state['probabilities']
```

---

#### `get_analysis(db: Session, analysis_id: str) -> Analysis | None`

Retrieve analysis by ID.

**Parameters**:
- `db`: Database session
- `analysis_id`: UUID string

**Returns**:
- `Analysis` if found, `None` otherwise

**Usage**:
```python
analysis = AnalysisService.get_analysis(db, "660e8400...")
```

---

#### `get_analysis_results(db: Session, analysis_id: str) -> dict`

Get full analysis results with metadata.

**Parameters**:
- `db`: Database session
- `analysis_id`: UUID string

**Returns**:
- `dict`: Complete results package

**Response Structure**:
```python
{
    "id": "660e8400...",
    "dataset_id": "550e8400...",
    "status": "completed",
    "dataset_name": "mydata.csv",
    "start_time": datetime,
    "end_time": datetime,
    "duration_seconds": 45.2,
    "summary": "AI-generated summary...",
    "probabilities": {"test_name": 0.003, ...},
    "results_detail": {...},  # Full results from JSON
    "log_available": True
}
```

**Usage**:
```python
results = AnalysisService.get_analysis_results(db, "660e8400...")
print(results["summary"])
print(results["probabilities"])
```

---

#### `get_analysis_log(db: Session, analysis_id: str) -> dict`

Get execution log content.

**Parameters**:
- `db`: Database session
- `analysis_id`: UUID string

**Returns**:
- `dict`: Log data

**Response Structure**:
```python
{
    "analysis_id": "660e8400...",
    "log_content": "Full log text..."
}
```

**Usage**:
```python
log = AnalysisService.get_analysis_log(db, "660e8400...")
print(log["log_content"])
```

---

## TaskService

**File**: `statmate/api/services/task_service.py`

### Purpose

Manages scheduled analysis tasks:
- Create one-time or recurring tasks
- Register with APScheduler
- Track execution history
- Pause/resume/delete tasks

### Methods

#### `create_task(db: Session, name: str, task_type: str, dataset_id: str, schedule: str, selected_columns: list[str] | None, configuration: dict | None) -> ScheduledTask`

Create and schedule a new task.

**Workflow**:
1. Validate dataset exists
2. Parse schedule (cron or ISO datetime)
3. Create database record
4. Register with APScheduler
5. Calculate next run time

**Parameters**:
- `db`: Database session
- `name`: Human-readable task name
- `task_type`: "one_time" or "recurring"
- `dataset_id`: Dataset to analyze
- `schedule`: Cron expression or ISO datetime
- `selected_columns`: Columns to analyze (optional)
- `configuration`: Analysis config (optional)

**Returns**:
- `ScheduledTask`: Created task record

**Usage**:
```python
# One-time task
task = TaskService.create_task(
    db,
    name="Weekend Analysis",
    task_type="one_time",
    dataset_id="550e8400...",
    schedule="2025-10-20T14:30:00",
    selected_columns=None,
    configuration=None
)

# Recurring task
task = TaskService.create_task(
    db,
    name="Daily Report",
    task_type="recurring",
    dataset_id="550e8400...",
    schedule="0 2 * * *",  # Daily at 2 AM
    selected_columns=None,
    configuration=None
)
```

**Schedule Formats**:
- **One-time**: ISO 8601 datetime string
- **Recurring**: Cron expression (minute hour day month weekday)

**APScheduler Registration**:
```python
# One-time
scheduler.add_job(
    execute_scheduled_analysis,
    trigger='date',
    run_date=datetime.fromisoformat(schedule),
    args=[task_id]
)

# Recurring
scheduler.add_job(
    execute_scheduled_analysis,
    trigger='cron',
    **parse_cron(schedule),
    args=[task_id]
)
```

---

#### `get_task(db: Session, task_id: str) -> ScheduledTask | None`

Retrieve task by ID.

**Parameters**:
- `db`: Database session
- `task_id`: UUID string

**Returns**:
- `ScheduledTask` if found, `None` otherwise

**Usage**:
```python
task = TaskService.get_task(db, "770e8400...")
```

---

#### `list_tasks(db: Session, skip: int, limit: int) -> list[ScheduledTask]`

List tasks with pagination.

**Parameters**:
- `db`: Database session
- `skip`: Records to skip
- `limit`: Max records

**Returns**:
- `list[ScheduledTask]`: List of tasks

**Usage**:
```python
tasks = TaskService.list_tasks(db, skip=0, limit=10)
```

---

#### `pause_task(db: Session, task_id: str) -> bool`

Pause a task temporarily.

**Workflow**:
1. Get task from database
2. Update status to "paused"
3. Remove from APScheduler
4. Set next_run to None

**Parameters**:
- `db`: Database session
- `task_id`: UUID string

**Returns**:
- `bool`: True if paused, False if not found

**Usage**:
```python
success = TaskService.pause_task(db, "770e8400...")
```

**Note**: Paused tasks can be resumed later.

---

#### `resume_task(db: Session, task_id: str) -> bool`

Resume a paused task.

**Workflow**:
1. Get task from database
2. Verify status is "paused"
3. Update status to "active"
4. Re-register with APScheduler
5. Calculate next run time

**Parameters**:
- `db`: Database session
- `task_id`: UUID string

**Returns**:
- `bool`: True if resumed, False if not found

**Usage**:
```python
success = TaskService.resume_task(db, "770e8400...")
```

---

#### `delete_task(db: Session, task_id: str) -> bool`

Delete task permanently.

**Workflow**:
1. Get task from database
2. Remove from APScheduler
3. Delete database record

**Parameters**:
- `db`: Database session
- `task_id`: UUID string

**Returns**:
- `bool`: True if deleted, False if not found

**Usage**:
```python
success = TaskService.delete_task(db, "770e8400...")
```

**Note**: Does NOT delete associated analyses.

---

#### `update_task_execution(db: Session, task_id: str, analysis_id: str) -> None`

Update task after execution.

**Called by**: `execute_scheduled_analysis` job function

**Workflow**:
1. Increment run_count
2. Update last_run timestamp
3. Calculate next_run (for recurring)
4. Link analysis to task

**Parameters**:
- `db`: Database session
- `task_id`: UUID string
- `analysis_id`: UUID of created analysis

**Returns**:
- `None` (updates database)

**Usage** (internal):
```python
def execute_scheduled_analysis(task_id: str):
    # Create analysis
    analysis = AnalysisService.create_analysis(...)
    
    # Run analysis
    AnalysisService.run_analysis(db, analysis.id)
    
    # Update task
    TaskService.update_task_execution(db, task_id, analysis.id)
```

---

## Service Best Practices

### 1. **Always Use Database Sessions**

```python
# ❌ Bad: Creating session inside service
def get_dataset(dataset_id: str):
    db = SessionLocal()
    result = db.query(Dataset).filter_by(id=dataset_id).first()
    db.close()
    return result

# ✅ Good: Accept session as parameter
def get_dataset(db: Session, dataset_id: str):
    return db.query(Dataset).filter_by(id=dataset_id).first()
```

**Why?** Allows caller to manage transactions and rollback on errors.

---

### 2. **Raise Descriptive Errors**

```python
# ❌ Bad: Generic error
if not dataset:
    raise Exception("Error")

# ✅ Good: Specific error with context
if not dataset:
    raise ValueError(f"Dataset not found: {dataset_id}")
```

---

### 3. **Clean Up Resources**

```python
# ✅ Always clean up files on failure
try:
    # Save file
    filepath = save_upload(file)
    # Process file
    df = read_dataset(filepath)
    # Create record
    dataset = create_dataset(db, df)
except Exception as e:
    # Clean up file before re-raising
    if filepath and os.path.exists(filepath):
        os.remove(filepath)
    raise
```

---

### 4. **Log Important Events**

```python
import logging

logger = logging.getLogger(__name__)

def run_analysis(db: Session, analysis_id: str):
    logger.info(f"Starting analysis: {analysis_id}")
    try:
        # ... execute workflow ...
        logger.info(f"Analysis completed: {analysis_id}")
    except Exception as e:
        logger.error(f"Analysis failed: {analysis_id}", exc_info=True)
        raise
```

---

### 5. **Return Domain Models, Not Dictionaries**

```python
# ❌ Bad: Returning dict
def get_dataset(db: Session, dataset_id: str) -> dict:
    dataset = db.query(Dataset).filter_by(id=dataset_id).first()
    return {
        "id": dataset.id,
        "name": dataset.filename,
        # ... manual serialization
    }

# ✅ Good: Return model (Pydantic handles serialization)
def get_dataset(db: Session, dataset_id: str) -> Dataset | None:
    return db.query(Dataset).filter_by(id=dataset_id).first()
```

---

## Testing Services

Services should be tested independently of HTTP layer:

```python
# tests/unit/services/test_dataset_service.py
import pytest
from statmate.api.services.dataset_service import DatasetService
from database import get_test_db

def test_create_dataset():
    db = get_test_db()
    
    # Create mock file
    mock_file = create_mock_upload("test.csv", "col1,col2\n1,2\n3,4")
    
    # Test service method
    dataset = DatasetService.create_dataset(db, mock_file, "Test dataset")
    
    # Assertions
    assert dataset.id is not None
    assert dataset.row_count == 2
    assert "col1" in dataset.column_names
    
    # Cleanup
    db.delete(dataset)
    db.commit()
```

---

## Service Dependencies

```
┌─────────────────────┐
│   TaskService       │
│  (Highest Level)    │
└──────────┬──────────┘
           │ Uses
           ▼
┌─────────────────────┐
│  AnalysisService    │
└──────────┬──────────┘
           │ Uses
           ▼
┌─────────────────────┐
│  DatasetService     │
└──────────┬──────────┘
           │ Uses
           ▼
┌─────────────────────┐
│  StorageService     │
│  (Lowest Level)     │
└─────────────────────┘
```

**Dependency Rule**: Higher-level services can use lower-level services, but NOT vice versa.

---

## Future Enhancements

### Planned Service Additions

1. **AuthService**: User authentication and authorization
2. **NotificationService**: Email/webhook notifications
3. **ExportService**: Export results to various formats (PDF, R, SPSS)
4. **CacheService**: Redis-based result caching
5. **BatchService**: Batch analysis processing
6. **ValidationService**: Data quality validation
7. **AuditService**: Audit log tracking

---

**For implementation details**, see individual service files with comprehensive docstrings.

