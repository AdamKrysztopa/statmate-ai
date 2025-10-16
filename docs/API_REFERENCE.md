# StatmateAI API Reference

**Version**: 0.1.0  
**Base URL**: `http://localhost:8000/api/v1`  
**Interactive Docs**: http://localhost:8000/docs  
**OpenAPI Schema**: http://localhost:8000/openapi.json

---

## Authentication

**Current Version**: No authentication required (development)  
**Planned**: JWT-based authentication in Phase 2

---

## Common Headers

All requests should include:

```
Content-Type: application/json
Accept: application/json
```

For file uploads:
```
Content-Type: multipart/form-data
```

---

## Response Format

### Success Response

```json
{
  "id": "uuid",
  "field": "value",
  ...
}
```

### Error Response

```json
{
  "detail": "Error message description"
}
```

### HTTP Status Codes

- `200 OK` - Request succeeded
- `201 Created` - Resource created successfully
- `400 Bad Request` - Invalid request parameters
- `404 Not Found` - Resource not found
- `500 Internal Server Error` - Server error

---

## Datasets API

### Upload Dataset

Upload a CSV or Excel file to create a new dataset.

**Endpoint**: `POST /datasets/upload`

**Request**:
```bash
curl -X POST http://localhost:8000/api/v1/datasets/upload \
  -F "file=@mydata.csv" \
  -F "description=Optional dataset description"
```

**Parameters**:
| Parameter   | Type   | Required | Description                           |
| ----------- | ------ | -------- | ------------------------------------- |
| file        | file   | Yes      | CSV or Excel file (.csv, .xlsx, .xls) |
| description | string | No       | Human-readable description            |

**Response**: `201 Created`
```json
{
  "dataset_id": "550e8400-e29b-41d4-a716-446655440000",
  "message": "Dataset uploaded successfully",
  "dataset": {
    "id": "550e8400-e29b-41d4-a716-446655440000",
    "filename": "20251016_080000_mydata.parquet",
    "original_filename": "mydata.csv",
    "upload_timestamp": "2025-10-16T08:00:00.123456",
    "file_size": 1024,
    "row_count": 100,
    "column_names": ["age", "treatment", "outcome"],
    "data_types": {
      "age": "int64",
      "treatment": "object",
      "outcome": "float64"
    },
    "description": "Optional dataset description"
  }
}
```

**Errors**:
- `400` - Invalid file format
- `413` - File too large (max 100MB)
- `500` - File processing error

---

### List Datasets

Retrieve a paginated list of all uploaded datasets.

**Endpoint**: `GET /datasets/`

**Request**:
```bash
curl http://localhost:8000/api/v1/datasets/?skip=0&limit=10
```

**Query Parameters**:
| Parameter | Type    | Default | Description                            |
| --------- | ------- | ------- | -------------------------------------- |
| skip      | integer | 0       | Number of records to skip (pagination) |
| limit     | integer | 100     | Maximum number of records to return    |

**Response**: `200 OK`
```json
[
  {
    "id": "550e8400-e29b-41d4-a716-446655440000",
    "filename": "20251016_080000_mydata.parquet",
    "original_filename": "mydata.csv",
    "upload_timestamp": "2025-10-16T08:00:00.123456",
    "file_size": 1024,
    "row_count": 100,
    "column_names": ["age", "treatment", "outcome"],
    "data_types": {
      "age": "int64",
      "treatment": "object",
      "outcome": "float64"
    },
    "description": "Optional dataset description"
  }
]
```

---

### Get Dataset Details

Retrieve metadata for a specific dataset.

**Endpoint**: `GET /datasets/{dataset_id}`

**Request**:
```bash
curl http://localhost:8000/api/v1/datasets/550e8400-e29b-41d4-a716-446655440000
```

**Path Parameters**:
| Parameter  | Type          | Required | Description               |
| ---------- | ------------- | -------- | ------------------------- |
| dataset_id | string (UUID) | Yes      | Unique dataset identifier |

**Response**: `200 OK`
```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "filename": "20251016_080000_mydata.parquet",
  "original_filename": "mydata.csv",
  "upload_timestamp": "2025-10-16T08:00:00.123456",
  "file_size": 1024,
  "row_count": 100,
  "column_names": ["age", "treatment", "outcome"],
  "data_types": {
    "age": "int64",
    "treatment": "object",
    "outcome": "float64"
  },
  "description": "Optional dataset description"
}
```

**Errors**:
- `404` - Dataset not found

---

### Preview Dataset

Get a preview of the dataset rows (first N rows).

**Endpoint**: `GET /datasets/{dataset_id}/preview`

**Request**:
```bash
curl http://localhost:8000/api/v1/datasets/550e8400-e29b-41d4-a716-446655440000/preview?num_rows=5
```

**Path Parameters**:
| Parameter  | Type          | Required | Description               |
| ---------- | ------------- | -------- | ------------------------- |
| dataset_id | string (UUID) | Yes      | Unique dataset identifier |

**Query Parameters**:
| Parameter | Type    | Default | Description               |
| --------- | ------- | ------- | ------------------------- |
| num_rows  | integer | 10      | Number of rows to preview |

**Response**: `200 OK`
```json
{
  "dataset_id": "550e8400-e29b-41d4-a716-446655440000",
  "original_filename": "mydata.csv",
  "row_count": 100,
  "column_names": ["age", "treatment", "outcome"],
  "data_types": {
    "age": "int64",
    "treatment": "object",
    "outcome": "float64"
  },
  "preview_data": [
    {"age": 25, "treatment": "A", "outcome": 85.3},
    {"age": 30, "treatment": "B", "outcome": 92.1},
    {"age": 28, "treatment": "A", "outcome": 78.5}
  ],
  "preview_rows": 3
}
```

**Errors**:
- `404` - Dataset not found
- `500` - Error reading dataset

---

### Delete Dataset

Delete a dataset and all associated analyses.

**Endpoint**: `DELETE /datasets/{dataset_id}`

**Request**:
```bash
curl -X DELETE http://localhost:8000/api/v1/datasets/550e8400-e29b-41d4-a716-446655440000
```

**Path Parameters**:
| Parameter  | Type          | Required | Description               |
| ---------- | ------------- | -------- | ------------------------- |
| dataset_id | string (UUID) | Yes      | Unique dataset identifier |

**Response**: `200 OK`
```json
{
  "message": "Dataset deleted successfully",
  "dataset_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

**Errors**:
- `404` - Dataset not found
- `500` - Error deleting files

---

## Analysis API

### Run Analysis

Execute statistical analysis on a dataset.

**Endpoint**: `POST /analysis/run`

**Request**:
```bash
curl -X POST http://localhost:8000/api/v1/analysis/run \
  -H "Content-Type: application/json" \
  -d '{
    "dataset_id": "550e8400-e29b-41d4-a716-446655440000",
    "selected_columns": ["age", "treatment", "outcome"],
    "configuration": {
      "alpha": 0.05,
      "paired": false
    }
  }'
```

**Request Body**:
```json
{
  "dataset_id": "string (UUID)",
  "selected_columns": ["string"] | null,
  "configuration": {
    "key": "value"
  } | null
}
```

**Parameters**:
| Parameter        | Type          | Required | Description                         |
| ---------------- | ------------- | -------- | ----------------------------------- |
| dataset_id       | string (UUID) | Yes      | Dataset to analyze                  |
| selected_columns | array[string] | No       | Specific columns (null = all)       |
| configuration    | object        | No       | Additional configuration parameters |

**Response**: `201 Created`
```json
{
  "id": "660e8400-e29b-41d4-a716-446655440001",
  "dataset_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "pending",
  "selected_columns": ["age", "treatment", "outcome"],
  "configuration": {
    "alpha": 0.05,
    "paired": false
  },
  "start_time": null,
  "end_time": null,
  "result_path": null,
  "log_path": null,
  "error_message": null,
  "summary": null,
  "probabilities": null
}
```

**Status Values**:
- `pending` - Queued for execution
- `running` - Currently executing
- `completed` - Finished successfully
- `failed` - Execution failed (see error_message)

**Errors**:
- `400` - Invalid dataset_id or parameters
- `404` - Dataset not found
- `500` - Analysis execution error

---

### Get Analysis Status

Check the status of a running or completed analysis.

**Endpoint**: `GET /analysis/{analysis_id}`

**Request**:
```bash
curl http://localhost:8000/api/v1/analysis/660e8400-e29b-41d4-a716-446655440001
```

**Path Parameters**:
| Parameter   | Type          | Required | Description                |
| ----------- | ------------- | -------- | -------------------------- |
| analysis_id | string (UUID) | Yes      | Unique analysis identifier |

**Response**: `200 OK`
```json
{
  "id": "660e8400-e29b-41d4-a716-446655440001",
  "status": "completed",
  "progress": 100.0,
  "message": "Analysis completed successfully"
}
```

**Progress Field**:
- `0-100` - Percentage complete
- Only available for `running` status

**Errors**:
- `404` - Analysis not found

---

### Get Analysis Results

Retrieve full results of a completed analysis.

**Endpoint**: `GET /analysis/{analysis_id}/results`

**Request**:
```bash
curl http://localhost:8000/api/v1/analysis/660e8400-e29b-41d4-a716-446655440001/results
```

**Path Parameters**:
| Parameter   | Type          | Required | Description                |
| ----------- | ------------- | -------- | -------------------------- |
| analysis_id | string (UUID) | Yes      | Unique analysis identifier |

**Response**: `200 OK`
```json
{
  "id": "660e8400-e29b-41d4-a716-446655440001",
  "dataset_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "completed",
  "dataset_name": "mydata.csv",
  "start_time": "2025-10-16T08:05:00.123456",
  "end_time": "2025-10-16T08:05:45.678901",
  "duration_seconds": 45.56,
  "summary": "The analysis revealed statistically significant differences between groups (p < 0.001). Paired t-test showed...",
  "probabilities": {
    "shapiro_wilk_test": 0.234,
    "paired_t_test": 0.0003,
    "effect_size": 0.85
  },
  "results_detail": {
    "tests_performed": [
      {
        "test_name": "Shapiro-Wilk Test",
        "test_type": "normality",
        "statistic": 0.98,
        "p_value": 0.234,
        "interpretation": "Data appears normally distributed"
      },
      {
        "test_name": "Paired t-test",
        "test_type": "comparison",
        "statistic": 3.87,
        "p_value": 0.0003,
        "degrees_of_freedom": 99,
        "confidence_interval": [2.1, 5.3],
        "interpretation": "Statistically significant difference (p < 0.001)"
      }
    ],
    "recommendations": [
      "Results show strong evidence for the alternative hypothesis",
      "Consider reporting effect size (Cohen's d = 0.85)"
    ]
  },
  "log_available": true
}
```

**Errors**:
- `404` - Analysis not found
- `400` - Analysis not completed yet

---

### Get Analysis Log

Retrieve the execution log for an analysis.

**Endpoint**: `GET /analysis/{analysis_id}/log`

**Request**:
```bash
curl http://localhost:8000/api/v1/analysis/660e8400-e29b-41d4-a716-446655440001/log
```

**Path Parameters**:
| Parameter   | Type          | Required | Description                |
| ----------- | ------------- | -------- | -------------------------- |
| analysis_id | string (UUID) | Yes      | Unique analysis identifier |

**Response**: `200 OK`
```json
{
  "analysis_id": "660e8400-e29b-41d4-a716-446655440001",
  "log_content": "2025-10-16 08:05:00 - INFO - Starting analysis...\n2025-10-16 08:05:01 - INFO - Loading dataset...\n2025-10-16 08:05:05 - INFO - Running normality tests...\n2025-10-16 08:05:20 - INFO - Running paired t-test...\n2025-10-16 08:05:45 - INFO - Analysis completed successfully"
}
```

**Errors**:
- `404` - Analysis or log not found

---

### List Analyses

Retrieve all analyses for a dataset or all datasets.

**Endpoint**: `GET /analysis/`

**Request**:
```bash
curl http://localhost:8000/api/v1/analysis/?dataset_id=550e8400-e29b-41d4-a716-446655440000&skip=0&limit=10
```

**Query Parameters**:
| Parameter  | Type          | Default | Description                  |
| ---------- | ------------- | ------- | ---------------------------- |
| dataset_id | string (UUID) | null    | Filter by dataset (optional) |
| skip       | integer       | 0       | Pagination offset            |
| limit      | integer       | 100     | Maximum records to return    |

**Response**: `200 OK`
```json
[
  {
    "id": "660e8400-e29b-41d4-a716-446655440001",
    "dataset_id": "550e8400-e29b-41d4-a716-446655440000",
    "status": "completed",
    "start_time": "2025-10-16T08:05:00.123456",
    "end_time": "2025-10-16T08:05:45.678901",
    "summary": "Analysis summary..."
  }
]
```

---

## Scheduled Tasks API

### Schedule Task

Create a one-time or recurring analysis task.

**Endpoint**: `POST /tasks/schedule`

**Request**:
```bash
curl -X POST http://localhost:8000/api/v1/tasks/schedule \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Daily Analysis",
    "task_type": "recurring",
    "dataset_id": "550e8400-e29b-41d4-a716-446655440000",
    "selected_columns": null,
    "configuration": null,
    "schedule": "0 2 * * *"
  }'
```

**Request Body**:
```json
{
  "name": "string",
  "task_type": "one_time" | "recurring",
  "dataset_id": "string (UUID)",
  "selected_columns": ["string"] | null,
  "configuration": object | null,
  "schedule": "string"
}
```

**Parameters**:
| Parameter        | Type          | Required | Description                     |
| ---------------- | ------------- | -------- | ------------------------------- |
| name             | string        | Yes      | Human-readable task name        |
| task_type        | enum          | Yes      | `one_time` or `recurring`       |
| dataset_id       | string (UUID) | Yes      | Dataset to analyze              |
| selected_columns | array[string] | No       | Columns to analyze              |
| configuration    | object        | No       | Analysis configuration          |
| schedule         | string        | Yes      | Cron expression or ISO datetime |

**Schedule Format**:
- **One-time**: ISO 8601 datetime (e.g., `"2025-10-20T14:30:00"`)
- **Recurring**: Cron expression (e.g., `"0 2 * * *"` = daily at 2 AM)

**Cron Examples**:
```
0 2 * * *       # Daily at 2:00 AM
0 */6 * * *     # Every 6 hours
0 9 * * 1-5     # Weekdays at 9:00 AM
0 0 1 * *       # First day of month at midnight
```

**Response**: `201 Created`
```json
{
  "id": "770e8400-e29b-41d4-a716-446655440002",
  "name": "Daily Analysis",
  "task_type": "recurring",
  "dataset_id": "550e8400-e29b-41d4-a716-446655440000",
  "selected_columns": null,
  "configuration": null,
  "schedule": "0 2 * * *",
  "status": "active",
  "next_run": "2025-10-17T02:00:00",
  "last_run": null,
  "run_count": 0,
  "created_at": "2025-10-16T08:10:00.123456",
  "updated_at": "2025-10-16T08:10:00.123456"
}
```

**Errors**:
- `400` - Invalid schedule format or parameters
- `404` - Dataset not found

---

### List Tasks

Retrieve all scheduled tasks.

**Endpoint**: `GET /tasks/`

**Request**:
```bash
curl http://localhost:8000/api/v1/tasks/?skip=0&limit=10
```

**Query Parameters**:
| Parameter | Type    | Default | Description               |
| --------- | ------- | ------- | ------------------------- |
| skip      | integer | 0       | Pagination offset         |
| limit     | integer | 100     | Maximum records to return |

**Response**: `200 OK`
```json
[
  {
    "id": "770e8400-e29b-41d4-a716-446655440002",
    "name": "Daily Analysis",
    "task_type": "recurring",
    "status": "active",
    "next_run": "2025-10-17T02:00:00",
    "last_run": null,
    "run_count": 0
  }
]
```

---

### Get Task Details

Retrieve details of a specific task.

**Endpoint**: `GET /tasks/{task_id}`

**Request**:
```bash
curl http://localhost:8000/api/v1/tasks/770e8400-e29b-41d4-a716-446655440002
```

**Response**: `200 OK`
```json
{
  "id": "770e8400-e29b-41d4-a716-446655440002",
  "name": "Daily Analysis",
  "task_type": "recurring",
  "dataset_id": "550e8400-e29b-41d4-a716-446655440000",
  "selected_columns": null,
  "configuration": null,
  "schedule": "0 2 * * *",
  "status": "active",
  "next_run": "2025-10-17T02:00:00",
  "last_run": null,
  "run_count": 0,
  "created_at": "2025-10-16T08:10:00.123456",
  "updated_at": "2025-10-16T08:10:00.123456"
}
```

**Errors**:
- `404` - Task not found

---

### Pause Task

Temporarily pause a scheduled task.

**Endpoint**: `PUT /tasks/{task_id}/pause`

**Request**:
```bash
curl -X PUT http://localhost:8000/api/v1/tasks/770e8400-e29b-41d4-a716-446655440002/pause
```

**Response**: `200 OK`
```json
{
  "message": "Task paused successfully",
  "task_id": "770e8400-e29b-41d4-a716-446655440002",
  "status": "paused"
}
```

---

### Resume Task

Resume a paused task.

**Endpoint**: `PUT /tasks/{task_id}/resume`

**Request**:
```bash
curl -X PUT http://localhost:8000/api/v1/tasks/770e8400-e29b-41d4-a716-446655440002/resume
```

**Response**: `200 OK`
```json
{
  "message": "Task resumed successfully",
  "task_id": "770e8400-e29b-41d4-a716-446655440002",
  "status": "active",
  "next_run": "2025-10-17T02:00:00"
}
```

---

### Delete Task

Delete a scheduled task.

**Endpoint**: `DELETE /tasks/{task_id}`

**Request**:
```bash
curl -X DELETE http://localhost:8000/api/v1/tasks/770e8400-e29b-41d4-a716-446655440002
```

**Response**: `200 OK`
```json
{
  "message": "Task deleted successfully",
  "task_id": "770e8400-e29b-41d4-a716-446655440002"
}
```

---

## Results API

### List All Results

Retrieve a list of all completed analyses with results.

**Endpoint**: `GET /results/`

**Request**:
```bash
curl http://localhost:8000/api/v1/results/?skip=0&limit=10
```

**Query Parameters**:
| Parameter | Type    | Default | Description               |
| --------- | ------- | ------- | ------------------------- |
| skip      | integer | 0       | Pagination offset         |
| limit     | integer | 100     | Maximum records to return |

**Response**: `200 OK`
```json
[
  {
    "id": "660e8400-e29b-41d4-a716-446655440001",
    "dataset_id": "550e8400-e29b-41d4-a716-446655440000",
    "dataset_name": "mydata.csv",
    "status": "completed",
    "start_time": "2025-10-16T08:05:00.123456",
    "end_time": "2025-10-16T08:05:45.678901",
    "summary": "Brief summary..."
  }
]
```

---

### Get Result Details

Get full details of a specific analysis result.

**Endpoint**: `GET /results/{result_id}`

**Request**:
```bash
curl http://localhost:8000/api/v1/results/660e8400-e29b-41d4-a716-446655440001
```

**Response**: Same as `GET /analysis/{analysis_id}/results`

---

## Health Check

Check API server health status.

**Endpoint**: `GET /health`

**Request**:
```bash
curl http://localhost:8000/health
```

**Response**: `200 OK`
```json
{
  "status": "healthy",
  "version": "0.1.0",
  "timestamp": "2025-10-16T08:00:00.123456"
}
```

---

## Rate Limiting

**Current Version**: No rate limiting  
**Planned**: 100 requests/minute per IP in production

---

## Pagination

All list endpoints support pagination via `skip` and `limit` parameters:

```
GET /api/v1/datasets/?skip=20&limit=10
```

- `skip`: Number of records to skip (default: 0)
- `limit`: Maximum records to return (default: 100, max: 1000)

**Response Headers** (planned for Phase 2):
```
X-Total-Count: 150
X-Page-Number: 3
X-Page-Size: 10
```

---

## Webhooks (Planned)

Future versions will support webhooks for:
- Analysis completion
- Task execution
- Error notifications

---

## SDK Examples

### Python

```python
import httpx

API_BASE = "http://localhost:8000/api/v1"

# Upload dataset
with open("data.csv", "rb") as f:
    response = httpx.post(
        f"{API_BASE}/datasets/upload",
        files={"file": f}
    )
dataset = response.json()

# Run analysis
response = httpx.post(
    f"{API_BASE}/analysis/run",
    json={"dataset_id": dataset["dataset_id"]}
)
analysis = response.json()

# Get results
response = httpx.get(
    f"{API_BASE}/analysis/{analysis['id']}/results"
)
results = response.json()
print(results["summary"])
```

### JavaScript/TypeScript

```javascript
const API_BASE = "http://localhost:8000/api/v1";

// Upload dataset
const formData = new FormData();
formData.append("file", fileInput.files[0]);

const uploadResponse = await fetch(`${API_BASE}/datasets/upload`, {
  method: "POST",
  body: formData,
});
const dataset = await uploadResponse.json();

// Run analysis
const analysisResponse = await fetch(`${API_BASE}/analysis/run`, {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ dataset_id: dataset.dataset_id }),
});
const analysis = await analysisResponse.json();

// Get results
const resultsResponse = await fetch(
  `${API_BASE}/analysis/${analysis.id}/results`
);
const results = await resultsResponse.json();
console.log(results.summary);
```

---

## Error Codes Reference

| Code | Meaning               | Resolution                       |
| ---- | --------------------- | -------------------------------- |
| 400  | Bad Request           | Check request parameters         |
| 404  | Not Found             | Verify resource ID exists        |
| 413  | Payload Too Large     | Reduce file size (max 100MB)     |
| 422  | Validation Error      | Fix request body schema          |
| 500  | Internal Server Error | Check logs, report if persistent |
| 503  | Service Unavailable   | Server overloaded, retry later   |

---

## Changelog

### Version 0.1.0 (2025-10-16)

- Initial API release
- 18 endpoints across 4 resource types
- Dataset upload and management
- Analysis execution and results
- Task scheduling (one-time and recurring)
- Health check endpoint

---

**For more information**, see:
- [Implementation Guide](IMPLEMENTATION_GUIDE.md)
- [Quick Start Guide](QUICK_START.md)
- [Architecture Proposal](ARCHITECTURE_PROPOSAL.md)

**Interactive Documentation**: http://localhost:8000/docs

