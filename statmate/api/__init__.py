"""FastAPI backend for StatmateAI.

This package contains the complete REST API implementation including:
- API routes (endpoints)
- Service layer (business logic)
- Pydantic models (request/response DTOs)
- Task scheduling (APScheduler)
- Database integration (SQLAlchemy)

Architecture:

    Browser/Client
         ↓ HTTP
    ┌────────────────────┐
    │   API Routes       │  routes/*.py
    │   (FastAPI)        │
    └────────┬───────────┘
             ↓
    ┌────────────────────┐
    │  Service Layer     │  services/*.py
    │  (Business Logic)  │
    └────────┬───────────┘
             ↓
    ┌────────────────────┐
    │  Database Models   │  database/models.py
    │  (SQLAlchemy ORM)  │
    └────────────────────┘

Packages:
    models/: Pydantic models for request/response validation
    routes/: FastAPI route handlers (endpoints)
    services/: Business logic layer
    scheduler/: Background task scheduling (APScheduler)

Main Entry Point:
    main.py - FastAPI application with lifecycle management

Usage:
    # Run development server
    python statmate/api/main.py

    # Or with uvicorn
    uvicorn statmate.api.main:app --reload --port 8000

    # Access interactive API docs
    http://localhost:8000/docs

API Endpoints (18 total):

    Datasets:
        POST   /api/v1/datasets/upload     - Upload CSV/Excel
        GET    /api/v1/datasets/           - List all datasets
        GET    /api/v1/datasets/{id}       - Get dataset details
        GET    /api/v1/datasets/{id}/preview - Preview data
        DELETE /api/v1/datasets/{id}       - Delete dataset

    Analysis:
        POST   /api/v1/analysis/run        - Run analysis
        GET    /api/v1/analysis/{id}       - Get status
        GET    /api/v1/analysis/{id}/results - Get results
        GET    /api/v1/analysis/{id}/log   - Get execution log
        GET    /api/v1/analysis/           - List analyses

    Tasks:
        POST   /api/v1/tasks/schedule      - Schedule task
        GET    /api/v1/tasks/              - List tasks
        GET    /api/v1/tasks/{id}          - Get task details
        PUT    /api/v1/tasks/{id}/pause    - Pause task
        PUT    /api/v1/tasks/{id}/resume   - Resume task
        DELETE /api/v1/tasks/{id}          - Delete task

    Results:
        GET    /api/v1/results/            - List all results
        GET    /api/v1/results/{id}        - Get result details

Integration with Core:
    The API wraps the existing StatMate workflow (LangGraph) without
    modifying any core logic. See services/analysis_service.py for
    integration point.

Features:
    - Automatic request/response validation
    - Auto-generated OpenAPI documentation
    - CORS middleware for frontend access
    - Background task execution
    - File upload handling
    - Error handling and logging
    - Health check endpoint

Dependencies:
    - FastAPI: Web framework
    - Pydantic: Data validation
    - SQLAlchemy: ORM
    - APScheduler: Task scheduling
    - Pandas: Data handling
    - HTTPX: HTTP client (for testing)
"""

__version__ = '0.1.0'
