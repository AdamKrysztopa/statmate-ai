"""Database package for StatmateAI.

This package contains database models, session management, and initialization
using SQLAlchemy ORM with SQLite (development) or PostgreSQL (production).

Modules:
    models: SQLAlchemy ORM models (Dataset, Analysis, ScheduledTask)
    session: Database session management and connection handling

Usage:
    from database import Dataset, Analysis, ScheduledTask, get_db, init_db

    # Initialize database (creates all tables)
    init_db()

    # Use in FastAPI dependency injection
    @app.get("/datasets")
    def list_datasets(db: Session = Depends(get_db)):
        return db.query(Dataset).all()

Database Schema:
    Dataset:
        - Stores metadata about uploaded CSV/Excel files
        - One-to-Many relationship with Analysis and ScheduledTask

    Analysis:
        - Tracks statistical analysis runs (status, results, logs)
        - Many-to-One with Dataset, optional link to ScheduledTask

    ScheduledTask:
        - Manages recurring and one-time analysis jobs
        - Many-to-One with Dataset, One-to-Many with Analysis

Storage:
    - SQLite: Default for development (single file database)
    - PostgreSQL: Production (set DATABASE_URL in .env)

    Connection string examples:
    - SQLite: sqlite:///./database/statmate.db
    - PostgreSQL: postgresql://user:pass@localhost:5432/statmate

Thread Safety:
    - Sessions are scoped to requests via FastAPI dependency injection
    - SessionLocal creates new sessions per request
    - Always close sessions after use (handled automatically by FastAPI)
"""

from database.models import Analysis, Base, Dataset, ScheduledTask
from database.session import SessionLocal, engine, get_db

__all__ = [
    'Base',
    'Dataset',
    'Analysis',
    'ScheduledTask',
    'engine',
    'SessionLocal',
    'get_db',
]
