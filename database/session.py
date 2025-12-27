"""Database session management and connection handling.

This module provides:
- Database engine creation
- Session factory
- Dependency injection for FastAPI routes
"""

import logging
from collections.abc import Generator
from typing import Any

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import Session, sessionmaker

from config.settings import settings

logger = logging.getLogger(__name__)

# Create database engine
engine = create_engine(
    settings.DATABASE_URL,
    echo=settings.DATABASE_ECHO,
    pool_pre_ping=True,  # Verify connections before using
    connect_args={'check_same_thread': False} if 'sqlite' in settings.DATABASE_URL else {},
)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def _ensure_analysis_columns() -> None:
    """Apply lightweight, idempotent migrations for the analyses table."""
    inspector = inspect(engine)
    try:
        columns = {col['name'] for col in inspector.get_columns('analyses')}
    except Exception as exc:  # pragma: no cover - defensive for missing table edge cases
        logger.warning('Could not inspect analyses table: %s', exc)
        return

    statements: list[str] = []
    if 'version' not in columns:
        statements.append('ALTER TABLE analyses ADD COLUMN version INTEGER NOT NULL DEFAULT 1')
        statements.append('UPDATE analyses SET version = COALESCE(version, 1)')
    if 'superseded_at' not in columns:
        statements.append('ALTER TABLE analyses ADD COLUMN superseded_at TIMESTAMP NULL')
    if 'comment' not in columns:
        statements.append('ALTER TABLE analyses ADD COLUMN comment TEXT')
    if 'model_name' not in columns:
        statements.append('ALTER TABLE analyses ADD COLUMN model_name VARCHAR(100)')
    if 'provider' not in columns:
        statements.append('ALTER TABLE analyses ADD COLUMN provider VARCHAR(50)')
    if 'decision_steps' not in columns:
        statements.append('ALTER TABLE analyses ADD COLUMN decision_steps JSON')
    if 'intermediate_log' not in columns:
        statements.append('ALTER TABLE analyses ADD COLUMN intermediate_log TEXT')
    if 'assumption_log' not in columns:
        statements.append('ALTER TABLE analyses ADD COLUMN assumption_log JSON')
    if 'result_path' not in columns:
        statements.append('ALTER TABLE analyses ADD COLUMN result_path VARCHAR(500)')
    if 'log_path' not in columns:
        statements.append('ALTER TABLE analyses ADD COLUMN log_path VARCHAR(500)')
    if 'probabilities' not in columns:
        statements.append('ALTER TABLE analyses ADD COLUMN probabilities JSON')

    if not statements:
        return

    with engine.begin() as conn:
        for stmt in statements:
            conn.execute(text(stmt))
    logger.info('Ensured analyses table has required columns (applied %d migration steps)', len(statements))


def get_db() -> Generator[Session, Any, None]:
    """Dependency for FastAPI routes to get database session.

    Yields:
        Database session.

    Example:
        ```python
        @app.get("/datasets")
        async def list_datasets(db: Session = Depends(get_db)):
            return db.query(Dataset).all()
        ```
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """Initialize database by creating all tables and applying lightweight migrations."""
    from database.models import Base

    Base.metadata.create_all(bind=engine)
    _ensure_analysis_columns()


def drop_db() -> None:
    """Drop all database tables.

    WARNING: This will delete all data. Use only for testing/development.
    """
    from database.models import Base

    Base.metadata.drop_all(bind=engine)
