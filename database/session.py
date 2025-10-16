"""Database session management and connection handling.

This module provides:
- Database engine creation
- Session factory
- Dependency injection for FastAPI routes
"""

from collections.abc import Generator
from typing import Any

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from config.settings import settings

# Create database engine
engine = create_engine(
    settings.DATABASE_URL,
    echo=settings.DATABASE_ECHO,
    pool_pre_ping=True,  # Verify connections before using
    connect_args={'check_same_thread': False} if 'sqlite' in settings.DATABASE_URL else {},
)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


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
    """Initialize database by creating all tables.

    This function should be called on application startup.
    """
    from database.models import Base

    Base.metadata.create_all(bind=engine)


def drop_db() -> None:
    """Drop all database tables.

    WARNING: This will delete all data. Use only for testing/development.
    """
    from database.models import Base

    Base.metadata.drop_all(bind=engine)
