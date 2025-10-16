"""Shared dependencies for FastAPI routes."""

from collections.abc import Generator

from sqlalchemy.orm import Session

from database.session import get_db


def get_database_session() -> Generator[Session, None, None]:
    """Dependency to get database session.

    Yields:
        Database session
    """
    yield from get_db()
