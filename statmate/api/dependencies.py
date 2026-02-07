"""Shared dependencies for FastAPI routes."""

from collections.abc import Generator

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session

from config.settings import settings
from database.models import User
from database.session import get_db
from statmate.api.security import decode_access_token

oauth2_scheme = OAuth2PasswordBearer(tokenUrl='/api/v1/auth/token', auto_error=False)


def get_database_session() -> Generator[Session, None, None]:
    """Dependency to get database session.

    Yields:
        Database session
    """
    yield from get_db()


def get_current_user_optional(token: str | None = Depends(oauth2_scheme), db: Session = Depends(get_db)) -> User | None:
    """Return the current user if a valid token is provided, otherwise None."""
    if not token:
        return None
    payload = decode_access_token(token)
    user_id = payload.get('sub')
    if not user_id:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail='Invalid token payload')
    user = db.query(User).filter(User.id == user_id, User.is_active == 1).first()
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail='User not found or inactive')
    return user


def require_authenticated_user(current_user: User | None = Depends(get_current_user_optional)) -> User:
    """Require an authenticated user, respecting settings.AUTH_REQUIRED."""
    if current_user is None:
        if settings.AUTH_REQUIRED:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail='Authentication required')
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail='Authentication required')
    return current_user
