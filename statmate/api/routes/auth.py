"""Authentication routes for user registration and login."""

import logging
from datetime import timedelta

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session

from database.models import User
from database.session import get_db
from statmate.api.models.user import TokenResponse, UserCreate, UserLogin, UserResponse
from statmate.api.security import create_access_token, get_password_hash, verify_password
from statmate.api.dependencies import get_current_user_optional
from config.settings import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix='/auth', tags=['auth'])


@router.post('/register', response_model=UserResponse, status_code=status.HTTP_201_CREATED)
def register_user(payload: UserCreate, db: Session = Depends(get_db)) -> UserResponse:
    """Register a new user account."""
    # Bcrypt only hashes the first 72 bytes; reject longer passwords up front so we return 400 instead of 500.
    if len(payload.password.encode('utf-8')) > 72:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail='Password too long (bcrypt limit is 72 bytes). Please shorten your password.',
        )
    try:
        existing = db.query(User).filter(User.email == payload.email.lower()).first()
        if existing:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail='Email already registered')

        try:
            hashed = get_password_hash(payload.password)
        except ValueError as exc:
            # passlib raises ValueError when the password exceeds bcrypt limits
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail='Password too long (bcrypt limit is 72 bytes). Please shorten your password.',
            ) from exc

        user = User(email=payload.email.lower(), hashed_password=hashed)
        db.add(user)
        db.commit()
        db.refresh(user)
        return UserResponse.model_validate(user)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception('Failed to register user %s', payload.email)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f'Failed to register user: {exc}',
        ) from exc


@router.post('/login', response_model=TokenResponse)
def login(payload: UserLogin, db: Session = Depends(get_db)) -> TokenResponse:
    """Login and obtain an access token."""
    user = db.query(User).filter(User.email == payload.email.lower()).first()
    if not user or not verify_password(payload.password, user.hashed_password):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail='Invalid credentials')
    access_token = create_access_token({'sub': user.id}, expires_delta=timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES))
    return TokenResponse(access_token=access_token)


@router.post('/token', response_model=TokenResponse)
def token(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)) -> TokenResponse:
    """OAuth2-compatible token endpoint."""
    user = db.query(User).filter(User.email == form_data.username.lower()).first()
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail='Invalid credentials')
    access_token = create_access_token({'sub': user.id}, expires_delta=timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES))
    return TokenResponse(access_token=access_token)


@router.get('/me', response_model=UserResponse)
def read_current_user(current_user: User = Depends(get_current_user_optional)) -> UserResponse:
    """Return current user info."""
    if current_user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail='Not authenticated')
    return UserResponse.model_validate(current_user)
