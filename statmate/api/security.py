"""Security utilities for authentication and authorization."""

import hashlib
import hmac
import re
from datetime import datetime, timedelta
from typing import Any

import jwt
from cryptography.fernet import Fernet, InvalidToken
from fastapi import HTTPException, status
from passlib.context import CryptContext

from config.settings import settings

# Strong, memory-hard hashing for passwords. Pepper is appended before hashing.
pwd_context = CryptContext(
    schemes=['argon2'],
    deprecated='auto',
    argon2__time_cost=3,
    argon2__memory_cost=64 * 1024,  # 64 MB
    argon2__parallelism=2,
)


def _pepper_password(password: str) -> str:
    """Append server-side pepper to the password prior to hashing."""
    pepper = settings.PASSWORD_PEPPER
    if not pepper:
        raise ValueError('PASSWORD_PEPPER must be set for secure password hashing')
    return password + pepper


def validate_password_strength(password: str) -> None:
    """Enforce a basic strong-password policy."""
    if len(password) < 12:
        raise ValueError('Password must be at least 12 characters')
    if not re.search(r'[A-Z]', password):
        raise ValueError('Password must include an uppercase letter')
    if not re.search(r'[a-z]', password):
        raise ValueError('Password must include a lowercase letter')
    if not re.search(r'[0-9]', password):
        raise ValueError('Password must include a digit')
    if not re.search(r'[^A-Za-z0-9]', password):
        raise ValueError('Password must include a symbol')


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a plaintext password against a hashed password."""
    return pwd_context.verify(_pepper_password(plain_password), hashed_password)


def get_password_hash(password: str) -> str:
    """Hash a password for storage."""
    validate_password_strength(password)
    return pwd_context.hash(_pepper_password(password))


def _get_email_hash_key() -> bytes:
    key = settings.EMAIL_HASH_SECRET or settings.SECRET_KEY
    if not key:
        raise ValueError('EMAIL_HASH_SECRET or SECRET_KEY must be set')
    return key.encode('utf-8')


def hash_email(email: str) -> str:
    """Return a stable, keyed hash of the normalized email for lookups."""
    normalized = normalize_email(email)
    digest = hmac.new(_get_email_hash_key(), normalized.encode('utf-8'), hashlib.sha256).hexdigest()
    return digest


def _get_email_cipher() -> Fernet:
    key = settings.EMAIL_ENCRYPTION_KEY
    if not key:
        raise ValueError('EMAIL_ENCRYPTION_KEY must be set (base64url-encoded 32-byte key)')
    try:
        return Fernet(key.encode('utf-8'))
    except Exception as exc:  # noqa: BLE001
        raise ValueError('Invalid EMAIL_ENCRYPTION_KEY; must be base64url-encoded 32-byte key') from exc


def encrypt_email(email: str) -> str:
    """Encrypt the normalized email for storage."""
    cipher = _get_email_cipher()
    token = cipher.encrypt(normalize_email(email).encode('utf-8'))
    return token.decode('utf-8')


def decrypt_email(token: str) -> str:
    """Decrypt an email ciphertext back to plaintext."""
    cipher = _get_email_cipher()
    try:
        return cipher.decrypt(token.encode('utf-8')).decode('utf-8')
    except InvalidToken as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail='Invalid email ciphertext'
        ) from exc


def _get_credential_cipher() -> Fernet:
    key = settings.API_CREDENTIAL_KEY
    if not key:
        raise ValueError('API_CREDENTIAL_KEY must be set (base64url-encoded 32-byte key)')
    try:
        return Fernet(key.encode('utf-8'))
    except Exception as exc:  # noqa: BLE001
        raise ValueError('Invalid API_CREDENTIAL_KEY; must be base64url-encoded 32-byte key') from exc


def encrypt_secret(value: str) -> str:
    """Encrypt a secret value (e.g., API key) for storage."""
    cipher = _get_credential_cipher()
    token = cipher.encrypt(value.encode('utf-8'))
    return token.decode('utf-8')


def decrypt_secret(token: str) -> str:
    """Decrypt a stored secret value."""
    cipher = _get_credential_cipher()
    try:
        return cipher.decrypt(token.encode('utf-8')).decode('utf-8')
    except InvalidToken as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail='Invalid credential ciphertext'
        ) from exc


def normalize_email(email: str) -> str:
    """Lowercase and strip surrounding whitespace for consistent storage and hashing."""
    return email.strip().lower()


def create_access_token(data: dict[str, Any], expires_delta: timedelta | None = None) -> str:
    """Create a signed JWT access token."""
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({'exp': expire})
    return jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.TOKEN_ALGORITHM)


def decode_access_token(token: str) -> dict[str, Any]:
    """Decode and validate an access token."""
    try:
        return jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.TOKEN_ALGORITHM])
    except jwt.ExpiredSignatureError as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail='Token expired') from exc
    except jwt.PyJWTError as exc:  # type: ignore[attr-defined]
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail='Invalid token') from exc
