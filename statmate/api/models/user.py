"""User-facing Pydantic models for authentication."""

from datetime import datetime

from pydantic import BaseModel, EmailStr, Field, field_validator


class UserCreate(BaseModel):
    email: EmailStr = Field(description='User email')
    password: str = Field(min_length=8, max_length=72, description='Password (8-72 characters; bcrypt max 72 bytes)')

    @field_validator('password')
    @classmethod
    def validate_password_bytes(cls, v: str) -> str:
        """Ensure password does not exceed bcrypt's 72-byte limit."""
        if len(v.encode('utf-8')) > 72:
            msg = 'Password too long (bcrypt limit is 72 bytes). Please shorten your password.'
            raise ValueError(msg)
        return v


class UserLogin(BaseModel):
    email: EmailStr
    password: str


class UserResponse(BaseModel):
    id: str
    email: EmailStr
    created_at: datetime

    class Config:
        from_attributes = True


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = 'bearer'
