"""Application settings and configuration management.

This module uses Pydantic Settings for environment-based configuration.
Settings are loaded from environment variables or .env file.
"""

import base64
import secrets
from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables.

    Attributes:
        # Application
        APP_NAME: Application name
        APP_VERSION: Application version
        DEBUG: Debug mode flag
        ENVIRONMENT: Environment (development, staging, production)

        # API
        API_HOST: API server host
        API_PORT: API server port
        API_PREFIX: API route prefix

        # Database
        DATABASE_URL: SQLAlchemy database URL
        DATABASE_ECHO: Echo SQL statements (for debugging)

        # Storage
        DATA_DIR: Root directory for data storage
        UPLOAD_DIR: Directory for uploaded files
        RESULTS_DIR: Directory for analysis results
        LOGS_DIR: Directory for logs
        MAX_UPLOAD_SIZE: Maximum file upload size in bytes

        # OpenAI
        OPENAI_API_KEY: OpenAI API key for LLM agents

        # Security
        SECRET_KEY: Secret key for JWT tokens
        CORS_ORIGINS: Allowed CORS origins

        # Scheduler
        SCHEDULER_TIMEZONE: Timezone for scheduled tasks
        SCHEDULER_JOBSTORE: APScheduler jobstore type
    """

    model_config = SettingsConfigDict(
        env_file='.env',
        env_file_encoding='utf-8',
        case_sensitive=True,
        extra='ignore',
    )

    # Application
    APP_NAME: str = 'StatmateAI'
    APP_VERSION: str = '0.1.0'
    DEBUG: bool = Field(default=False)
    ENVIRONMENT: str = Field(default='development')

    # API
    API_HOST: str = Field(default='0.0.0.0')
    API_PORT: int = Field(default=8000)
    API_PREFIX: str = Field(default='/api/v1')

    # Database
    DATABASE_URL: str = Field(default='sqlite:///./database/statmate.db')
    DATABASE_ECHO: bool = Field(default=False)

    # Storage
    DATA_DIR: Path = Field(default=Path('./data'))
    UPLOAD_DIR: Path | None = None
    RESULTS_DIR: Path | None = None
    LOGS_DIR: Path | None = None
    MAX_UPLOAD_SIZE: int = Field(default=100 * 1024 * 1024)  # 100 MB

    # AI Model Configuration
    # Default model settings
    DEFAULT_MODEL_PROVIDER: str = Field(default='openai')
    DEFAULT_MODEL_NAME: str = Field(default='gpt-4o')
    MODEL_TEMPERATURE: float = Field(default=0.0)
    MODEL_TOP_P: float = Field(default=1.0)
    MODEL_FREQUENCY_PENALTY: float = Field(default=0.0)
    MODEL_PRESENCE_PENALTY: float = Field(default=0.0)
    MODEL_MAX_TOKENS: int | None = Field(default=None)
    REQUIRE_REASONING_MODELS_FOR_TOOLS: bool = Field(default=True)

    # OpenAI
    OPENAI_API_KEY: str = Field(default='')
    OPENAI_API_BASE: str | None = Field(default=None)

    # Anthropic
    ANTHROPIC_API_KEY: str = Field(default='')
    ANTHROPIC_API_BASE: str | None = Field(default=None)

    # Google/Gemini
    GOOGLE_API_KEY: str = Field(default='')
    GEMINI_API_KEY: str = Field(default='')  # Alias for Google

    # Groq
    GROQ_API_KEY: str = Field(default='')
    GROQ_API_BASE: str | None = Field(default=None)

    # Ollama (Local models)
    OLLAMA_ENABLED: bool = Field(default=False)
    OLLAMA_BASE_URL: str = Field(default='http://localhost:11434/v1')
    OLLAMA_DEFAULT_MODEL: str = Field(default='deepseek-r1:8b')  # Reasoning model by default

    # Security
    SECRET_KEY: str = Field(default='', description='Required: JWT signing key')
    PASSWORD_PEPPER: str = Field(default='', description='Required: server-side pepper for passwords')
    EMAIL_HASH_SECRET: str = Field(default='', description='Required: keyed HMAC secret for email hashing')
    EMAIL_ENCRYPTION_KEY: str = Field(
        default='',
        description='Required: base64url-encoded 32-byte key used to encrypt stored emails',
    )
    CORS_ORIGINS: list[str] = Field(default=['http://localhost:8501', 'http://localhost:3000'])
    AUTH_REQUIRED: bool = Field(default=True, description='Require authentication for API access')
    ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(default=60, description='Access token lifetime in minutes')
    TOKEN_ALGORITHM: str = Field(default='HS256', description='JWT signing algorithm')

    # Scheduler
    SCHEDULER_TIMEZONE: str = Field(default='UTC')
    SCHEDULER_JOBSTORE: str = Field(default='memory')  # 'memory' or 'sqlite'

    @field_validator('DATA_DIR', mode='before')
    @classmethod
    def resolve_data_dir(cls, v: Any) -> Path:
        """Resolve DATA_DIR to absolute path."""
        if isinstance(v, str):
            v = Path(v)
        if not v.is_absolute():
            v = Path.cwd() / v
        return v

    def model_post_init(self, __context: Any) -> None:
        """Initialize derived paths after model creation."""
        # Set up storage directories
        self.UPLOAD_DIR = self.DATA_DIR / 'uploads'
        self.RESULTS_DIR = self.DATA_DIR / 'results'
        self.LOGS_DIR = self.DATA_DIR / 'logs'

        # Create directories if they don't exist
        for directory in [self.DATA_DIR, self.UPLOAD_DIR, self.RESULTS_DIR, self.LOGS_DIR]:
            if directory:
                directory.mkdir(parents=True, exist_ok=True)

    @model_validator(mode='after')
    def validate_secrets(self) -> 'Settings':
        """Fail fast when security-critical secrets are not configured."""
        missing: list[str] = []

        is_dev = self.ENVIRONMENT.lower() == 'development'
        placeholder_secret = 'your-secret-key-change-in-production-use-random-string'

        if not self.SECRET_KEY or self.SECRET_KEY == placeholder_secret:
            if is_dev:
                self.SECRET_KEY = secrets.token_hex(32)
            else:
                missing.append('SECRET_KEY')

        if not self.PASSWORD_PEPPER:
            if is_dev:
                self.PASSWORD_PEPPER = secrets.token_hex(32)
            else:
                missing.append('PASSWORD_PEPPER')

        if not self.EMAIL_HASH_SECRET:
            if is_dev:
                self.EMAIL_HASH_SECRET = secrets.token_hex(32)
            else:
                missing.append('EMAIL_HASH_SECRET')
        elif self.EMAIL_HASH_SECRET == self.SECRET_KEY:
            raise ValueError('EMAIL_HASH_SECRET must differ from SECRET_KEY to avoid key reuse')

        if not self.EMAIL_ENCRYPTION_KEY:
            if is_dev:
                self.EMAIL_ENCRYPTION_KEY = base64.urlsafe_b64encode(secrets.token_bytes(32)).decode()
            else:
                missing.append('EMAIL_ENCRYPTION_KEY (base64url-encoded 32-byte key)')
        else:
            try:
                decoded = base64.urlsafe_b64decode(self.EMAIL_ENCRYPTION_KEY.encode())
            except Exception as exc:  # noqa: BLE001
                raise ValueError('EMAIL_ENCRYPTION_KEY must be valid base64url-encoded') from exc
            if len(decoded) != 32:
                raise ValueError('EMAIL_ENCRYPTION_KEY must decode to exactly 32 bytes')

        if missing:
            raise ValueError(f'The following secrets must be set: {", ".join(missing)}')
        return self

    @property
    def database_path(self) -> Path | None:
        """Get database file path if using SQLite."""
        if self.DATABASE_URL.startswith('sqlite:///'):
            db_path = self.DATABASE_URL.replace('sqlite:///', '')
            path = Path(db_path)
            # Create parent directory if it doesn't exist
            path.parent.mkdir(parents=True, exist_ok=True)
            return path
        return None

    def get_upload_path(self, filename: str) -> Path:
        """Get full path for an uploaded file.

        Args:
            filename: Name of the file

        Returns:
            Full path to the file in upload directory
        """
        if self.UPLOAD_DIR is None:
            msg = 'UPLOAD_DIR not initialized'
            raise ValueError(msg)
        return self.UPLOAD_DIR / filename

    def get_results_path(self, analysis_id: str) -> Path:
        """Get directory path for analysis results.

        Args:
            analysis_id: UUID of the analysis

        Returns:
            Full path to the results directory
        """
        if self.RESULTS_DIR is None:
            msg = 'RESULTS_DIR not initialized'
            raise ValueError(msg)
        results_dir = self.RESULTS_DIR / analysis_id
        results_dir.mkdir(parents=True, exist_ok=True)
        return results_dir

    def get_log_path(self, analysis_id: str) -> Path:
        """Get path for analysis log file.

        Args:
            analysis_id: UUID of the analysis

        Returns:
            Full path to the log file
        """
        if self.LOGS_DIR is None:
            msg = 'LOGS_DIR not initialized'
            raise ValueError(msg)
        return self.LOGS_DIR / f'{analysis_id}.log'

    def create_multi_model_config(self) -> 'MultiModelConfig':
        """Create MultiModelConfig from settings.

        Returns:
            MultiModelConfig instance with all configured providers.
        """
        from statmate.core.model_config import (
            ModelProvider,
            ModelProviderConfig,
            MultiModelConfig,
        )

        # Create provider configurations
        providers = {}

        # OpenAI
        if self.OPENAI_API_KEY:
            providers[ModelProvider.OPENAI] = ModelProviderConfig(
                provider=ModelProvider.OPENAI,
                api_key=self.OPENAI_API_KEY,
                api_base=self.OPENAI_API_BASE,
                enabled=True,
            )

        # Anthropic
        if self.ANTHROPIC_API_KEY:
            providers[ModelProvider.ANTHROPIC] = ModelProviderConfig(
                provider=ModelProvider.ANTHROPIC,
                api_key=self.ANTHROPIC_API_KEY,
                api_base=self.ANTHROPIC_API_BASE,
                enabled=True,
            )

        # Google/Gemini (prefer GOOGLE_API_KEY, fallback to GEMINI_API_KEY)
        google_key = self.GOOGLE_API_KEY or self.GEMINI_API_KEY
        if google_key:
            providers[ModelProvider.GOOGLE] = ModelProviderConfig(
                provider=ModelProvider.GOOGLE,
                api_key=google_key,
                enabled=True,
            )

        # Groq
        if self.GROQ_API_KEY:
            providers[ModelProvider.GROQ] = ModelProviderConfig(
                provider=ModelProvider.GROQ,
                api_key=self.GROQ_API_KEY,
                api_base=self.GROQ_API_BASE,
                enabled=True,
            )

        # Ollama
        if self.OLLAMA_ENABLED:
            providers[ModelProvider.OLLAMA] = ModelProviderConfig(
                provider=ModelProvider.OLLAMA,
                api_key=None,  # Ollama doesn't need API key
                api_base=self.OLLAMA_BASE_URL,
                default_model=self.OLLAMA_DEFAULT_MODEL,
                enabled=True,
            )

        # Create multi-model config
        try:
            default_provider = ModelProvider(self.DEFAULT_MODEL_PROVIDER.lower())
        except ValueError:
            default_provider = ModelProvider.OPENAI

        return MultiModelConfig(
            default_provider=default_provider,
            default_model_name=self.DEFAULT_MODEL_NAME,
            providers=providers,
            temperature=self.MODEL_TEMPERATURE,
            top_p=self.MODEL_TOP_P,
            frequency_penalty=self.MODEL_FREQUENCY_PENALTY,
            presence_penalty=self.MODEL_PRESENCE_PENALTY,
            max_tokens=self.MODEL_MAX_TOKENS,
            require_reasoning_models=self.REQUIRE_REASONING_MODELS_FOR_TOOLS,
        )


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance.

    Returns:
        Settings instance
    """
    return Settings()


# Global settings instance
settings = get_settings()
