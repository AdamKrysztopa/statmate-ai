"""Application settings and configuration management.

This module uses Pydantic Settings for environment-based configuration.
Settings are loaded from environment variables or .env file.
"""

from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import Field, field_validator
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

    # OpenAI
    OPENAI_API_KEY: str = Field(default='')

    # Security
    SECRET_KEY: str = Field(default='your-secret-key-change-in-production')
    CORS_ORIGINS: list[str] = Field(default=['http://localhost:8501', 'http://localhost:3000'])

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


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance.

    Returns:
        Settings instance
    """
    return Settings()


# Global settings instance
settings = get_settings()
