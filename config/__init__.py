"""Configuration package for StatmateAI.

This package contains application settings and configuration management using
Pydantic Settings for type-safe, environment-based configuration.

Modules:
    settings: Pydantic Settings model with all application configuration

Usage:
    from config import settings

    # Access configuration
    api_host = settings.API_HOST
    db_url = settings.DATABASE_URL

    # Get file paths
    upload_path = settings.get_upload_path('file.csv')
    results_path = settings.get_results_path(analysis_id)

Configuration Sources (in order of precedence):
    1. Environment variables (highest priority)
    2. .env file in project root
    3. Default values in Settings class (lowest priority)

Key Settings:
    Application:
        - APP_NAME: Application name (default: StatmateAI)
        - APP_VERSION: Version number
        - DEBUG: Debug mode flag
        - ENVIRONMENT: dev/staging/production

    API:
        - API_HOST: Server host (default: 0.0.0.0)
        - API_PORT: Server port (default: 8000)
        - API_PREFIX: API route prefix (default: /api/v1)

    Database:
        - DATABASE_URL: SQLAlchemy connection string
        - DATABASE_ECHO: Log SQL queries (default: False)

    Storage:
        - DATA_DIR: Root directory for data files
        - MAX_UPLOAD_SIZE: Max file upload size in bytes

    Security:
        - SECRET_KEY: JWT secret key
        - CORS_ORIGINS: Allowed CORS origins

    OpenAI:
        - OPENAI_API_KEY: API key for LLM agents (REQUIRED)

Environment File (.env):
    Create a .env file in the project root:

    ```
    OPENAI_API_KEY=sk-your-key-here
    DATABASE_URL=sqlite:///./database/statmate.db
    DEBUG=True
    ```

Validation:
    - All settings are validated on application startup
    - Type mismatches raise ValidationError
    - Required fields without defaults must be provided
    - Paths are automatically created if they don't exist
"""

from config.settings import Settings, settings

__all__ = ['Settings', 'settings']
