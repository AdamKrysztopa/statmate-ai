"""API routes for model configuration and selection."""

import logging

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from config.settings import settings
from statmate.api.models.model_config import (
    AvailableModelsResponse,
    CurrentModelResponse,
    ModelInfoResponse,
)
from statmate.workflow.model_factory import get_default_factory, initialize_default_factory
from statmate.api.dependencies import get_current_user_optional
from database.models import User

logger = logging.getLogger(__name__)

router = APIRouter(prefix='/models', tags=['models'])


# Request/Response models for credentials
class CredentialsRequest(BaseModel):
    """User-provided credentials for PROD mode."""

    openai_api_key: str | None = None
    anthropic_api_key: str | None = None
    google_api_key: str | None = None
    groq_api_key: str | None = None
    ollama_enabled: str | None = None
    ollama_base_url: str | None = None
    ollama_default_model: str | None = None


class CredentialsResponse(BaseModel):
    """Response after setting credentials."""

    success: bool
    message: str
    configured_providers: list[str]


class EnvironmentResponse(BaseModel):
    """Current environment mode."""

    environment: str
    requires_user_credentials: bool


@router.get('/available', response_model=AvailableModelsResponse)
async def get_available_models(for_tools: bool = True) -> AvailableModelsResponse:
    """Get list of available models.

    Args:
        for_tools: If True, only return models suitable for tool calling.

    Returns:
        AvailableModelsResponse with list of models.
    """
    try:
        factory = get_default_factory()
        models = factory.list_available_models(for_tools=for_tools)

        model_responses = [
            ModelInfoResponse(
                name=model.name,
                provider=model.provider.value,
                display_name=model.display_name,
                description=model.description,
                context_window=model.context_window,
                supports_tools=model.supports_tools,
                capabilities=[cap.value for cap in model.capabilities],
            )
            for model in models
        ]

        return AvailableModelsResponse(
            models=model_responses,
            default_model=settings.DEFAULT_MODEL_NAME,
            default_provider=settings.DEFAULT_MODEL_PROVIDER,
        )
    except Exception as e:
        logger.error(f'Error getting available models: {e}', exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get('/current', response_model=CurrentModelResponse)
async def get_current_model() -> CurrentModelResponse:
    """Get current model configuration.

    Returns:
        CurrentModelResponse with current settings.
    """
    try:
        factory = get_default_factory()
        multi_model_config = factory.multi_model_config

        # Get list of configured providers
        configured_providers = [
            provider.value for provider, config in multi_model_config.providers.items() if config.is_configured()
        ]

        return CurrentModelResponse(
            model_name=multi_model_config.default_model_name,
            provider=multi_model_config.default_provider.value,
            temperature=multi_model_config.temperature,
            require_reasoning_for_tools=multi_model_config.require_reasoning_models,
            available_providers=configured_providers,
        )
    except Exception as e:
        logger.error(f'Error getting current model: {e}', exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get('/info/{model_name}', response_model=ModelInfoResponse)
async def get_model_info(model_name: str) -> ModelInfoResponse:
    """Get information about a specific model.

    Args:
        model_name: Name of the model.

    Returns:
        ModelInfoResponse with model details.

    Raises:
        HTTPException: If model not found.
    """
    try:
        factory = get_default_factory()
        model_info = factory.get_model_info(model_name)

        if not model_info:
            raise HTTPException(status_code=404, detail=f'Model {model_name} not found')

        return ModelInfoResponse(
            name=model_info.name,
            provider=model_info.provider.value,
            display_name=model_info.display_name,
            description=model_info.description,
            context_window=model_info.context_window,
            supports_tools=model_info.supports_tools,
            capabilities=[cap.value for cap in model_info.capabilities],
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f'Error getting model info for {model_name}: {e}', exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get('/environment', response_model=EnvironmentResponse)
async def get_environment() -> EnvironmentResponse:
    """Get current environment mode.

    Returns:
        EnvironmentResponse with environment mode.
    """
    is_prod = settings.ENVIRONMENT.lower() == 'production'
    return EnvironmentResponse(
        environment=settings.ENVIRONMENT.lower(),
        requires_user_credentials=is_prod,
    )


@router.post('/credentials', response_model=CredentialsResponse)
async def set_credentials(
    credentials: CredentialsRequest,
    current_user: User | None = Depends(get_current_user_optional),
) -> CredentialsResponse:
    """Set user credentials for PROD mode.

    Args:
        credentials: User-provided API credentials.

    Returns:
        CredentialsResponse with success status.

    Raises:
        HTTPException: If environment is not production or credentials are invalid.
    """
    # Only allow in production mode
    if settings.ENVIRONMENT.lower() != 'production':
        raise HTTPException(
            status_code=403,
            detail='Credential management only available in production mode',
        )

    if settings.AUTH_REQUIRED and not current_user:
        raise HTTPException(status_code=401, detail='Authentication required')

    try:
        # Update settings with user-provided credentials
        configured_providers = []

        if credentials.openai_api_key:
            settings.OPENAI_API_KEY = credentials.openai_api_key
            configured_providers.append('openai')
            logger.info('OpenAI credentials configured')

        if credentials.anthropic_api_key:
            settings.ANTHROPIC_API_KEY = credentials.anthropic_api_key
            configured_providers.append('anthropic')
            logger.info('Anthropic credentials configured')

        if credentials.google_api_key:
            settings.GOOGLE_API_KEY = credentials.google_api_key
            configured_providers.append('google')
            logger.info('Google credentials configured')

        if credentials.groq_api_key:
            settings.GROQ_API_KEY = credentials.groq_api_key
            configured_providers.append('groq')
            logger.info('Groq credentials configured')

        if credentials.ollama_enabled == 'true':
            settings.OLLAMA_ENABLED = True
            if credentials.ollama_base_url:
                settings.OLLAMA_BASE_URL = credentials.ollama_base_url
            if credentials.ollama_default_model:
                settings.OLLAMA_DEFAULT_MODEL = credentials.ollama_default_model
            configured_providers.append('ollama')
            logger.info('Ollama configured')

        if not configured_providers:
            raise HTTPException(
                status_code=400,
                detail='No valid credentials provided',
            )

        # Reinitialize model factory with new credentials
        multi_model_config = settings.create_multi_model_config()
        initialize_default_factory(multi_model_config)

        logger.info(f'Model factory reinitialized with {len(configured_providers)} provider(s)')

        return CredentialsResponse(
            success=True,
            message=f'Successfully configured {len(configured_providers)} provider(s)',
            configured_providers=configured_providers,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f'Error setting credentials: {e}', exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e
