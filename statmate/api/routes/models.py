"""API routes for model configuration and selection."""

import logging

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from config.settings import settings
from database.models import User
from database.session import get_db
from statmate.api.dependencies import require_authenticated_user
from statmate.api.models.model_config import (
    AvailableModelsResponse,
    CurrentModelResponse,
    ModelInfoResponse,
    ModelValidationResponse,
)
from statmate.api.services.credential_service import CredentialService
from statmate.core.model_config import SUPPORTED_MODELS, ModelProvider, resolve_model_alias
from statmate.workflow.model_factory import get_default_factory, initialize_default_factory

logger = logging.getLogger(__name__)

router = APIRouter(prefix='/models', tags=['models'])


# Request/Response models for credentials
class CredentialsRequest(BaseModel):
    """User-provided credentials for PROD mode."""

    openai_api_key: str | None = None
    anthropic_api_key: str | None = None
    google_api_key: str | None = None
    gemini_api_key: str | None = None
    groq_api_key: str | None = None
    ollama_enabled: str | None = None
    ollama_base_url: str | None = None
    ollama_default_model: str | None = None
    openai_quota: int | None = None
    anthropic_quota: int | None = None
    google_quota: int | None = None
    gemini_quota: int | None = None
    groq_quota: int | None = None


class CredentialsResponse(BaseModel):
    """Response after setting credentials."""

    success: bool
    message: str
    configured_providers: list[str]


class CredentialsListResponse(BaseModel):
    """Providers configured for the current user."""

    configured_providers: list[str]
    stored_credentials: dict[str, str] | None = None
    provider_quotas: dict[str, int] | None = None


class EnvironmentResponse(BaseModel):
    """Current environment mode."""

    environment: str
    requires_user_credentials: bool


@router.get('/available', response_model=AvailableModelsResponse)
async def get_available_models(for_tools: bool = True, include_all: bool = False) -> AvailableModelsResponse:
    """Get list of available models.

    Args:
        for_tools: If True, only return models suitable for tool calling.

    Returns:
        AvailableModelsResponse with list of models.
    """
    try:
        factory = get_default_factory()
        if include_all:
            models = list(SUPPORTED_MODELS.values())
            if for_tools:
                models = [model for model in models if model.supports_tools]
        else:
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


@router.get('/validate', response_model=ModelValidationResponse)
async def validate_model(
    model_name: str,
    provider: str | None = None,
    for_tools: bool = True,
) -> ModelValidationResponse:
    """Validate whether a model is available for the current configuration.

    Args:
        model_name: Requested model name (may be an alias).
        provider: Optional provider override to validate against.
        for_tools: If True, require tool-calling capability.

    Returns:
        ModelValidationResponse with resolved model and validation status.
    """
    try:
        factory = get_default_factory()
        resolved = resolve_model_alias(model_name)
        model_info = factory.get_model_info(resolved)
        if not model_info:
            return ModelValidationResponse(
                requested_model=model_name,
                resolved_model=resolved,
                provider=provider or 'unknown',
                valid=False,
                error=f'Model {resolved} not found in supported models',
            )

        resolved_provider = model_info.provider.value
        if provider:
            try:
                provider_enum = ModelProvider(provider)
            except ValueError:
                return ModelValidationResponse(
                    requested_model=model_name,
                    resolved_model=resolved,
                    provider=provider,
                    valid=False,
                    error=f'Provider {provider} is not supported',
                )
            if provider_enum != model_info.provider:
                return ModelValidationResponse(
                    requested_model=model_name,
                    resolved_model=resolved,
                    provider=provider,
                    valid=False,
                    error=(
                        f'Model {resolved} belongs to provider {resolved_provider}, '
                        f'not {provider_enum.value}'
                    ),
                )

        is_valid, error = factory.multi_model_config.validate_model(resolved, for_tools=for_tools)
        return ModelValidationResponse(
            requested_model=model_name,
            resolved_model=resolved,
            provider=resolved_provider,
            valid=is_valid,
            error=error or None,
        )
    except Exception as e:
        logger.error(f'Error validating model {model_name}: {e}', exc_info=True)
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
    current_user: User = Depends(require_authenticated_user),
    db: Session = Depends(get_db),
) -> CredentialsResponse:
    """Set user credentials for PROD mode.

    Args:
        credentials: User-provided API credentials.

    Returns:
        CredentialsResponse with success status.

    Raises:
        HTTPException: If environment is not production or credentials are invalid.
    """

    try:
        incoming: dict[str, str] = {}
        quota_limits: dict[str, int] = {}
        if credentials.openai_api_key:
            incoming['openai'] = credentials.openai_api_key
        if credentials.openai_quota is not None:
            quota_limits['openai'] = credentials.openai_quota
        if credentials.anthropic_api_key:
            incoming['anthropic'] = credentials.anthropic_api_key
        if credentials.anthropic_quota is not None:
            quota_limits['anthropic'] = credentials.anthropic_quota
        google_key = credentials.google_api_key
        if credentials.gemini_api_key:
            google_key = credentials.gemini_api_key
        if google_key:
            incoming['google'] = google_key
        if credentials.google_quota is not None:
            quota_limits['google'] = credentials.google_quota
        if credentials.gemini_quota is not None:
            quota_limits['google'] = credentials.gemini_quota
        if credentials.groq_api_key:
            incoming['groq'] = credentials.groq_api_key
        if credentials.groq_quota is not None:
            quota_limits['groq'] = credentials.groq_quota

        configured_providers = CredentialService.upsert_credentials(
            db=db, user_id=current_user.id, credentials=incoming, quota_limits=quota_limits
        )
        if quota_limits:
            CredentialService.update_quota_limits(db=db, user_id=current_user.id, limits=quota_limits)

        # Apply configured providers to runtime settings for immediate use
        stored = CredentialService.load_credentials(db=db, user_id=current_user.id)
        if 'openai' in stored:
            settings.OPENAI_API_KEY = stored['openai']
        if 'anthropic' in stored:
            settings.ANTHROPIC_API_KEY = stored['anthropic']
        if 'google' in stored:
            settings.GOOGLE_API_KEY = stored['google']
        if 'groq' in stored:
            settings.GROQ_API_KEY = stored['groq']

        if credentials.ollama_enabled == 'true':
            settings.OLLAMA_ENABLED = True
            if credentials.ollama_base_url:
                settings.OLLAMA_BASE_URL = credentials.ollama_base_url
            if credentials.ollama_default_model:
                settings.OLLAMA_DEFAULT_MODEL = credentials.ollama_default_model
            configured_providers.append('ollama')

        if not configured_providers:
            raise HTTPException(status_code=400, detail='No valid credentials provided')

        # Reinitialize model factory with updated keys
        multi_model_config = settings.create_multi_model_config()
        initialize_default_factory(multi_model_config)

        return CredentialsResponse(
            success=True,
            message=f'Successfully stored {len(configured_providers)} provider credential(s)',
            configured_providers=sorted(set(configured_providers)),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f'Error setting credentials: {e}', exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get('/credentials', response_model=CredentialsListResponse)
async def get_credentials(
    current_user: User = Depends(require_authenticated_user),
    db: Session = Depends(get_db),
) -> CredentialsListResponse:
    """Return configured providers for the current user."""

    stored = CredentialService.load_credentials(db=db, user_id=current_user.id)
    quotas = CredentialService.get_quota_limits(db=db, user_id=current_user.id)
    enriched = dict(stored)
    if 'google' in stored and 'gemini' not in stored:
        enriched['gemini'] = stored['google']
    providers = sorted(set(stored.keys()) | ({'gemini'} if 'google' in stored else set()))
    return CredentialsListResponse(
        configured_providers=providers, stored_credentials=enriched, provider_quotas=quotas if quotas else None
    )
