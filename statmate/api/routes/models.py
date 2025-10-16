"""API routes for model configuration and selection."""

import logging

from fastapi import APIRouter, HTTPException

from config.settings import settings
from statmate.api.models.model_config import (
    AvailableModelsResponse,
    CurrentModelResponse,
    ModelInfoResponse,
)
from statmate.workflow.model_factory import get_default_factory

logger = logging.getLogger(__name__)

router = APIRouter(prefix='/models', tags=['models'])


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
