"""Model provider system for creating AI models from different providers.

This module provides a unified interface for creating models from various
providers (OpenAI, Anthropic, Google, Ollama, Groq) using Pydantic AI.
"""

import logging
from typing import Any

from pydantic_ai.models import Model

from statmate.core.model_config import (
    SUPPORTED_MODELS,
    ModelInfo,
    ModelProvider,
    ModelProviderConfig,
    MultiModelConfig,
)

logger = logging.getLogger(__name__)


class ModelProviderError(Exception):
    """Exception raised for model provider errors."""


class ModelProviderSystem:
    """System for managing and creating models from different providers."""

    def __init__(self, config: MultiModelConfig):
        """Initialize the model provider system.

        Args:
            config: Multi-model configuration
        """
        self.config = config
        self._cache: dict[str, Model] = {}

    def create_model(
        self,
        model_name: str | None = None,
        provider: ModelProvider | None = None,
        api_key: str | None = None,
        **kwargs: Any,
    ) -> Model:
        """Create a model instance.

        Args:
            model_name: Name of the model (e.g., 'gpt-4o', 'claude-3-7-sonnet-20250219')
            provider: Provider to use. If None, infers from model_name or uses default
            api_key: Optional API key override for this specific call
            **kwargs: Additional parameters for model creation

        Returns:
            Configured Model instance

        Raises:
            ModelProviderError: If model cannot be created
        """
        # Use defaults if not specified
        if model_name is None:
            model_name = self.config.default_model_name
        if provider is None:
            # Try to infer provider from model name
            model_info = SUPPORTED_MODELS.get(model_name)
            if model_info:
                provider = model_info.provider
            else:
                provider = self.config.default_provider

        # Get model info
        model_info = SUPPORTED_MODELS.get(model_name)
        if not model_info:
            if self.config.fallback_to_default:
                logger.warning(f'Model {model_name} not found, falling back to default')
                model_name = self.config.default_model_name
                model_info = SUPPORTED_MODELS.get(model_name)
                if model_info:
                    provider = model_info.provider
            else:
                msg = f'Model {model_name} not found in supported models'
                raise ModelProviderError(msg)

        # Get provider config
        provider_config = self.config.get_provider_config(provider)
        if not provider_config or not provider_config.is_configured():
            msg = f'Provider {provider.value} is not configured. Please set API key in settings.'
            raise ModelProviderError(msg)

        # Use override API key if provided
        effective_api_key = api_key or provider_config.api_key

        # Create model based on provider
        try:
            if provider == ModelProvider.OPENAI:
                return self._create_openai_model(model_name, effective_api_key, provider_config, **kwargs)
            if provider == ModelProvider.ANTHROPIC:
                return self._create_anthropic_model(model_name, effective_api_key, provider_config, **kwargs)
            if provider == ModelProvider.GOOGLE:
                return self._create_google_model(model_name, effective_api_key, provider_config, **kwargs)
            if provider == ModelProvider.GROQ:
                return self._create_groq_model(model_name, effective_api_key, provider_config, **kwargs)
            if provider == ModelProvider.OLLAMA:
                return self._create_ollama_model(model_name, provider_config, **kwargs)
            msg = f'Provider {provider.value} not yet implemented'
            raise ModelProviderError(msg)
        except Exception as e:
            msg = f'Failed to create model {model_name} from provider {provider.value}: {e}'
            logger.error(msg)
            raise ModelProviderError(msg) from e

    def _create_openai_model(
        self,
        model_name: str,
        api_key: str | None,
        provider_config: ModelProviderConfig,
        **kwargs: Any,
    ) -> Model:
        """Create OpenAI model."""
        from pydantic_ai.models.openai import OpenAIModel
        from pydantic_ai.providers.openai import OpenAIProvider

        # Create provider with API key and base_url
        provider_params = {}
        if api_key:
            provider_params['api_key'] = api_key
        if provider_config.api_base:
            provider_params['base_url'] = provider_config.api_base

        provider = OpenAIProvider(**provider_params)

        # Create model with provider
        return OpenAIModel(model_name, provider=provider, **kwargs)

    def _create_anthropic_model(
        self,
        model_name: str,
        api_key: str | None,
        provider_config: ModelProviderConfig,
        **kwargs: Any,
    ) -> Model:
        """Create Anthropic model."""
        from pydantic_ai.models.anthropic import AnthropicModel
        from pydantic_ai.providers.anthropic import AnthropicProvider

        # Create provider with API key and base_url
        provider_params = {}
        if api_key:
            provider_params['api_key'] = api_key
        if provider_config.api_base:
            provider_params['base_url'] = provider_config.api_base

        provider = AnthropicProvider(**provider_params)

        # Create model with provider
        return AnthropicModel(model_name, provider=provider, **kwargs)

    def _create_google_model(
        self,
        model_name: str,
        api_key: str | None,
        provider_config: ModelProviderConfig,
        **kwargs: Any,
    ) -> Model:
        """Create Google/Gemini model."""
        from pydantic_ai.models.gemini import GeminiModel
        from pydantic_ai.providers.gemini import GeminiProvider

        # Create provider with API key
        provider_params = {}
        if api_key:
            provider_params['api_key'] = api_key

        provider = GeminiProvider(**provider_params)

        # Create model with provider
        return GeminiModel(model_name, provider=provider, **kwargs)

    def _create_groq_model(
        self,
        model_name: str,
        api_key: str | None,
        provider_config: ModelProviderConfig,
        **kwargs: Any,
    ) -> Model:
        """Create Groq model."""
        from pydantic_ai.models.openai import OpenAIModel
        from pydantic_ai.providers.groq import GroqProvider

        # Create provider with API key and base_url
        provider_params = {}
        if api_key:
            provider_params['api_key'] = api_key
        if provider_config.api_base:
            provider_params['base_url'] = provider_config.api_base

        provider = GroqProvider(**provider_params)

        # Create model with provider (Groq uses OpenAI-compatible API)
        return OpenAIModel(model_name, provider=provider, **kwargs)

    def _create_ollama_model(
        self,
        model_name: str,
        provider_config: ModelProviderConfig,
        **kwargs: Any,
    ) -> Model:
        """Create Ollama (local) model."""
        from pydantic_ai.models.openai import OpenAIModel
        from pydantic_ai.providers.ollama import OllamaProvider

        # Default Ollama base URL
        base_url = provider_config.api_base or 'http://localhost:11434/v1'

        # Create provider with base_url
        provider = OllamaProvider(base_url=base_url)

        # Create model with provider (Ollama uses OpenAI-compatible API)
        return OpenAIModel(model_name, provider=provider, **kwargs)

    def get_model_for_tools(
        self,
        model_name: str | None = None,
        provider: ModelProvider | None = None,
        **kwargs: Any,
    ) -> Model:
        """Get a model suitable for tool/function calling.

        This method ensures the returned model supports reasoning and tool calling,
        which is required for agent operations.

        Args:
            model_name: Name of the model
            provider: Provider to use
            **kwargs: Additional parameters

        Returns:
            Model instance with tool calling support

        Raises:
            ModelProviderError: If no suitable model is available
        """
        # If no model specified, find best available reasoning model
        if model_name is None:
            reasoning_models = self.config.get_reasoning_models()
            if not reasoning_models:
                msg = 'No reasoning models available. Please configure at least one provider with a reasoning model.'
                raise ModelProviderError(msg)
            # Use the first available reasoning model
            model_info = reasoning_models[0]
            model_name = model_info.name
            provider = model_info.provider
            logger.info(f'Using default reasoning model: {model_name} from {provider.value}')

        # Validate model supports tools
        is_valid, error = self.config.validate_model(model_name, for_tools=True)
        if not is_valid:
            if self.config.fallback_to_default:
                logger.warning(f'{error}. Falling back to default reasoning model.')
                return self.get_model_for_tools(model_name=None, **kwargs)
            raise ModelProviderError(error)

        return self.create_model(model_name=model_name, provider=provider, **kwargs)

    def list_available_models(self, for_tools: bool = False) -> list[ModelInfo]:
        """List all available models.

        Args:
            for_tools: If True, only list models suitable for tool calling

        Returns:
            List of available model information
        """
        if for_tools:
            return self.config.get_reasoning_models()
        return self.config.get_available_models()

    def get_model_info(self, model_name: str) -> ModelInfo | None:
        """Get information about a specific model.

        Args:
            model_name: Name of the model

        Returns:
            ModelInfo if found, None otherwise
        """
        return SUPPORTED_MODELS.get(model_name)


def create_model_provider_system(config: MultiModelConfig) -> ModelProviderSystem:
    """Create a model provider system instance.

    Args:
        config: Multi-model configuration

    Returns:
        ModelProviderSystem instance
    """
    return ModelProviderSystem(config)
