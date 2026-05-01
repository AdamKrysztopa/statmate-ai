"""Model provider system for creating AI models from different providers.

This module provides a unified interface for creating models from various
providers (OpenAI, Anthropic, Google, Ollama, Groq) using Pydantic AI.
"""

import logging
import math
import time
from collections.abc import Callable
from typing import Any

from pydantic_ai.models import Model

from statmate.core.model_config import (
    SUPPORTED_MODELS,
    ModelInfo,
    ModelProvider,
    ModelProviderConfig,
    MultiModelConfig,
    resolve_model_alias,
)

logger = logging.getLogger(__name__)


def _retry_after_seconds(exc: Exception) -> float | None:
    """Extract Retry-After seconds when available on provider errors."""
    for attr in ('retry_after', 'retry_after_ms', 'retry_after_seconds'):
        if hasattr(exc, attr):
            try:
                value = getattr(exc, attr)
                return float(value) / (1000 if 'ms' in attr else 1)
            except Exception:
                continue
    response = getattr(exc, 'response', None)
    if response is not None:
        header = getattr(response, 'headers', {}) or {}
        retry_after = header.get('Retry-After') or header.get('retry-after')
        if retry_after:
            try:
                return float(retry_after)
            except Exception:
                return None
    return None


def is_rate_limit_error(exc: Exception) -> bool:
    """Heuristic to detect rate limit or quota errors from providers."""
    msg = str(exc).lower()
    status = getattr(exc, 'status_code', None) or getattr(getattr(exc, 'response', None), 'status_code', None)
    return any(
        token in msg for token in ['rate limit', 'retry', 'too many requests', '429', 'quota']
    ) or status == 429


def execute_with_backoff(
    func: Callable[[], Any],
    *,
    max_retries: int = 3,
    base_delay: float = 1.0,
    on_retry: Callable[[int, float, Exception], None] | None = None,
) -> Any:
    """Execute a callable with exponential backoff respecting Retry-After."""
    attempt = 0
    while True:
        try:
            return func()
        except Exception as exc:  # pragma: no cover - exercised at runtime
            attempt += 1
            if attempt > max_retries or not is_rate_limit_error(exc):
                raise
            delay = _retry_after_seconds(exc)
            if delay is None:
                delay = base_delay * math.pow(2, attempt - 1)
            if on_retry:
                try:
                    on_retry(attempt, delay, exc)
                except Exception:
                    pass
            time.sleep(delay)


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

        # Resolve friendly aliases (e.g. 'sonnet-4.5' → 'claude-sonnet-4-5-20250514')
        original_name = model_name
        model_name = resolve_model_alias(model_name)
        if model_name != original_name:
            logger.info('Resolved model alias %s → %s', original_name, model_name)

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
