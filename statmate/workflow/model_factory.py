"""Model factory for creating AI models with dependency injection.

This module provides factory functions to create AI models with proper
configuration, making the code more testable and easier to change.

Now supports multiple model providers including OpenAI, Anthropic, Google,
Groq, and Ollama (local models).
"""

from typing import Any

from pydantic_ai.models import Model, ModelSettings

from statmate.core.config import Config, default_config
from statmate.core.model_config import (
    ModelInfo,
    ModelProvider,
    MultiModelConfig,
    create_default_multi_model_config,
)
from statmate.core.model_provider import create_model_provider_system


class ModelFactory:
    """Factory for creating AI models with consistent configuration.

    This enhanced factory supports multiple providers and allows users to:
    - Choose from popular models (OpenAI, Anthropic, Google, Groq, Ollama)
    - Provide API keys per provider
    - Use local models via Ollama
    - Automatically select reasoning-capable models for tools
    """

    def __init__(
        self,
        config: Config | None = None,
        multi_model_config: MultiModelConfig | None = None,
    ):
        """Initialize the model factory.

        Args:
            config: Configuration object. If None, uses default_config.
            multi_model_config: Multi-model configuration. If None, creates default.
        """
        self.config = config or default_config
        self.multi_model_config = multi_model_config or create_default_multi_model_config()
        self.provider_system = create_model_provider_system(self.multi_model_config)

    def create_model(
        self,
        model_name: str | None = None,
        provider: ModelProvider | str | None = None,
        api_key: str | None = None,
        for_tools: bool = False,
        **overrides: Any,
    ) -> Model:
        """Create an AI model with configuration.

        Args:
            model_name: Name of the model (e.g., 'gpt-4o', 'claude-3-7-sonnet-20250219').
                       If None, uses configured default.
            provider: Provider to use. Can be ModelProvider enum or string.
                     If None, infers from model_name.
            api_key: Optional API key override for this specific call.
            for_tools: If True, ensures model supports tool/function calling.
            **overrides: Additional model parameter overrides.

        Returns:
            Configured Model instance.

        Raises:
            ModelProviderError: If model cannot be created.
        """
        # Convert string provider to enum if needed
        if isinstance(provider, str):
            provider = ModelProvider(provider)

        # Use tool-specific model if requested
        if for_tools:
            return self.provider_system.get_model_for_tools(
                model_name=model_name,
                provider=provider,
                api_key=api_key,
                **overrides,
            )

        return self.provider_system.create_model(
            model_name=model_name,
            provider=provider,
            api_key=api_key,
            **overrides,
        )

    def create_model_settings(
        self,
        model_name: str | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        frequency_penalty: float | None = None,
        presence_penalty: float | None = None,
        max_tokens: int | None = None,
    ) -> ModelSettings:
        """Create model settings with configuration.

        Automatically adjusts settings for models with parameter restrictions
        (e.g., GPT-5, o1, o3 require temperature=1).

        Args:
            model_name: Model name to check for restrictions. If None, uses defaults.
            temperature: Temperature setting. If None, uses config default.
            top_p: Top-p setting. If None, uses config default.
            frequency_penalty: Frequency penalty. If None, uses config default.
            presence_penalty: Presence penalty. If None, uses config default.
            max_tokens: Maximum tokens. If None, uses config default.

        Returns:
            Configured ModelSettings instance.
        """
        # Resolve model name (handle None or "default")
        effective_model_name = self._resolve_model_name(model_name)

        # Check if model has parameter restrictions
        # For restricted models (GPT-5, o1, etc.), this returns None to skip temperature
        if effective_model_name:
            adjusted_temp = self._adjust_temperature_for_model(effective_model_name, temperature)
        else:
            adjusted_temp = temperature

        # Build settings dict, omitting temperature for restricted models
        settings_dict = {}

        # Include temperature only if not restricted (adjusted_temp != None from restriction check)
        if effective_model_name and adjusted_temp is None and self._is_restricted_model(effective_model_name):
            # Restricted model - omit temperature entirely to use API default
            pass
        elif adjusted_temp is not None:
            # Explicit temperature provided and not restricted
            settings_dict['temperature'] = adjusted_temp
        else:
            # No explicit temperature and not restricted - use config default
            settings_dict['temperature'] = self.multi_model_config.temperature

        settings_dict['top_p'] = top_p if top_p is not None else self.multi_model_config.top_p
        settings_dict['frequency_penalty'] = (
            frequency_penalty if frequency_penalty is not None else self.multi_model_config.frequency_penalty
        )
        settings_dict['presence_penalty'] = (
            presence_penalty if presence_penalty is not None else self.multi_model_config.presence_penalty
        )
        if max_tokens is not None or self.multi_model_config.max_tokens is not None:
            settings_dict['max_tokens'] = max_tokens if max_tokens is not None else self.multi_model_config.max_tokens

        return ModelSettings(**settings_dict)

    def _resolve_model_name(self, model_name: str | None) -> str | None:
        """Resolve model name to actual model (handle None or "default").

        Args:
            model_name: Model name to resolve

        Returns:
            Resolved model name, or None if unable to resolve
        """
        # If None or "default", get the actual default reasoning model
        if model_name is None or model_name == 'default':
            reasoning_models = self.multi_model_config.get_reasoning_models()
            if reasoning_models:
                return reasoning_models[0].name
            # Fallback to config default
            return self.multi_model_config.default_model_name
        return model_name

    def _is_restricted_model(self, model_name: str) -> bool:
        """Check if a model has parameter restrictions.

        Args:
            model_name: Name of the model

        Returns:
            True if model has temperature restrictions, False otherwise
        """
        # Models that require temperature=1 (reasoning models)
        restricted_models = [
            'gpt-5',
            'gpt-5-mini',
            'gpt-5-nano',
            'gpt-5-pro',
            'o1',
            'o1-preview',
            'o1-mini',
            'o3',
            'o3-mini',
        ]

        # Check if model is restricted (partial match)
        for restricted in restricted_models:
            if restricted in model_name.lower():
                return True

        return False

    def _adjust_temperature_for_model(self, model_name: str, temperature: float | None) -> float | None:
        """Adjust temperature for models with restrictions.

        Some models (GPT-5, o1, o3 series) only support default temperature=1.

        Args:
            model_name: Name of the model
            temperature: Requested temperature

        Returns:
            Adjusted temperature or None to use defaults
        """
        if self._is_restricted_model(model_name):
            # Return None to let the API use its default (typically 1)
            return None

        return temperature

    def list_available_models(self, for_tools: bool = False) -> list[ModelInfo]:
        """List all available models based on configured providers.

        Args:
            for_tools: If True, only list models suitable for tool calling.

        Returns:
            List of available model information.
        """
        return self.provider_system.list_available_models(for_tools=for_tools)

    def get_model_info(self, model_name: str) -> ModelInfo | None:
        """Get information about a specific model.

        Args:
            model_name: Name of the model.

        Returns:
            ModelInfo if found, None otherwise.
        """
        return self.provider_system.get_model_info(model_name)


# Global factory instance - will be initialized with proper config
default_factory: ModelFactory | None = None


def initialize_default_factory(multi_model_config: MultiModelConfig) -> None:
    """Initialize the default factory with a multi-model configuration.

    This should be called during application startup with the loaded configuration.

    Args:
        multi_model_config: Multi-model configuration from settings.
    """
    global default_factory
    default_factory = ModelFactory(multi_model_config=multi_model_config)


def get_default_factory() -> ModelFactory:
    """Get the default factory instance.

    Returns:
        ModelFactory instance.
    """
    global default_factory
    if default_factory is None:
        # Create with default config if not initialized
        default_factory = ModelFactory()
    return default_factory


def create_model(
    model_name: str | None = None,
    provider: ModelProvider | str | None = None,
    for_tools: bool = False,
    **overrides: Any,
) -> Model:
    """Create an AI model using the default factory.

    Args:
        model_name: Name of the model. If None, uses config default.
        provider: Provider to use. If None, infers from model_name.
        for_tools: If True, ensures model supports tool/function calling.
        **overrides: Additional model parameter overrides.

    Returns:
        Configured Model instance.
    """
    factory = get_default_factory()
    return factory.create_model(model_name=model_name, provider=provider, for_tools=for_tools, **overrides)


def create_model_settings(model_name: str | None = None, **overrides: Any) -> ModelSettings:
    """Create model settings using the default factory.

    Args:
        model_name: Model name to check for restrictions.
        **overrides: Model setting overrides.

    Returns:
        Configured ModelSettings instance.
    """
    factory = get_default_factory()
    return factory.create_model_settings(model_name=model_name, **overrides)
