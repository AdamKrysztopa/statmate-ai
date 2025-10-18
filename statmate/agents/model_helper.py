"""Helper utilities for creating models in agents.

This module provides convenience functions for agents to create models
using the flexible model factory system.
"""

from typing import Any

from pydantic_ai.models import Model, ModelSettings

from statmate.core.model_config import ModelProvider
from statmate.workflow.model_factory import create_model, create_model_settings


def get_agent_model(
    model_name: str | None = None,
    provider: ModelProvider | str | None = None,
    api_key: str | None = None,
    **overrides: Any,
) -> Model:
    """Get a model for agent use (ensures tool/reasoning support).

    This is the recommended way for agents to create models, as it ensures
    the model supports tool calling and reasoning capabilities.

    Args:
        model_name: Name of the model (e.g., 'gpt-4o', 'claude-3-7-sonnet-20250219').
                   If None, uses the configured default reasoning model.
        provider: Provider to use. Can be ModelProvider enum or string.
                 If None, infers from model_name.
        api_key: Optional API key override for this specific call.
        **overrides: Additional model parameter overrides.

    Returns:
        Configured Model instance with tool calling support.

    Example:
        >>> model = get_agent_model()  # Uses default
        >>> model = get_agent_model(model_name='claude-3-7-sonnet-20250219')
        >>> model = get_agent_model(model_name='llama3.1:8b', provider='ollama')
    """
    return create_model(
        model_name=model_name,
        provider=provider,
        api_key=api_key,
        for_tools=True,  # Always use reasoning models for agents
        **overrides,
    )


def get_agent_model_settings(
    model_name: str | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    frequency_penalty: float | None = None,
    presence_penalty: float | None = None,
    max_tokens: int | None = None,
) -> ModelSettings:
    """Get model settings for agent use.

    Args:
        model_name: Name of the model (for parameter restriction adjustments).
        temperature: Temperature setting. If None, uses config default.
        top_p: Top-p setting. If None, uses config default.
        frequency_penalty: Frequency penalty. If None, uses config default.
        presence_penalty: Presence penalty. If None, uses config default.
        max_tokens: Maximum tokens. If None, uses config default.

    Returns:
        Configured ModelSettings instance.

    Example:
        >>> settings = get_agent_model_settings(temperature=0.0)
        >>> settings = get_agent_model_settings(temperature=0.7, max_tokens=1000)
    """
    return create_model_settings(
        model_name=model_name,
        temperature=temperature,
        top_p=top_p,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        max_tokens=max_tokens,
    )


def create_agent_model_and_settings(
    model_name: str | None = None,
    provider: ModelProvider | str | None = None,
    temperature: float = 0.0,
    top_p: float = 1.0,
    frequency_penalty: float = 0.0,
    presence_penalty: float = 0.0,
    max_tokens: int | None = None,
) -> tuple[Model, ModelSettings]:
    """Create both model and settings for agent use.

    This is a convenience function that creates both the model and settings
    in one call, which is common in agent initialization.

    Args:
        model_name: Name of the model. If None, uses default.
        provider: Provider to use. If None, infers from model_name.
        temperature: Temperature setting.
        top_p: Top-p setting.
        frequency_penalty: Frequency penalty.
        presence_penalty: Presence penalty.
        max_tokens: Maximum tokens.

    Returns:
        Tuple of (Model, ModelSettings).

    Example:
        >>> model, settings = create_agent_model_and_settings()
        >>> model, settings = create_agent_model_and_settings(
        ...     model_name='gpt-4o',
        ...     temperature=0.0,
        ...     max_tokens=500
        ... )
    """
    model = get_agent_model(model_name=model_name, provider=provider)
    settings = get_agent_model_settings(
        model_name=model_name,
        temperature=temperature,
        top_p=top_p,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        max_tokens=max_tokens,
    )
    return model, settings
