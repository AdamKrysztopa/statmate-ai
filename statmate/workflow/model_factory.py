"""Model factory for creating AI models with dependency injection.

This module provides factory functions to create AI models with proper
configuration, making the code more testable and easier to change.
"""

from pydantic_ai.models.openai import Model, ModelSettings, OpenAIModel

from statmate.config import Config, default_config


class ModelFactory:
    """Factory for creating AI models with consistent configuration."""

    def __init__(self, config: Config | None = None):
        """Initialize the model factory.

        Args:
            config: Configuration object. If None, uses default_config.
        """
        self.config = config or default_config

    def create_model(
        self,
        model_name: str | None = None,
        temperature: float | None = None,
        **overrides: any,
    ) -> Model:
        """Create an AI model with configuration.

        Args:
            model_name: Name of the model. If None, uses config default.
            temperature: Temperature setting. If None, uses config default.
            **overrides: Additional model parameter overrides.

        Returns:
            Configured Model instance.
        """
        name = model_name or self.config.model.model_name
        return OpenAIModel(name)

    def create_model_settings(
        self,
        temperature: float | None = None,
        top_p: float | None = None,
        frequency_penalty: float | None = None,
        presence_penalty: float | None = None,
        max_tokens: int | None = None,
    ) -> ModelSettings:
        """Create model settings with configuration.

        Args:
            temperature: Temperature setting. If None, uses config default.
            top_p: Top-p setting. If None, uses config default.
            frequency_penalty: Frequency penalty. If None, uses config default.
            presence_penalty: Presence penalty. If None, uses config default.
            max_tokens: Maximum tokens. If None, uses config default.

        Returns:
            Configured ModelSettings instance.
        """
        return ModelSettings(
            temperature=temperature if temperature is not None else self.config.model.temperature,
            top_p=top_p if top_p is not None else self.config.model.top_p,
            frequency_penalty=frequency_penalty
            if frequency_penalty is not None
            else self.config.model.frequency_penalty,
            presence_penalty=presence_penalty if presence_penalty is not None else self.config.model.presence_penalty,
            max_tokens=max_tokens if max_tokens is not None else self.config.model.max_tokens,
        )


# Global factory instance
default_factory = ModelFactory()


def create_model(model_name: str | None = None, **overrides: any) -> Model:
    """Create an AI model using the default factory.

    Args:
        model_name: Name of the model. If None, uses config default.
        **overrides: Additional model parameter overrides.

    Returns:
        Configured Model instance.
    """
    return default_factory.create_model(model_name=model_name, **overrides)


def create_model_settings(**overrides: any) -> ModelSettings:
    """Create model settings using the default factory.

    Args:
        **overrides: Model setting overrides.

    Returns:
        Configured ModelSettings instance.
    """
    return default_factory.create_model_settings(**overrides)
