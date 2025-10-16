"""Enhanced model configuration supporting multiple AI providers.

This module provides flexible configuration for various AI model providers
including OpenAI, Anthropic, Google, Ollama (local), and Groq.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ModelProvider(str, Enum):
    """Supported AI model providers."""

    OPENAI = 'openai'
    ANTHROPIC = 'anthropic'
    GOOGLE = 'google'
    OLLAMA = 'ollama'
    GROQ = 'groq'
    GEMINI = 'gemini'


class ModelCapability(str, Enum):
    """Model capabilities."""

    REASONING = 'reasoning'  # Models with reasoning/tool calling
    CHAT = 'chat'  # Standard chat models
    VISION = 'vision'  # Vision capabilities
    FUNCTION_CALLING = 'function_calling'  # Function/tool calling


@dataclass
class ModelInfo:
    """Information about a specific model."""

    name: str
    provider: ModelProvider
    display_name: str
    capabilities: list[ModelCapability]
    context_window: int
    supports_tools: bool = True
    description: str = ''

    @property
    def full_name(self) -> str:
        """Get full model name with provider prefix."""
        return f'{self.provider.value}:{self.name}'


# Popular models with reasoning capabilities
SUPPORTED_MODELS: dict[str, ModelInfo] = {
    # OpenAI models
    'gpt-4o': ModelInfo(
        name='gpt-4o',
        provider=ModelProvider.OPENAI,
        display_name='GPT-4o',
        capabilities=[ModelCapability.REASONING, ModelCapability.FUNCTION_CALLING, ModelCapability.VISION],
        context_window=128000,
        supports_tools=True,
        description='Latest OpenAI model with vision and tool calling',
    ),
    'gpt-4o-mini': ModelInfo(
        name='gpt-4o-mini',
        provider=ModelProvider.OPENAI,
        display_name='GPT-4o Mini',
        capabilities=[ModelCapability.REASONING, ModelCapability.FUNCTION_CALLING, ModelCapability.VISION],
        context_window=128000,
        supports_tools=True,
        description='Faster and cheaper version of GPT-4o',
    ),
    'gpt-4-turbo': ModelInfo(
        name='gpt-4-turbo',
        provider=ModelProvider.OPENAI,
        display_name='GPT-4 Turbo',
        capabilities=[ModelCapability.REASONING, ModelCapability.FUNCTION_CALLING],
        context_window=128000,
        supports_tools=True,
        description='High-performance GPT-4 with extended context',
    ),
    'o1': ModelInfo(
        name='o1',
        provider=ModelProvider.OPENAI,
        display_name='OpenAI o1',
        capabilities=[ModelCapability.REASONING, ModelCapability.FUNCTION_CALLING],
        context_window=200000,
        supports_tools=True,
        description='OpenAI reasoning model',
    ),
    'o1-mini': ModelInfo(
        name='o1-mini',
        provider=ModelProvider.OPENAI,
        display_name='OpenAI o1 Mini',
        capabilities=[ModelCapability.REASONING, ModelCapability.FUNCTION_CALLING],
        context_window=128000,
        supports_tools=True,
        description='Faster reasoning model',
    ),
    # Anthropic models
    'claude-3-7-sonnet-20250219': ModelInfo(
        name='claude-3-7-sonnet-20250219',
        provider=ModelProvider.ANTHROPIC,
        display_name='Claude 3.7 Sonnet',
        capabilities=[ModelCapability.REASONING, ModelCapability.FUNCTION_CALLING, ModelCapability.VISION],
        context_window=200000,
        supports_tools=True,
        description='Latest Claude with extended thinking',
    ),
    'claude-3-5-sonnet-20241022': ModelInfo(
        name='claude-3-5-sonnet-20241022',
        provider=ModelProvider.ANTHROPIC,
        display_name='Claude 3.5 Sonnet',
        capabilities=[ModelCapability.REASONING, ModelCapability.FUNCTION_CALLING, ModelCapability.VISION],
        context_window=200000,
        supports_tools=True,
        description='Claude 3.5 Sonnet',
    ),
    'claude-3-5-haiku-20241022': ModelInfo(
        name='claude-3-5-haiku-20241022',
        provider=ModelProvider.ANTHROPIC,
        display_name='Claude 3.5 Haiku',
        capabilities=[ModelCapability.REASONING, ModelCapability.FUNCTION_CALLING],
        context_window=200000,
        supports_tools=True,
        description='Fast and efficient Claude model',
    ),
    'claude-3-opus-20240229': ModelInfo(
        name='claude-3-opus-20240229',
        provider=ModelProvider.ANTHROPIC,
        display_name='Claude 3 Opus',
        capabilities=[ModelCapability.REASONING, ModelCapability.FUNCTION_CALLING, ModelCapability.VISION],
        context_window=200000,
        supports_tools=True,
        description='Most capable Claude model',
    ),
    # Google models
    'gemini-2.0-flash-exp': ModelInfo(
        name='gemini-2.0-flash-exp',
        provider=ModelProvider.GOOGLE,
        display_name='Gemini 2.0 Flash',
        capabilities=[ModelCapability.REASONING, ModelCapability.FUNCTION_CALLING, ModelCapability.VISION],
        context_window=1000000,
        supports_tools=True,
        description='Google Gemini 2.0 with thinking',
    ),
    'gemini-2.0-flash-thinking-exp-01-21': ModelInfo(
        name='gemini-2.0-flash-thinking-exp-01-21',
        provider=ModelProvider.GOOGLE,
        display_name='Gemini 2.0 Flash Thinking',
        capabilities=[ModelCapability.REASONING, ModelCapability.FUNCTION_CALLING, ModelCapability.VISION],
        context_window=1000000,
        supports_tools=True,
        description='Google Gemini 2.0 with extended thinking',
    ),
    'gemini-1.5-pro': ModelInfo(
        name='gemini-1.5-pro',
        provider=ModelProvider.GOOGLE,
        display_name='Gemini 1.5 Pro',
        capabilities=[ModelCapability.REASONING, ModelCapability.FUNCTION_CALLING, ModelCapability.VISION],
        context_window=2000000,
        supports_tools=True,
        description='Google Gemini Pro with huge context window',
    ),
    # Groq models (fast inference)
    'llama-3.3-70b-versatile': ModelInfo(
        name='llama-3.3-70b-versatile',
        provider=ModelProvider.GROQ,
        display_name='Llama 3.3 70B',
        capabilities=[ModelCapability.REASONING, ModelCapability.FUNCTION_CALLING],
        context_window=128000,
        supports_tools=True,
        description='Meta Llama 3.3 70B on Groq',
    ),
    'llama-3.3-70b-specdec': ModelInfo(
        name='llama-3.3-70b-specdec',
        provider=ModelProvider.GROQ,
        display_name='Llama 3.3 70B Speculative',
        capabilities=[ModelCapability.REASONING, ModelCapability.FUNCTION_CALLING],
        context_window=8192,
        supports_tools=True,
        description='Fast inference Llama 3.3 70B',
    ),
    # Ollama local models (examples - user can add more)
    'llama3.1:8b': ModelInfo(
        name='llama3.1:8b',
        provider=ModelProvider.OLLAMA,
        display_name='Llama 3.1 8B (Local)',
        capabilities=[ModelCapability.REASONING, ModelCapability.FUNCTION_CALLING],
        context_window=128000,
        supports_tools=True,
        description='Local Llama 3.1 8B via Ollama',
    ),
    'deepseek-r1:8b': ModelInfo(
        name='deepseek-r1:8b',
        provider=ModelProvider.OLLAMA,
        display_name='DeepSeek-R1 8B (Local)',
        capabilities=[ModelCapability.REASONING, ModelCapability.FUNCTION_CALLING],
        context_window=64000,
        supports_tools=True,
        description='Local DeepSeek-R1 8B reasoning model via Ollama - competes with o1',
    ),
    'deepseek-r1:14b': ModelInfo(
        name='deepseek-r1:14b',
        provider=ModelProvider.OLLAMA,
        display_name='DeepSeek-R1 14B (Local)',
        capabilities=[ModelCapability.REASONING, ModelCapability.FUNCTION_CALLING],
        context_window=64000,
        supports_tools=True,
        description='Local DeepSeek-R1 14B reasoning model via Ollama - high quality',
    ),
    'deepseek-r1:70b': ModelInfo(
        name='deepseek-r1:70b',
        provider=ModelProvider.OLLAMA,
        display_name='DeepSeek-R1 70B (Local)',
        capabilities=[ModelCapability.REASONING, ModelCapability.FUNCTION_CALLING],
        context_window=64000,
        supports_tools=True,
        description='Local DeepSeek-R1 70B reasoning model via Ollama - top quality',
    ),
    'qwen2.5:7b': ModelInfo(
        name='qwen2.5:7b',
        provider=ModelProvider.OLLAMA,
        display_name='Qwen 2.5 7B (Local)',
        capabilities=[ModelCapability.REASONING, ModelCapability.FUNCTION_CALLING],
        context_window=32768,
        supports_tools=True,
        description='Local Qwen 2.5 7B via Ollama',
    ),
    'mistral:7b': ModelInfo(
        name='mistral:7b',
        provider=ModelProvider.OLLAMA,
        display_name='Mistral 7B (Local)',
        capabilities=[ModelCapability.REASONING, ModelCapability.FUNCTION_CALLING],
        context_window=32768,
        supports_tools=True,
        description='Local Mistral 7B via Ollama',
    ),
}


@dataclass
class ModelProviderConfig:
    """Configuration for a specific model provider."""

    provider: ModelProvider
    api_key: str | None = None
    api_base: str | None = None
    default_model: str | None = None
    enabled: bool = True
    extra_params: dict[str, Any] = field(default_factory=dict)

    def is_configured(self) -> bool:
        """Check if provider is properly configured."""
        if self.provider == ModelProvider.OLLAMA:
            # Ollama doesn't need API key, just needs to be running
            return self.enabled
        return self.enabled and bool(self.api_key)


@dataclass
class MultiModelConfig:
    """Configuration for multiple model providers."""

    # Default provider and model
    default_provider: ModelProvider = ModelProvider.OPENAI
    default_model_name: str = 'gpt-4o'

    # Provider configurations
    providers: dict[ModelProvider, ModelProviderConfig] = field(default_factory=dict)

    # Model generation settings
    temperature: float = 0.0
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    max_tokens: int | None = None
    retries: int = 3

    # Tool/reasoning settings
    require_reasoning_models: bool = True  # For tools, always use reasoning models
    fallback_to_default: bool = True  # Fallback if requested model unavailable

    def get_provider_config(self, provider: ModelProvider) -> ModelProviderConfig | None:
        """Get configuration for a specific provider."""
        return self.providers.get(provider)

    def get_available_models(self) -> list[ModelInfo]:
        """Get list of available models based on configured providers."""
        available = []
        for model_info in SUPPORTED_MODELS.values():
            provider_config = self.get_provider_config(model_info.provider)
            if provider_config and provider_config.is_configured():
                available.append(model_info)
        return available

    def get_reasoning_models(self) -> list[ModelInfo]:
        """Get list of available models with reasoning capabilities."""
        available = self.get_available_models()
        return [m for m in available if ModelCapability.REASONING in m.capabilities]

    def validate_model(self, model_name: str, for_tools: bool = False) -> tuple[bool, str]:
        """Validate if a model is available and suitable.

        Args:
            model_name: Name of the model to validate
            for_tools: Whether the model will be used with tools

        Returns:
            Tuple of (is_valid, error_message)
        """
        # Find model info
        model_info = SUPPORTED_MODELS.get(model_name)
        if not model_info:
            return False, f'Model {model_name} not found in supported models'

        # Check if provider is configured
        provider_config = self.get_provider_config(model_info.provider)
        if not provider_config or not provider_config.is_configured():
            return False, f'Provider {model_info.provider.value} is not configured'

        # Check reasoning capability for tools
        if for_tools and self.require_reasoning_models:
            if not model_info.supports_tools:
                return False, f'Model {model_name} does not support tool calling'

        return True, ''


def create_default_multi_model_config() -> MultiModelConfig:
    """Create default multi-model configuration."""
    return MultiModelConfig(
        default_provider=ModelProvider.OPENAI,
        default_model_name='gpt-4o',
        providers={},
    )
