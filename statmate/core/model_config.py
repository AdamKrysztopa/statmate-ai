"""Enhanced model configuration supporting multiple AI providers.

This module provides flexible configuration for various AI model providers
including OpenAI, Anthropic, Google, Ollama (local), and Groq.
"""

import os
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
    # OpenAI models (2025 - Latest)
    # NOTE: Reasoning models (gpt-5, o1, o3) only support temperature=1 (default)
    'gpt-5': ModelInfo(
        name='gpt-5',
        provider=ModelProvider.OPENAI,
        display_name='GPT-5',
        capabilities=[ModelCapability.REASONING, ModelCapability.FUNCTION_CALLING, ModelCapability.VISION],
        context_window=256000,
        supports_tools=True,
        description='Latest GPT-5 reasoning model - requires default temperature=1',
    ),
    'gpt-4o': ModelInfo(
        name='gpt-4o',
        provider=ModelProvider.OPENAI,
        display_name='GPT-4o',
        capabilities=[ModelCapability.REASONING, ModelCapability.FUNCTION_CALLING, ModelCapability.VISION],
        context_window=128000,
        supports_tools=True,
        description='Latest OpenAI model with vision and tool calling',
    ),
    'gpt-4o-2024-11-20': ModelInfo(
        name='gpt-4o-2024-11-20',
        provider=ModelProvider.OPENAI,
        display_name='GPT-4o (Nov 2024)',
        capabilities=[ModelCapability.REASONING, ModelCapability.FUNCTION_CALLING, ModelCapability.VISION],
        context_window=128000,
        supports_tools=True,
        description='GPT-4o snapshot from November 2024',
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
    'gpt-4o-mini-2024-07-18': ModelInfo(
        name='gpt-4o-mini-2024-07-18',
        provider=ModelProvider.OPENAI,
        display_name='GPT-4o Mini (Jul 2024)',
        capabilities=[ModelCapability.REASONING, ModelCapability.FUNCTION_CALLING, ModelCapability.VISION],
        context_window=128000,
        supports_tools=True,
        description='GPT-4o Mini snapshot from July 2024',
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
    'gpt-4-turbo-2024-04-09': ModelInfo(
        name='gpt-4-turbo-2024-04-09',
        provider=ModelProvider.OPENAI,
        display_name='GPT-4 Turbo (Apr 2024)',
        capabilities=[ModelCapability.REASONING, ModelCapability.FUNCTION_CALLING],
        context_window=128000,
        supports_tools=True,
        description='GPT-4 Turbo snapshot from April 2024',
    ),
    'o1': ModelInfo(
        name='o1',
        provider=ModelProvider.OPENAI,
        display_name='OpenAI o1',
        capabilities=[ModelCapability.REASONING, ModelCapability.FUNCTION_CALLING],
        context_window=200000,
        supports_tools=True,
        description='OpenAI reasoning model - best for complex reasoning (requires temperature=1)',
    ),
    'o1-preview': ModelInfo(
        name='o1-preview',
        provider=ModelProvider.OPENAI,
        display_name='OpenAI o1 Preview',
        capabilities=[ModelCapability.REASONING, ModelCapability.FUNCTION_CALLING],
        context_window=128000,
        supports_tools=True,
        description='Preview version of o1 reasoning model (requires temperature=1)',
    ),
    'o1-mini': ModelInfo(
        name='o1-mini',
        provider=ModelProvider.OPENAI,
        display_name='OpenAI o1 Mini',
        capabilities=[ModelCapability.REASONING, ModelCapability.FUNCTION_CALLING],
        context_window=128000,
        supports_tools=True,
        description='Faster reasoning model (requires temperature=1)',
    ),
    'o3-mini': ModelInfo(
        name='o3-mini',
        provider=ModelProvider.OPENAI,
        display_name='OpenAI o3 Mini',
        capabilities=[ModelCapability.REASONING, ModelCapability.FUNCTION_CALLING],
        context_window=128000,
        supports_tools=True,
        description='Latest mini reasoning model (requires temperature=1)',
    ),
    # Anthropic models (2025/2026 - Latest)
    'claude-sonnet-4-5-20250514': ModelInfo(
        name='claude-sonnet-4-5-20250514',
        provider=ModelProvider.ANTHROPIC,
        display_name='Claude Sonnet 4.5',
        capabilities=[ModelCapability.REASONING, ModelCapability.FUNCTION_CALLING, ModelCapability.VISION],
        context_window=200000,
        supports_tools=True,
        description='Claude Sonnet 4.5 - advanced reasoning model (May 2025)',
    ),
    'claude-sonnet-4-5-latest': ModelInfo(
        name='claude-sonnet-4-5-latest',
        provider=ModelProvider.ANTHROPIC,
        display_name='Claude Sonnet 4.5 (Latest)',
        capabilities=[ModelCapability.REASONING, ModelCapability.FUNCTION_CALLING, ModelCapability.VISION],
        context_window=200000,
        supports_tools=True,
        description='Always uses the latest Claude Sonnet 4.5',
    ),
    'claude-opus-4-20250514': ModelInfo(
        name='claude-opus-4-20250514',
        provider=ModelProvider.ANTHROPIC,
        display_name='Claude Opus 4',
        capabilities=[ModelCapability.REASONING, ModelCapability.FUNCTION_CALLING, ModelCapability.VISION],
        context_window=200000,
        supports_tools=True,
        description='Claude Opus 4 - most capable model (May 2025)',
    ),
    'claude-opus-4-latest': ModelInfo(
        name='claude-opus-4-latest',
        provider=ModelProvider.ANTHROPIC,
        display_name='Claude Opus 4 (Latest)',
        capabilities=[ModelCapability.REASONING, ModelCapability.FUNCTION_CALLING, ModelCapability.VISION],
        context_window=200000,
        supports_tools=True,
        description='Always uses the latest Claude Opus 4',
    ),
    'claude-haiku-4-20250514': ModelInfo(
        name='claude-haiku-4-20250514',
        provider=ModelProvider.ANTHROPIC,
        display_name='Claude Haiku 4',
        capabilities=[ModelCapability.REASONING, ModelCapability.FUNCTION_CALLING],
        context_window=200000,
        supports_tools=True,
        description='Claude Haiku 4 - fast and efficient (May 2025)',
    ),
    'claude-haiku-4-latest': ModelInfo(
        name='claude-haiku-4-latest',
        provider=ModelProvider.ANTHROPIC,
        display_name='Claude Haiku 4 (Latest)',
        capabilities=[ModelCapability.REASONING, ModelCapability.FUNCTION_CALLING],
        context_window=200000,
        supports_tools=True,
        description='Always uses the latest Claude Haiku 4',
    ),
    'claude-3-7-sonnet-20250219': ModelInfo(
        name='claude-3-7-sonnet-20250219',
        provider=ModelProvider.ANTHROPIC,
        display_name='Claude 3.7 Sonnet',
        capabilities=[ModelCapability.REASONING, ModelCapability.FUNCTION_CALLING, ModelCapability.VISION],
        context_window=200000,
        supports_tools=True,
        description='Claude 3.7 with extended thinking - Feb 2025',
    ),
    'claude-3-5-sonnet-20241022': ModelInfo(
        name='claude-3-5-sonnet-20241022',
        provider=ModelProvider.ANTHROPIC,
        display_name='Claude 3.5 Sonnet (Oct 2024)',
        capabilities=[ModelCapability.REASONING, ModelCapability.FUNCTION_CALLING, ModelCapability.VISION],
        context_window=200000,
        supports_tools=True,
        description='Claude 3.5 Sonnet October 2024 snapshot',
    ),
    'claude-3-5-sonnet-latest': ModelInfo(
        name='claude-3-5-sonnet-latest',
        provider=ModelProvider.ANTHROPIC,
        display_name='Claude 3.5 Sonnet (Latest)',
        capabilities=[ModelCapability.REASONING, ModelCapability.FUNCTION_CALLING, ModelCapability.VISION],
        context_window=200000,
        supports_tools=True,
        description='Always uses the latest Claude 3.5 Sonnet',
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
    'claude-3-5-haiku-latest': ModelInfo(
        name='claude-3-5-haiku-latest',
        provider=ModelProvider.ANTHROPIC,
        display_name='Claude 3.5 Haiku (Latest)',
        capabilities=[ModelCapability.REASONING, ModelCapability.FUNCTION_CALLING],
        context_window=200000,
        supports_tools=True,
        description='Always uses the latest Claude 3.5 Haiku',
    ),
    'claude-3-opus-20240229': ModelInfo(
        name='claude-3-opus-20240229',
        provider=ModelProvider.ANTHROPIC,
        display_name='Claude 3 Opus',
        capabilities=[ModelCapability.REASONING, ModelCapability.FUNCTION_CALLING, ModelCapability.VISION],
        context_window=200000,
        supports_tools=True,
        description='Most capable Claude 3 model',
    ),
    'claude-3-opus-latest': ModelInfo(
        name='claude-3-opus-latest',
        provider=ModelProvider.ANTHROPIC,
        display_name='Claude 3 Opus (Latest)',
        capabilities=[ModelCapability.REASONING, ModelCapability.FUNCTION_CALLING, ModelCapability.VISION],
        context_window=200000,
        supports_tools=True,
        description='Always uses the latest Claude 3 Opus',
    ),
    # Google models (2025 - Latest)
    'gemini-2.5-pro-preview': ModelInfo(
        name='gemini-2.5-pro-preview',
        provider=ModelProvider.GOOGLE,
        display_name='Gemini 2.5 Pro Preview',
        capabilities=[ModelCapability.REASONING, ModelCapability.FUNCTION_CALLING, ModelCapability.VISION],
        context_window=2000000,
        supports_tools=True,
        description='Latest Gemini 2.5 Pro preview (when available)',
    ),
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
        description='Google Gemini Pro with huge context window (2M tokens)',
    ),
    'gemini-1.5-pro-latest': ModelInfo(
        name='gemini-1.5-pro-latest',
        provider=ModelProvider.GOOGLE,
        display_name='Gemini 1.5 Pro (Latest)',
        capabilities=[ModelCapability.REASONING, ModelCapability.FUNCTION_CALLING, ModelCapability.VISION],
        context_window=2000000,
        supports_tools=True,
        description='Always uses the latest Gemini 1.5 Pro',
    ),
    'gemini-1.5-flash': ModelInfo(
        name='gemini-1.5-flash',
        provider=ModelProvider.GOOGLE,
        display_name='Gemini 1.5 Flash',
        capabilities=[ModelCapability.REASONING, ModelCapability.FUNCTION_CALLING, ModelCapability.VISION],
        context_window=1000000,
        supports_tools=True,
        description='Fast and efficient Gemini model',
    ),
    'gemini-1.5-flash-latest': ModelInfo(
        name='gemini-1.5-flash-latest',
        provider=ModelProvider.GOOGLE,
        display_name='Gemini 1.5 Flash (Latest)',
        capabilities=[ModelCapability.REASONING, ModelCapability.FUNCTION_CALLING, ModelCapability.VISION],
        context_window=1000000,
        supports_tools=True,
        description='Always uses the latest Gemini 1.5 Flash',
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


# Friendly / shorthand aliases → canonical model names in SUPPORTED_MODELS.
# Users can type these in the UI instead of the full identifier.
MODEL_ALIASES: dict[str, str] = {
    # Anthropic friendly names
    'sonnet-4.5': 'claude-sonnet-4-5-20250514',
    'sonnet-4-5': 'claude-sonnet-4-5-20250514',
    'claude-sonnet-4.5': 'claude-sonnet-4-5-20250514',
    'claude-sonnet-4-5': 'claude-sonnet-4-5-latest',
    'opus-4': 'claude-opus-4-20250514',
    'claude-opus-4': 'claude-opus-4-latest',
    'haiku-4': 'claude-haiku-4-20250514',
    'claude-haiku-4': 'claude-haiku-4-latest',
    'claude-3.7-sonnet': 'claude-3-7-sonnet-20250219',
    'claude-3.5-sonnet': 'claude-3-5-sonnet-latest',
    'claude-3.5-haiku': 'claude-3-5-haiku-latest',
    'claude-3-opus': 'claude-3-opus-latest',
    # OpenAI friendly names
    'gpt5': 'gpt-5',
    'gpt4o': 'gpt-4o',
    'gpt-4o-mini': 'gpt-4o-mini',
    'o1-mini': 'o1-mini',
    # Google friendly names
    'gemini-pro': 'gemini-1.5-pro-latest',
    'gemini-flash': 'gemini-1.5-flash-latest',
    'gemini-2-flash': 'gemini-2.0-flash-exp',
    'gemini-2.5-pro': 'gemini-2.5-pro-preview',
}


def resolve_model_alias(model_name: str) -> str:
    """Resolve a friendly model alias to its canonical name.

    If *model_name* is already a canonical key in ``SUPPORTED_MODELS``, it is
    returned unchanged.  Otherwise the ``MODEL_ALIASES`` mapping is consulted.
    """
    if model_name in SUPPORTED_MODELS:
        return model_name
    return MODEL_ALIASES.get(model_name, model_name)


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
    """Create default multi-model configuration from environment variables."""
    # Default provider configs (can be overridden by a settings file)
    openai_api_key = os.getenv('OPENAI_API_KEY')
    default_providers = {
        ModelProvider.OPENAI: ModelProviderConfig(
            provider=ModelProvider.OPENAI,
            api_key=openai_api_key,
            default_model='gpt-4o',
            enabled=bool(openai_api_key),
        )
    }

    return MultiModelConfig(
        default_provider=ModelProvider.OPENAI,
        default_model_name='gpt-4o',
        providers=default_providers,
    )
