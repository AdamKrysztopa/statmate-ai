# Flexible Model System - Implementation Summary

## Overview

The StatMate AI flexible model system has been successfully implemented, allowing users to:

1. ✅ **Choose from popular AI providers**: OpenAI, Anthropic, Google, Groq
2. ✅ **Use local models via Ollama**: Run models privately without API costs
3. ✅ **Configure multiple providers simultaneously**: Automatic fallbacks
4. ✅ **Ensure reasoning models for tools**: Automatic validation for agent operations
5. ✅ **Easy provider and API key management**: Environment variable configuration

## What Changed

### New Files Created

1. **`statmate/core/model_config.py`**
   - `ModelProvider`: Enum for supported providers
   - `ModelInfo`: Metadata about each model
   - `MultiModelConfig`: Configuration for multiple providers
   - `SUPPORTED_MODELS`: Registry of 20+ popular models

2. **`statmate/core/model_provider.py`**
   - `ModelProviderSystem`: Creates models from different providers
   - Provider-specific instantiation logic
   - Automatic model validation and fallbacks

3. **`statmate/agents/model_helper.py`**
   - `get_agent_model()`: Convenience function for agents
   - `get_agent_model_settings()`: Create model settings
   - `create_agent_model_and_settings()`: One-call setup

4. **`docs/MODEL_CONFIGURATION.md`**
   - Comprehensive guide for all providers
   - Setup instructions for each provider
   - Model selection tips and cost considerations

5. **`examples/flexible_model_usage.py`**
   - 8 practical examples
   - Demonstrates all key features

### Modified Files

1. **`statmate/workflow/model_factory.py`**
   - Enhanced `ModelFactory` with multi-provider support
   - Added `initialize_default_factory()` for app initialization
   - Maintains backward compatibility

2. **`config/settings.py`**
   - Added settings for all providers
   - Added `create_multi_model_config()` method
   - Supports API keys for: OpenAI, Anthropic, Google, Groq, Ollama

3. **`statmate/api/main.py`**
   - Initializes model factory on startup
   - Logs available models
   - Warns if no models configured

4. **`statmate/workflow/statmate_flow.py`**
   - Updated all agent instantiations to use flexible models
   - Removed hardcoded `OpenAIModel('gpt-4o')`
   - Uses `get_agent_model()` and `create_agent_model_and_settings()`

5. **`statmate/core/__init__.py`**
   - Exports new model configuration classes

6. **`statmate/agents/__init__.py`**
   - Exports model helper functions

7. **`pyproject.toml`**
   - Added dependencies for all providers:
     - `openai>=1.0.0`
     - `anthropic>=0.18.0`
     - `google-generativeai>=0.3.0`
     - `groq>=0.4.0`
     - `ollama>=0.1.0`

## Supported Models

### OpenAI (Default)
- `gpt-4o` - Latest with vision (recommended)
- `gpt-4o-mini` - Faster and cheaper
- `gpt-4-turbo` - High performance
- `o1` - Reasoning model
- `o1-mini` - Faster reasoning

### Anthropic (Claude)
- `claude-3-7-sonnet-20250219` - Latest with extended thinking
- `claude-3-5-sonnet-20241022` - Recommended
- `claude-3-5-haiku-20241022` - Fast and efficient
- `claude-3-opus-20240229` - Most capable

### Google (Gemini)
- `gemini-2.0-flash-exp` - Fast with thinking
- `gemini-2.0-flash-thinking-exp-01-21` - Extended thinking
- `gemini-1.5-pro` - Huge context (2M tokens)

### Groq (Fast Inference)
- `llama-3.3-70b-versatile` - Recommended
- `llama-3.3-70b-specdec` - Fastest

### Ollama (Local)
- `llama3.1:8b` - Good balance
- `llama3.1:70b` - More capable
- `qwen2.5:7b` - Efficient
- `mistral:7b` - Alternative
- Any other Ollama-compatible model

## Configuration

### Environment Variables

Create a `.env` file with your configuration:

```bash
# Default provider and model
DEFAULT_MODEL_PROVIDER=openai  # openai, anthropic, google, groq, ollama
DEFAULT_MODEL_NAME=gpt-4o

# Model settings
MODEL_TEMPERATURE=0.0
REQUIRE_REASONING_MODELS_FOR_TOOLS=True

# Provider API Keys
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=...
GROQ_API_KEY=gsk_...

# Ollama (local)
OLLAMA_ENABLED=True
OLLAMA_BASE_URL=http://localhost:11434/v1
OLLAMA_DEFAULT_MODEL=llama3.1:8b
```

### Quick Start Examples

**1. Use Default Model:**
```python
from statmate.workflow.model_factory import create_model

model = create_model()  # Uses configured default
```

**2. Use Specific Model:**
```python
# OpenAI GPT-4o
model = create_model(model_name='gpt-4o')

# Anthropic Claude
model = create_model(
    model_name='claude-3-7-sonnet-20250219',
    provider='anthropic'
)

# Google Gemini
model = create_model(
    model_name='gemini-2.0-flash-exp',
    provider='google'
)

# Local Ollama
model = create_model(
    model_name='llama3.1:8b',
    provider='ollama'
)
```

**3. In Agent Code (Recommended):**
```python
from statmate.agents.model_helper import create_agent_model_and_settings

# Gets model with reasoning capabilities
model, settings = create_agent_model_and_settings(
    temperature=0.0,
    max_tokens=500
)

# Use with any agent
agent = pearson_agent(model=model, model_settings=settings)
```

**4. List Available Models:**
```python
from statmate.workflow.model_factory import get_default_factory

factory = get_default_factory()
models = factory.list_available_models(for_tools=True)

for model_info in models:
    print(f"{model_info.display_name}: {model_info.description}")
```

## Key Features

### 1. Automatic Reasoning Model Selection
When using `for_tools=True` or `get_agent_model()`, the system automatically ensures the selected model supports tool calling and reasoning capabilities.

### 2. Fallback Mechanism
If a requested model is unavailable, the system can automatically fall back to the default configured model (configurable via `fallback_to_default=True`).

### 3. Multi-Provider Support
Configure multiple providers simultaneously. Switch between them by changing environment variables without code changes.

### 4. Provider-Specific Optimizations
Each provider is instantiated with its optimal configuration:
- OpenAI: Standard API
- Anthropic: Extended thinking mode
- Google: Gemini-specific features
- Groq: Fast inference settings
- Ollama: Local endpoints

### 5. Validation and Error Handling
- Validates API keys are set before creating models
- Checks model capabilities (reasoning, tools, vision)
- Provides clear error messages for missing configuration
- Logs available models on startup

## Using Ollama (Local Models)

### Why Use Ollama?
- ✅ **Free**: No API costs
- ✅ **Private**: Data stays on your machine
- ✅ **Offline**: Works without internet
- ✅ **Fast**: No network latency
- ✅ **Flexible**: Run any compatible model

### Setup

1. **Install Ollama:**
   ```bash
   # Visit https://ollama.ai for installation
   # Or use package managers:
   # macOS: brew install ollama
   # Linux: curl -fsSL https://ollama.com/install.sh | sh
   ```

2. **Pull a Model:**
   ```bash
   ollama pull llama3.1:8b  # Or any other model
   ```

3. **Start Ollama:**
   ```bash
   ollama serve  # Runs on port 11434 by default
   ```

4. **Configure StatMate:**
   ```bash
   # In .env file
   OLLAMA_ENABLED=True
   DEFAULT_MODEL_PROVIDER=ollama
   DEFAULT_MODEL_NAME=llama3.1:8b
   ```

5. **Use in Code:**
   ```python
   # Automatically uses Ollama if configured as default
   model = create_model()
   
   # Or explicitly
   model = create_model(
       model_name='llama3.1:8b',
       provider='ollama'
   )
   ```

### Recommended Ollama Models for StatMate

- **llama3.1:8b** - Best balance of performance and resource usage
- **qwen2.5:7b** - Good for statistical reasoning
- **llama3.1:70b** - Most capable (requires 32GB+ RAM)
- **mistral:7b** - Fast alternative

## Migration Guide

### From Hardcoded OpenAI Models

**Before:**
```python
from pydantic_ai.models.openai import OpenAIModel

model = OpenAIModel('gpt-4o')
```

**After:**
```python
from statmate.agents.model_helper import get_agent_model

model = get_agent_model()  # Flexible, configured via .env
```

### From Legacy Agent Code

**Before:**
```python
model = OpenAIModel('gpt-4o')
settings = ModelSettings(temperature=0.0)
agent = pearson_agent(model=model, model_settings=settings)
```

**After:**
```python
from statmate.agents.model_helper import create_agent_model_and_settings

model, settings = create_agent_model_and_settings(temperature=0.0)
agent = pearson_agent(model=model, model_settings=settings)
```

## Best Practices

1. **Configure Default in .env**: Set `DEFAULT_MODEL_PROVIDER` and `DEFAULT_MODEL_NAME`
2. **Use Helper Functions**: Prefer `get_agent_model()` over direct model creation
3. **Set API Keys Securely**: Use environment variables, never hardcode
4. **Test with Ollama First**: Free and fast for development
5. **Use Reasoning Models for Tools**: Let the system handle validation
6. **Configure Multiple Providers**: For redundancy and fallback
7. **Monitor Costs**: Use cheaper models for development, premium for production

## Performance Comparison

| Provider  | Model             | Speed      | Cost | Reasoning | Local |
| --------- | ----------------- | ---------- | ---- | --------- | ----- |
| OpenAI    | gpt-4o            | Fast       | $$   | ✅         | ❌     |
| OpenAI    | gpt-4o-mini       | Very Fast  | $    | ✅         | ❌     |
| OpenAI    | o1                | Slow       | $$$  | ⭐⭐⭐       | ❌     |
| Anthropic | claude-3-7-sonnet | Medium     | $$$  | ⭐⭐⭐       | ❌     |
| Anthropic | claude-3-5-haiku  | Very Fast  | $    | ✅         | ❌     |
| Google    | gemini-2.0-flash  | Very Fast  | $    | ✅         | ❌     |
| Groq      | llama-3.3-70b     | Ultra Fast | $    | ✅         | ❌     |
| Ollama    | deepseek-r1:8b    | Fast       | Free | ⭐⭐⭐ 🔥     | ✅     |
| Ollama    | llama3.1:8b       | Fast       | Free | ✅         | ✅     |

## Testing

Run the example file to test your configuration:

```bash
python examples/flexible_model_usage.py
```

This will:
- Show configured models
- Test different providers
- Demonstrate all key features
- Provide setup guidance if needed

## Troubleshooting

### "Provider not configured" Error
**Solution**: Set the API key in `.env`:
```bash
OPENAI_API_KEY=your-key-here
```

### "Model not found" Error
**Solution**: Check model name spelling or configure fallback:
```bash
# In settings
fallback_to_default=True
```

### Ollama Connection Error
**Solution**: Ensure Ollama is running:
```bash
ollama serve
# In another terminal:
ollama list  # Should show pulled models
```

### Out of Memory (Ollama)
**Solution**: Use a smaller model:
```bash
ollama pull llama3.1:8b  # Instead of 70b
```

## Future Enhancements

Potential future additions:
- [ ] Support for more providers (Mistral AI, Cohere, etc.)
- [ ] Model performance benchmarking
- [ ] Cost tracking and analytics
- [ ] Automatic model selection based on task
- [ ] Caching and rate limiting per provider
- [ ] A/B testing different models

## Support

- **Documentation**: See `docs/MODEL_CONFIGURATION.md`
- **Examples**: See `examples/flexible_model_usage.py`
- **Issues**: https://github.com/yourusername/statmate-ai/issues

## Summary

The flexible model system provides:
- ✅ **5 major providers** (OpenAI, Anthropic, Google, Groq, Ollama)
- ✅ **20+ models** to choose from
- ✅ **Local inference** via Ollama (free, private)
- ✅ **Automatic reasoning validation** for tool use
- ✅ **Simple configuration** via environment variables
- ✅ **Backward compatible** with existing code
- ✅ **Production ready** with error handling and logging

No more hardcoded models! 🎉

