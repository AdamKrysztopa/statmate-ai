# Model Configuration Guide

## Overview

StatMate AI now supports flexible model selection from multiple AI providers, including local models via Ollama. This allows you to:

- Choose from popular cloud providers (OpenAI, Anthropic, Google, Groq)
- Run models locally using Ollama (free and private)
- Configure multiple providers simultaneously
- Let the system automatically select the best model for tool/reasoning tasks

## Supported Providers

### OpenAI
**Popular Models:**
- `gpt-4o` - Latest with vision (recommended)
- `gpt-4o-mini` - Faster and cheaper
- `gpt-4-turbo` - High performance
- `o1` - Reasoning model
- `o1-mini` - Faster reasoning

**Configuration:**
```bash
OPENAI_API_KEY=your-key-here
DEFAULT_MODEL_PROVIDER=openai
DEFAULT_MODEL_NAME=gpt-4o
```

Get your API key: https://platform.openai.com/api-keys

### Anthropic (Claude)
**Popular Models:**
- `claude-3-7-sonnet-20250219` - Latest with extended thinking
- `claude-3-5-sonnet-20241022` - Recommended
- `claude-3-5-haiku-20241022` - Fast and efficient
- `claude-3-opus-20240229` - Most capable

**Configuration:**
```bash
ANTHROPIC_API_KEY=your-key-here
DEFAULT_MODEL_PROVIDER=anthropic
DEFAULT_MODEL_NAME=claude-3-7-sonnet-20250219
```

Get your API key: https://console.anthropic.com/

### Google (Gemini)
**Popular Models:**
- `gemini-2.0-flash-exp` - Fast with thinking
- `gemini-2.0-flash-thinking-exp-01-21` - Extended thinking
- `gemini-1.5-pro` - Huge context (2M tokens)

**Configuration:**
```bash
GOOGLE_API_KEY=your-key-here
DEFAULT_MODEL_PROVIDER=google
DEFAULT_MODEL_NAME=gemini-2.0-flash-exp
```

Get your API key: https://makersuite.google.com/app/apikey

### Groq (Fast Inference)
**Popular Models:**
- `llama-3.3-70b-versatile` - Recommended
- `llama-3.3-70b-specdec` - Fastest inference

**Configuration:**
```bash
GROQ_API_KEY=your-key-here
DEFAULT_MODEL_PROVIDER=groq
DEFAULT_MODEL_NAME=llama-3.3-70b-versatile
```

Get your API key: https://console.groq.com/keys

### Ollama (Local Models)
**Popular Models:**
- `deepseek-r1:8b` - **DeepSeek-R1 8B reasoning model** (🔥 Recommended! Competes with o1)
- `deepseek-r1:14b` - DeepSeek-R1 14B (higher quality reasoning)
- `deepseek-r1:70b` - DeepSeek-R1 70B (top quality, needs 32GB+ RAM)
- `llama3.1:8b` - Meta Llama 3.1 8B (good balance)
- `llama3.1:70b` - Meta Llama 3.1 70B (more capable)
- `qwen2.5:7b` - Qwen 2.5 7B (efficient)
- `mistral:7b` - Mistral 7B
- `mixtral:8x7b` - Mixture of experts

**Configuration:**
```bash
OLLAMA_ENABLED=True
OLLAMA_BASE_URL=http://localhost:11434/v1
OLLAMA_DEFAULT_MODEL=deepseek-r1:8b  # Reasoning model!
DEFAULT_MODEL_PROVIDER=ollama
DEFAULT_MODEL_NAME=deepseek-r1:8b
```

**Setup:**
1. Install Ollama: https://ollama.ai
2. Pull a model: `ollama pull deepseek-r1:8b` (or `llama3.1:8b`)
3. Start service: `ollama serve`
4. No API key needed!

**Benefits:**
- ✅ Free and private
- ✅ No internet required
- ✅ No API rate limits
- ✅ Full data privacy

**Requirements:**
- 8GB+ RAM for 7B models
- 16GB+ RAM for 13B models
- 32GB+ RAM for 70B models
- GPU optional but recommended

## Environment Variables

Create a `.env` file in the project root with these settings:

### Core Settings
```bash
# Choose default provider
DEFAULT_MODEL_PROVIDER=openai  # openai, anthropic, google, groq, ollama
DEFAULT_MODEL_NAME=gpt-4o

# Model generation settings
MODEL_TEMPERATURE=0.0
MODEL_TOP_P=1.0
MODEL_FREQUENCY_PENALTY=0.0
MODEL_PRESENCE_PENALTY=0.0
MODEL_MAX_TOKENS=

# For tools, always use models with reasoning
REQUIRE_REASONING_MODELS_FOR_TOOLS=True
```

### Provider API Keys
```bash
# OpenAI
OPENAI_API_KEY=your-openai-key
OPENAI_API_BASE=  # Optional custom base URL

# Anthropic
ANTHROPIC_API_KEY=your-anthropic-key
ANTHROPIC_API_BASE=  # Optional

# Google/Gemini
GOOGLE_API_KEY=your-google-key
GEMINI_API_KEY=your-google-key  # Alias

# Groq
GROQ_API_KEY=your-groq-key
GROQ_API_BASE=  # Optional

# Ollama (Local)
OLLAMA_ENABLED=False
OLLAMA_BASE_URL=http://localhost:11434/v1
OLLAMA_DEFAULT_MODEL=llama3.1:8b
```

## Model Selection Tips

### For Statistical Analysis (Reasoning Tasks)
- **Best Cloud:** `o1`, `claude-3-7-sonnet-20250219`, `gemini-2.0-flash-thinking`
- **Good Cloud:** `gpt-4o`, `claude-3-5-sonnet`, `gemini-2.0-flash-exp`
- **Fast Cloud:** `gpt-4o-mini`, `claude-3-5-haiku`, `llama-3.3-70b-versatile`
- **Best Local:** `deepseek-r1:8b` 🔥 (via Ollama - FREE reasoning model!)
- **Good Local:** `deepseek-r1:14b`, `llama3.1:8b`, `qwen2.5:7b` (via Ollama)

### For Tool/Function Calling
All models above support tool calling. The system automatically ensures only reasoning-capable models are used for agent operations.

## Cost Considerations

### Cloud Providers
- **Most Expensive:** `claude-3-opus`, `o1`
- **Mid-range:** `gpt-4o`, `claude-3-7-sonnet`, `gemini-1.5-pro`
- **Affordable:** `gpt-4o-mini`, `claude-3-5-haiku`
- **Very Fast & Cheap:** Groq (`llama-3.3-70b`)
- **Free:** Ollama (local models - **DeepSeek-R1 recommended!**)

## Using Multiple Providers

You can configure multiple providers simultaneously! The system will:
1. Use your default provider/model when not specified
2. Fall back to available models if the requested one is unavailable
3. Automatically select reasoning models for tool/agent tasks

Example multi-provider setup:
```bash
# Primary: OpenAI
OPENAI_API_KEY=your-openai-key
DEFAULT_MODEL_PROVIDER=openai
DEFAULT_MODEL_NAME=gpt-4o

# Backup: Anthropic
ANTHROPIC_API_KEY=your-anthropic-key

# Fast inference: Groq
GROQ_API_KEY=your-groq-key

# Local fallback: Ollama
OLLAMA_ENABLED=True
```

## Programmatic Usage

### In Python Code

```python
from statmate.workflow.model_factory import create_model, get_default_factory

# Use default configured model
model = create_model()

# Specify a model
model = create_model(model_name='gpt-4o')

# Specify provider and model
model = create_model(
    model_name='claude-3-7-sonnet-20250219',
    provider='anthropic'
)

# Get a model for tool calling (ensures reasoning support)
model = create_model(for_tools=True)

# List available models
factory = get_default_factory()
models = factory.list_available_models(for_tools=True)
for model_info in models:
    print(f"{model_info.display_name}: {model_info.description}")
```

### Override API Key Per Request

```python
# Override API key for a specific request
model = create_model(
    model_name='gpt-4o',
    api_key='different-api-key'
)
```

## Best Practices

1. **Start with OpenAI or Anthropic** - Well-tested and reliable
2. **Use Groq for speed** - When you need fast inference
3. **Use Ollama for privacy** - When working with sensitive data
4. **Configure fallbacks** - Set up multiple providers for reliability
5. **Monitor costs** - Use cheaper models for development, premium for production
6. **Test locally first** - Ollama is great for development

## Troubleshooting

### "Provider not configured" Error
- Ensure you've set the API key for the provider in `.env`
- Check that the provider name is correct
- Verify the API key is valid

### Ollama Connection Error
- Ensure Ollama is installed and running: `ollama serve`
- Check the base URL is correct: `http://localhost:11434/v1`
- Verify the model is pulled: `ollama list`

### Model Not Found
- Check the model name is exactly correct
- Ensure the model exists for that provider
- Try listing available models in code

### Out of Memory (Ollama)
- Use a smaller model (7B instead of 70B)
- Close other applications
- Add more RAM or use a cloud provider

## Migration from Hardcoded Models

If you have existing code using hardcoded `OpenAIModel`:

**Before:**
```python
from pydantic_ai.models.openai import OpenAIModel
model = OpenAIModel('gpt-4o')
```

**After:**
```python
from statmate.workflow.model_factory import create_model
model = create_model(model_name='gpt-4o')  # or any other model
```

The system will handle provider selection automatically!

## Support

For issues or questions:
- GitHub Issues: https://github.com/yourusername/statmate-ai/issues
- Documentation: https://docs.statmate-ai.com

