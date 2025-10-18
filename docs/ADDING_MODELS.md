# Adding New AI Models to StatmateAI

## Overview

StatmateAI uses a flexible model configuration system that allows you to add new AI models from any provider. This guide explains how to find available models and add them to your installation.

---

## Where Models Are Defined

All models are defined in:
```
statmate/core/model_config.py
```

Look for the `SUPPORTED_MODELS` dictionary around line 51.

---

## Finding Available Model Names

### OpenAI Models

**Official Documentation:** https://platform.openai.com/docs/models

**How to find models:**
1. Go to https://platform.openai.com/docs/models
2. Look for the "Model" column - this is the exact name to use
3. Check "Context Window" for token limits

**Examples:**
- `gpt-4o` - Latest GPT-4o
- `gpt-4o-mini` - Faster, cheaper GPT-4o
- `gpt-5` - When available
- `o1` - Reasoning model
- `o1-mini` - Faster reasoning
- `o3-mini` - Next generation reasoning

**Test if a model exists:**
```bash
# Using curl
curl https://api.openai.com/v1/models \
  -H "Authorization: Bearer $OPENAI_API_KEY" | jq '.data[].id'

# Or use Python
from openai import OpenAI
client = OpenAI()
models = client.models.list()
for model in models.data:
    print(model.id)
```

---

### Anthropic Models (Claude)

**Official Documentation:** https://docs.anthropic.com/en/docs/about-claude/models

**How to find models:**
1. Go to https://docs.anthropic.com/en/docs/about-claude/models
2. Look for model names in the format `claude-{version}-{variant}-{date}`
3. Check which have "Extended thinking" or "Tool use"

**Examples:**
- `claude-3-7-sonnet-20250219` - Latest with extended thinking
- `claude-3-5-sonnet-latest` - Always latest 3.5 Sonnet
- `claude-3-5-haiku-latest` - Always latest Haiku
- `claude-3-opus-latest` - Always latest Opus

**API Model Names:**
According to the [Pydantic AI documentation](https://ai.pydantic.dev/models/openai/), Anthropic models use the exact model IDs from their API.

---

### Google (Gemini) Models

**Official Documentation:** https://ai.google.dev/gemini-api/docs/models

**How to find models:**
1. Go to https://ai.google.dev/gemini-api/docs/models/gemini
2. Model names are in the "Model code" column
3. Check "Context window" and "Mode"

**Examples:**
- `gemini-2.5-pro-preview` - Latest preview
- `gemini-2.0-flash-exp` - Experimental with thinking
- `gemini-1.5-pro` - Stable with 2M token context
- `gemini-1.5-flash` - Fast and efficient

**List models via API:**
```python
import google.generativeai as genai
genai.configure(api_key="YOUR_API_KEY")

for model in genai.list_models():
    print(f"{model.name} - {model.description}")
```

---

### Groq Models

**Official Documentation:** https://console.groq.com/docs/models

**How to find models:**
1. Go to https://console.groq.com/docs/models
2. Model IDs are in the "Model ID" column
3. All models have function calling

**Examples:**
- `llama-3.3-70b-versatile` - Recommended
- `llama-3.3-70b-specdec` - Fastest inference
- `mixtral-8x7b-32768` - Mixture of experts

---

### Ollama (Local Models)

**Official Models:** https://ollama.ai/library

**How to find models:**
1. Go to https://ollama.ai/library
2. Click on any model to see available tags
3. Format: `{model-name}:{tag}`

**Examples:**
- `deepseek-r1:8b` - Reasoning model (recommended!)
- `llama3.1:8b` - General purpose
- `qwen2.5:7b` - Efficient
- `mistral:7b` - Fast

**List installed models:**
```bash
ollama list
```

**Search for models:**
```bash
ollama search deepseek
ollama search llama
```

---

## How to Add a New Model

### Step 1: Edit `statmate/core/model_config.py`

Add your model to the `SUPPORTED_MODELS` dictionary:

```python
SUPPORTED_MODELS: dict[str, ModelInfo] = {
    # ... existing models ...
    
    # Your new model
    'your-model-name': ModelInfo(
        name='your-model-name',              # Exact API name
        provider=ModelProvider.OPENAI,        # or ANTHROPIC, GOOGLE, GROQ, OLLAMA
        display_name='Your Model Display Name',
        capabilities=[
            ModelCapability.REASONING,        # If it supports reasoning
            ModelCapability.FUNCTION_CALLING, # If it supports tools
            ModelCapability.VISION,           # If it supports images
        ],
        context_window=128000,                # Token limit
        supports_tools=True,                  # Usually True for modern models
        description='Brief description of what makes this model special',
    ),
}
```

### Step 2: Restart the Application

```bash
make kill
make dev
make ui
```

### Step 3: Verify in UI

1. Open http://localhost:8501
2. Go to **"🤖 Model Config"** or check model selector
3. Your new model should appear in the list!

---

## Real Examples

### Example 1: Adding GPT-5

```python
'gpt-5': ModelInfo(
    name='gpt-5',
    provider=ModelProvider.OPENAI,
    display_name='GPT-5',
    capabilities=[
        ModelCapability.REASONING,
        ModelCapability.FUNCTION_CALLING,
        ModelCapability.VISION,
    ],
    context_window=256000,  # Hypothetical - check actual docs
    supports_tools=True,
    description='Latest GPT-5 model (when available)',
),
```

### Example 2: Adding a New Claude Model

```python
'claude-4-opus-20260101': ModelInfo(
    name='claude-4-opus-20260101',
    provider=ModelProvider.ANTHROPIC,
    display_name='Claude 4 Opus',
    capabilities=[
        ModelCapability.REASONING,
        ModelCapability.FUNCTION_CALLING,
        ModelCapability.VISION,
    ],
    context_window=300000,
    supports_tools=True,
    description='Claude 4 Opus - most capable Claude model',
),
```

### Example 3: Adding a Local Ollama Model

```python
'llama3.2:90b': ModelInfo(
    name='llama3.2:90b',
    provider=ModelProvider.OLLAMA,
    display_name='Llama 3.2 90B (Local)',
    capabilities=[
        ModelCapability.REASONING,
        ModelCapability.FUNCTION_CALLING,
    ],
    context_window=128000,
    supports_tools=True,
    description='Local Llama 3.2 90B via Ollama - requires 64GB+ RAM',
),
```

---

## Using Models Without Adding to Config

According to the [Pydantic AI documentation](https://ai.pydantic.dev/models/overview/), you can use **any model name** directly by format:

```python
agent = Agent('openai:gpt-5')           # OpenAI model
agent = Agent('anthropic:claude-4')     # Anthropic model  
agent = Agent('ollama:deepseek-r1:32b') # Ollama model
```

However, for StatmateAI UI to show these models in dropdowns, you need to add them to `SUPPORTED_MODELS`.

---

## Troubleshooting

### Model not appearing in UI

**Solution:**
1. Check spelling in `model_config.py`
2. Restart: `make kill && make dev && make ui`
3. Check browser console for errors

### "Model not found" error

**Solution:**
1. Verify the exact model name from provider docs
2. Check if you have API access to that model
3. Some models require special access/waitlist

### "Provider not configured" error

**Solution:**
Make sure your `.env` has the API key:
```bash
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=...
```

---

## Quick Reference: Model Provider Links

| Provider | Model List | API Docs |
|----------|-----------|----------|
| **OpenAI** | [platform.openai.com/docs/models](https://platform.openai.com/docs/models) | [API Reference](https://platform.openai.com/docs/api-reference) |
| **Anthropic** | [docs.anthropic.com/en/docs/about-claude/models](https://docs.anthropic.com/en/docs/about-claude/models) | [API Reference](https://docs.anthropic.com/en/api) |
| **Google** | [ai.google.dev/gemini-api/docs/models](https://ai.google.dev/gemini-api/docs/models) | [API Reference](https://ai.google.dev/api) |
| **Groq** | [console.groq.com/docs/models](https://console.groq.com/docs/models) | [API Reference](https://console.groq.com/docs/api-reference) |
| **Ollama** | [ollama.ai/library](https://ollama.ai/library) | [GitHub](https://github.com/ollama/ollama) |

---

## Advanced: Custom Providers

If you want to add a completely new provider (e.g., Azure, Hugging Face, etc.):

1. **Add to `ModelProvider` enum** in `model_config.py`:
   ```python
   class ModelProvider(str, Enum):
       OPENAI = 'openai'
       ANTHROPIC = 'anthropic'
       GOOGLE = 'google'
       OLLAMA = 'ollama'
       GROQ = 'groq'
       YOUR_PROVIDER = 'your_provider'  # Add this
   ```

2. **Add provider creation method** in `statmate/core/model_provider.py`:
   ```python
   def _create_your_provider_model(self, model_name: str, ...) -> Model:
       # Implementation here
   ```

3. **Update `create_model()` method** to handle your provider

See [Pydantic AI Custom Models docs](https://ai.pydantic.dev/models/overview/#custom-models) for details.

---

## Need Help?

- Check model provider documentation links above
- Test models directly via their APIs first
- Join our community for support
- Open an issue on GitHub

---

*Last updated: 2025-10-17*

