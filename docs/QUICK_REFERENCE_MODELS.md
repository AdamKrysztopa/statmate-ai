# Quick Reference: Flexible Model System

## 🚀 30-Second Setup

### 1. Create `.env` file
```bash
# Choose ONE (or configure multiple):

# OpenAI
OPENAI_API_KEY=sk-your-key
DEFAULT_MODEL_PROVIDER=openai
DEFAULT_MODEL_NAME=gpt-4o

# Anthropic
ANTHROPIC_API_KEY=sk-ant-your-key
DEFAULT_MODEL_PROVIDER=anthropic
DEFAULT_MODEL_NAME=claude-3-7-sonnet-20250219

# Google
GOOGLE_API_KEY=your-key
DEFAULT_MODEL_PROVIDER=google
DEFAULT_MODEL_NAME=gemini-2.0-flash-exp

# Ollama (LOCAL, FREE, REASONING!)
OLLAMA_ENABLED=True
DEFAULT_MODEL_PROVIDER=ollama
DEFAULT_MODEL_NAME=deepseek-r1:8b  # 🔥 Reasoning!
```

### 2. Install & Run
```bash
pip install -e .
bash scripts/run_dev.sh
```

## 💡 Common Use Cases

### Use Default Model
```python
from statmate.workflow.model_factory import create_model
model = create_model()
```

### Use Specific Model
```python
# GPT-4o
model = create_model(model_name='gpt-4o')

# Claude
model = create_model(
    model_name='claude-3-7-sonnet-20250219',
    provider='anthropic'
)

# Local Ollama
model = create_model(
    model_name='llama3.1:8b',
    provider='ollama'
)
```

### In Agents (Recommended)
```python
from statmate.agents.model_helper import create_agent_model_and_settings

model, settings = create_agent_model_and_settings(temperature=0.0)
agent = pearson_agent(model=model, model_settings=settings)
```

### List Available Models
```python
from statmate.workflow.model_factory import get_default_factory

factory = get_default_factory()
models = factory.list_available_models(for_tools=True)
```

## 🏠 Ollama Setup (5 minutes)

```bash
# 1. Install
# Visit https://ollama.ai

# 2. Pull DeepSeek-R1 (reasoning model! 🔥)
ollama pull deepseek-r1:8b

# 3. Start
ollama serve

# 4. Configure .env
OLLAMA_ENABLED=True
DEFAULT_MODEL_PROVIDER=ollama
DEFAULT_MODEL_NAME=deepseek-r1:8b

# Done! FREE reasoning model locally! 🎉
```

## 📊 Model Cheat Sheet

| Provider  | Model             | Cost       | Speed      | Use For               |
| --------- | ----------------- | ---------- | ---------- | --------------------- |
| OpenAI    | gpt-4o            | $$         | Fast       | Production            |
| OpenAI    | gpt-4o-mini       | $          | Very Fast  | Development           |
| OpenAI    | o1                | $$$        | Slow       | Complex reasoning     |
| Anthropic | claude-3-7-sonnet | $$$        | Medium     | Deep analysis         |
| Anthropic | claude-3-5-haiku  | $          | Very Fast  | Quick tasks           |
| Google    | gemini-2.0-flash  | $          | Very Fast  | Good balance          |
| Groq      | llama-3.3-70b     | $          | Ultra Fast | Speed priority        |
| Ollama    | deepseek-r1:8b    | **FREE** 🔥 | Fast       | **Reasoning/offline** |
| Ollama    | llama3.1:8b       | **FREE**   | Fast       | Privacy/general       |

## 🔑 Where to Get API Keys

- **OpenAI**: https://platform.openai.com/api-keys
- **Anthropic**: https://console.anthropic.com/
- **Google**: https://makersuite.google.com/app/apikey
- **Groq**: https://console.groq.com/keys
- **Ollama**: No key needed! Just install.

## 🐛 Troubleshooting

### "Provider not configured"
→ Add API key to `.env`

### "Model not found"
→ Check spelling or use `list_available_models()`

### Ollama connection error
→ Run `ollama serve` first

### Out of memory (Ollama)
→ Use smaller model: `llama3.1:8b` instead of `:70b`

## 📖 Full Documentation

- Setup: `docs/MODEL_CONFIGURATION.md`
- Details: `docs/FLEXIBLE_MODEL_SYSTEM.md`
- Examples: `examples/flexible_model_usage.py`
- Summary: `FLEXIBLE_MODELS_SUMMARY.md`

## ✅ You're Ready!

The system will:
- ✅ Load your config on startup
- ✅ Validate API keys
- ✅ Log available models
- ✅ Use reasoning models for tools
- ✅ Provide clear errors if misconfigured

Start analyzing with your chosen provider! 🎉

