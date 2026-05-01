# UI Model Integration - Complete Guide

## Overview

The flexible model system has been fully integrated into both the **Backend API** and **Streamlit Frontend UI**, allowing users to:

- ✅ **View available models** from all configured providers
- ✅ **Select models per analysis** via UI or API
- ✅ **See which model was used** in analysis results
- ✅ **Access model configuration page** to explore all options
- ✅ **Use default or custom models** per analysis

## Backend API Changes

### New API Endpoints

#### 1. List Available Models
```
GET /api/v1/models/available?for_tools=true
```

**Response:**
```json
{
  "models": [
    {
      "name": "gpt-4o",
      "provider": "openai",
      "display_name": "GPT-4o",
      "description": "Latest OpenAI model...",
      "context_window": 128000,
      "supports_tools": true,
      "capabilities": ["reasoning", "function_calling", "vision"]
    },
    {
      "name": "deepseek-r1:8b",
      "provider": "ollama",
      "display_name": "DeepSeek-R1 8B (Local)",
      "description": "Local reasoning model...",
      "context_window": 64000,
      "supports_tools": true,
      "capabilities": ["reasoning", "function_calling"]
    }
  ],
  "default_model": "gpt-4o",
  "default_provider": "openai"
}
```

#### 2. Get Current Model Configuration
```
GET /api/v1/models/current
```

**Response:**
```json
{
  "model_name": "gpt-4o",
  "provider": "openai",
  "temperature": 0.0,
  "require_reasoning_for_tools": true,
  "available_providers": ["openai", "anthropic", "ollama"]
}
```

#### 3. Get Model Info
```
GET /api/v1/models/info/{model_name}
```

**Response:**
```json
{
  "name": "deepseek-r1:8b",
  "provider": "ollama",
  "display_name": "DeepSeek-R1 8B (Local)",
  "description": "Local DeepSeek-R1 8B reasoning model via Ollama - competes with o1",
  "context_window": 64000,
  "supports_tools": true,
  "capabilities": ["reasoning", "function_calling"]
}
```

### Updated Analysis API

#### Run Analysis with Model Selection
```
POST /api/v1/analysis/run
```

**Request Body:**
```json
{
  "dataset_id": "uuid-here",
  "selected_columns": ["age", "treatment"],
  "model_name": "deepseek-r1:8b",  // Optional - new!
  "provider": "ollama"              // Optional - new!
}
```

If `model_name` and `provider` are omitted, the system uses the configured default model.

#### Analysis Results Include Model Info
```
GET /api/v1/analysis/{analysis_id}/results
```

**Response includes:**
```json
{
  "id": "analysis-uuid",
  "model_name": "deepseek-r1:8b",  // New!
  "provider": "ollama",             // New!
  "summary": "...",
  "probabilities": {...}
}
```

### Database Changes

**New columns in `analyses` table:**
- `model_name` (VARCHAR(100)) - AI model used
- `provider` (VARCHAR(50)) - Model provider

**Migration Script:**
```sql
-- See: database/migrations/add_model_fields_to_analysis.sql
ALTER TABLE analyses ADD COLUMN model_name VARCHAR(100);
ALTER TABLE analyses ADD COLUMN provider VARCHAR(50);
```

To apply migration:
```bash
sqlite3 database/statmate.db < database/migrations/add_model_fields_to_analysis.sql
```

## Frontend UI Changes

### 1. Model Configuration Page

Access via **"🤖 Model Config"** button in top-right corner.

**Features:**
- View current default model and provider
- See all available models grouped by provider
- Model details (capabilities, context window, etc.)
- Recommendations for statistical analysis
- Setup instructions for unconfigured providers

**How to Use:**
1. Click "🤖 Model Config" button
2. Browse available models
3. See which providers are configured
4. Return to analysis with "← Back to Analysis"

### 2. Model Selector in Analysis Tab

In **Tab 2: Run Analysis**, users can now:

**Default Behavior:**
- Checkbox: "Use specific model for this analysis" (unchecked by default)
- When unchecked, uses the configured default model

**Custom Model Selection:**
1. Check "Use specific model for this analysis"
2. Select **Provider** from dropdown (shows model count)
3. Select **Model** from provider's models
4. View **Model Details** in expander

**Model Details Show:**
- Model name and ID
- Provider
- Description
- Context window size
- Tool/function calling support
- Capabilities (reasoning, vision, etc.)

### 3. Results Display Enhancement

In **Tab 3: View Results**, completed analyses now show:

**Model Information Metrics:**
- 🤖 **Model Used**: e.g., "deepseek-r1:8b" or "gpt-4o"
- **Provider**: e.g., "OLLAMA" or "OPENAI"

Displayed prominently at the top of results, before the summary.

## Usage Examples

### Example 1: Use Default Model

**UI:**
1. Select dataset
2. Click "Run Stat Test" (don't check custom model)
3. Analysis runs with default configured model

**API:**
```bash
curl -X POST http://localhost:8000/api/v1/analysis/run \
  -H "Content-Type: application/json" \
  -d '{
    "dataset_id": "uuid-here"
  }'
```

### Example 2: Use DeepSeek-R1 (Local)

**UI:**
1. Select dataset
2. Check "Use specific model for this analysis"
3. Select **Provider**: "OLLAMA (3 models)"
4. Select **Model**: "DeepSeek-R1 8B (Local)"
5. Click "Run Stat Test"

**API:**
```bash
curl -X POST http://localhost:8000/api/v1/analysis/run \
  -H "Content-Type: application/json" \
  -d '{
    "dataset_id": "uuid-here",
    "model_name": "deepseek-r1:8b",
    "provider": "ollama"
  }'
```

### Example 3: Use Claude for Deep Analysis

**UI:**
1. Select dataset
2. Check "Use specific model for this analysis"
3. Select **Provider**: "ANTHROPIC (4 models)"
4. Select **Model**: "Claude 3.7 Sonnet"
5. View model details (reasoning capabilities)
6. Click "Run Stat Test"

**API:**
```bash
curl -X POST http://localhost:8000/api/v1/analysis/run \
  -H "Content-Type: application/json" \
  -d '{
    "dataset_id": "uuid-here",
    "model_name": "claude-3-7-sonnet-20250219",
    "provider": "anthropic"
  }'
```

## Component Architecture

### Backend Components

```
statmate/api/
├── routes/
│   └── models.py          # Model management endpoints
├── models/
│   ├── model_config.py    # Pydantic models for API
│   └── analysis.py        # Updated with model_name/provider
└── services/
    └── analysis_service.py # Handles model selection logic
```

### Frontend Components

```
statmate/ui/
├── app.py                     # Main Streamlit app (updated)
└── components/
    └── model_selector.py      # Reusable model selection UI
```

### Key Functions

**Backend:**
- `get_available_models()` - List available models from factory
- `get_current_model()` - Get default configuration
- `get_model_info(model_name)` - Get specific model details

**Frontend:**
- `render_model_selector()` - UI component for model selection
- `render_model_info_page()` - Full model configuration page
- `run_analysis(model_name, provider)` - Run with selected model

## Configuration

### Environment Variables

Configure providers in `.env`:

```bash
# Default model
DEFAULT_MODEL_PROVIDER=ollama
DEFAULT_MODEL_NAME=deepseek-r1:8b

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
```

### Model Visibility

Models appear in the UI only if:
1. Provider is configured (API key set or Ollama enabled)
2. Model is listed in `SUPPORTED_MODELS`
3. For analysis: Model supports tools/reasoning

## User Benefits

### For Researchers

1. **Flexibility**: Choose best model for each analysis
2. **Cost Control**: Use free local models for privacy-sensitive data
3. **Quality**: Select premium models for critical analyses
4. **Transparency**: See which model generated results

### For Administrators

1. **Control**: Configure which providers/models are available
2. **Compliance**: Use local models for HIPAA/GDPR compliance
3. **Cost Management**: Monitor and control API usage
4. **Flexibility**: Add new providers without code changes

## Best Practices

### For Users

1. **Use default model** for most analyses (pre-configured for quality)
2. **Use DeepSeek-R1** (Ollama) for:
   - Sensitive data requiring privacy
   - Cost-free local processing
   - Offline/air-gapped environments
3. **Use premium models** (o1, Claude 3.7) for:
   - Complex statistical reasoning
   - Critical research analyses
   - Publication-quality results
4. **Check model info** before selection:
   - Verify reasoning capabilities
   - Check context window size
   - Review provider status

### For Administrators

1. **Set sensible defaults** in `.env`:
   - Use reasoning models by default
   - Configure multiple providers for redundancy
2. **Monitor usage**:
   - Track which models are used
   - Monitor API costs
   - Review analysis success rates
3. **Document choices**:
   - Explain why certain models are recommended
   - Provide guidelines for model selection

## Troubleshooting

### No Models Available

**Problem**: UI shows "No AI models configured"

**Solution:**
1. Set API keys in `.env` file
2. Or enable Ollama: `OLLAMA_ENABLED=True`
3. Restart application
4. Check Model Config page

### Model Not Appearing

**Problem**: Specific model doesn't appear in dropdown

**Solution:**
1. Verify provider is configured (API key set)
2. Check model name in `statmate/core/model_config.py`
3. For Ollama: Verify model is pulled (`ollama list`)

### Analysis Fails with Custom Model

**Problem**: Analysis fails when using custom model

**Solution:**
1. Check model supports tools (reasoning capability)
2. Verify API key is valid
3. For Ollama: Check service is running (`ollama serve`)
4. Review error message in results

## Future Enhancements

Potential additions:
- [ ] Save model preferences per user
- [ ] A/B testing between models
- [ ] Model performance metrics
- [ ] Cost tracking per analysis
- [ ] Automatic model selection based on data type
- [ ] Model comparison feature

## Summary

The flexible model system is now fully integrated:

✅ **Backend API** provides model management endpoints
✅ **Database** stores model info with each analysis
✅ **Frontend UI** allows model selection and configuration
✅ **Results** display which model was used
✅ **Documentation** explains all features

Users can now choose from 20+ models across 5 providers, including free local models via Ollama with DeepSeek-R1! 🎉

