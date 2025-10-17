"""Streamlit component for AI model selection."""

import httpx
import streamlit as st


def get_available_models(api_base_url: str) -> dict:
    """Get available models from API.

    Args:
        api_base_url: Base URL of the API.

    Returns:
        Dictionary with models and current selection.
    """
    try:
        response = httpx.get(f'{api_base_url}/models/available?for_tools=true', timeout=10.0)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f'Failed to load models: {e}')
        return {'models': [], 'default_model': None, 'default_provider': None}


def get_current_model(api_base_url: str) -> dict | None:
    """Get current model configuration from API.

    Args:
        api_base_url: Base URL of the API.

    Returns:
        Dictionary with current model config or None.
    """
    try:
        response = httpx.get(f'{api_base_url}/models/current', timeout=10.0)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.warning(f'Failed to load current model: {e}')
        return None


def render_model_selector(api_base_url: str, key_prefix: str = 'model') -> tuple[str | None, str | None]:
    """Render model selection UI component.

    Args:
        api_base_url: Base URL of the API.
        key_prefix: Prefix for session state keys.

    Returns:
        Tuple of (selected_model_name, selected_provider) or (None, None) for default.
    """
    # Get available models
    models_data = get_available_models(api_base_url)
    current_config = get_current_model(api_base_url)

    if not models_data['models']:
        # Check environment mode to show appropriate message
        try:
            import httpx
            env_response = httpx.get(f'{api_base_url}/models/environment', timeout=5.0)
            env_data = env_response.json()
            is_prod = env_data.get('environment') == 'production'
        except Exception:
            is_prod = False

        if is_prod:
            st.warning('⚠️ No AI models configured. Please configure your credentials.')
            st.info(
                '**Quick Setup (PROD Mode):**\n'
                '1. Click "⚙️ Change Keys" button above\n'
                '2. Enter your API keys (OpenAI, Anthropic, Google, Groq, or enable Ollama)\n'
                '3. Click "Configure & Start"'
            )
        else:
            st.warning('⚠️ No AI models configured. Please set API keys in `.env` file.')
            st.info(
                '**Quick Setup (DEV Mode):**\n'
                '1. Copy `.env.example` to `.env`\n'
                '2. Add your API keys (OpenAI, Anthropic, Google, Groq, or enable Ollama)\n'
                '3. Restart the application: `make dev`'
            )
        return None, None

    # Display current default
    if current_config:
        default_model = current_config.get('model_name', 'Unknown')
        default_provider = current_config.get('provider', 'Unknown')
        providers_count = len(current_config.get('available_providers', []))

        col1, col2 = st.columns(2)
        with col1:
            st.metric('Default Model', default_model)
        with col2:
            st.metric('Configured Providers', providers_count)

    # Model selection
    use_custom = st.checkbox(
        'Use specific model for this analysis', key=f'{key_prefix}_use_custom', help='Override the default model'
    )

    if not use_custom:
        return None, None

    # Group models by provider
    models_by_provider = {}
    for model in models_data['models']:
        provider = model['provider']
        if provider not in models_by_provider:
            models_by_provider[provider] = []
        models_by_provider[provider].append(model)

    # Provider selection
    provider_options = list(models_by_provider.keys())
    provider_display = [f'{p.upper()} ({len(models_by_provider[p])} models)' for p in provider_options]

    selected_provider_idx = st.selectbox(
        'Select Provider',
        range(len(provider_options)),
        format_func=lambda i: provider_display[i],
        key=f'{key_prefix}_provider',
    )

    selected_provider = provider_options[selected_provider_idx]

    # Model selection within provider
    provider_models = models_by_provider[selected_provider]
    model_options = [f'{m["display_name"]} - {m["description"][:50]}...' for m in provider_models]

    selected_model_idx = st.selectbox(
        'Select Model',
        range(len(provider_models)),
        format_func=lambda i: model_options[i],
        key=f'{key_prefix}_name',
    )

    selected_model = provider_models[selected_model_idx]

    # Display model details
    with st.expander('Model Details'):
        st.write(f'**Name:** `{selected_model["name"]}`')
        st.write(f'**Provider:** {selected_model["provider"]}')
        st.write(f'**Description:** {selected_model["description"]}')
        st.write(f'**Context Window:** {selected_model["context_window"]:,} tokens')
        st.write(f'**Supports Tools:** {"✅" if selected_model["supports_tools"] else "❌"}')

        capabilities = selected_model.get('capabilities', [])
        if 'reasoning' in capabilities:
            st.success('🧠 Reasoning Model - Excellent for statistical analysis!')
        if 'function_calling' in capabilities:
            st.info('🔧 Function Calling - Can use statistical tools')

    return selected_model['name'], selected_model['provider']


def render_model_info_page(api_base_url: str):
    """Render full model information page.

    Args:
        api_base_url: Base URL of the API.
    """
    st.header('🤖 AI Model Configuration')

    # Current configuration
    current_config = get_current_model(api_base_url)
    if current_config:
        st.subheader('Current Configuration')

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric('Default Model', current_config['model_name'])
        with col2:
            st.metric('Provider', current_config['provider'].upper())
        with col3:
            st.metric('Temperature', f'{current_config["temperature"]:.1f}')

        providers = current_config.get('available_providers', [])
        if providers:
            st.success(f'✅ Configured Providers: {", ".join([p.upper() for p in providers])}')
        else:
            st.error('❌ No providers configured!')

        st.divider()

    # Available models
    st.subheader('Available Models')

    models_data = get_available_models(api_base_url)

    if not models_data['models']:
        # Check environment mode
        try:
            import httpx
            env_response = httpx.get(f'{api_base_url}/models/environment', timeout=5.0)
            env_data = env_response.json()
            is_prod = env_data.get('environment') == 'production'
        except Exception:
            is_prod = False

        st.warning('No models available. Please configure API keys.')

        if is_prod:
            st.info(
                '**Setup Instructions (PROD Mode):**\n\n'
                'Use the "⚙️ Change Keys" button on the main page to enter your credentials.\n\n'
                '**Available Providers:**\n'
                '1. **OpenAI** (GPT-4, GPT-4o)\n'
                '2. **Anthropic** (Claude)\n'
                '3. **Google** (Gemini)\n'
                '4. **Groq** (Free fast inference)\n'
                '5. **Ollama** (Local, FREE!)\n'
            )
        else:
            st.info(
                '**Setup Instructions (DEV Mode):**\n\n'
                '1. **OpenAI:** Set `OPENAI_API_KEY` in `.env`\n'
                '2. **Anthropic:** Set `ANTHROPIC_API_KEY` in `.env`\n'
                '3. **Google:** Set `GOOGLE_API_KEY` in `.env`\n'
                '4. **Groq:** Set `GROQ_API_KEY` in `.env`\n'
                '5. **Ollama (Local):** Set `OLLAMA_ENABLED=True` and run `ollama serve`\n\n'
                'Then restart the application with `make dev`'
            )
        return

    # Group by provider
    models_by_provider = {}
    for model in models_data['models']:
        provider = model['provider']
        if provider not in models_by_provider:
            models_by_provider[provider] = []
        models_by_provider[provider].append(model)

    # Display by provider
    for provider, models in models_by_provider.items():
        with st.expander(f'{provider.upper()} ({len(models)} models)', expanded=True):
            for model in models:
                col1, col2 = st.columns([3, 1])

                with col1:
                    st.markdown(f'**{model["display_name"]}**')
                    st.caption(model['description'])

                with col2:
                    capabilities = model.get('capabilities', [])
                    if 'reasoning' in capabilities:
                        st.badge('🧠 Reasoning', type='primary')
                    if provider == 'ollama':
                        st.badge('🏠 Local', type='success')

                st.caption(f'Model ID: `{model["name"]}` | Context: {model["context_window"]:,} tokens')
                st.divider()

    # Recommendations
    st.subheader('📊 Recommendations for Statistical Analysis')

    rec_col1, rec_col2 = st.columns(2)

    with rec_col1:
        st.markdown('**Best Quality (Reasoning):**')
        st.markdown('- OpenAI o1')
        st.markdown('- Claude 3.7 Sonnet')
        st.markdown('- Gemini 2.0 Flash Thinking')
        st.markdown('- **DeepSeek-R1 🔥 (FREE local)**')

    with rec_col2:
        st.markdown('**Fast & Cost-Effective:**')
        st.markdown('- GPT-4o-mini')
        st.markdown('- Claude 3.5 Haiku')
        st.markdown('- Groq Llama 3.3 70B')
        st.markdown('- **DeepSeek-R1 🔥 (FREE local)**')

    st.info(
        '💡 **Tip:** For best results with statistical analysis, use models with reasoning capabilities. '
        'DeepSeek-R1 via Ollama is highly recommended for local, private, and free reasoning!'
    )
